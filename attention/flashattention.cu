#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <math.h>

struct FlashAttentionConfig {
    int batch_size;
    int num_heads;
    int seq_len;
    int head_dim;
    float scale;
};

struct OnlineSoftmaxState {
    float row_max;
    float row_sum;
};

__device__ __forceinline__
OnlineSoftmaxState init_online_softmax() {
    OnlineSoftmaxState state;
    state.row_max = -FLT_MAX;
    state.row_sum = 0.0f;
    return state;
}

__device__ __forceinline__
OnlineSoftmaxState update_online_softmax(
    OnlineSoftmaxState old_state,
    float block_max,
    float block_sum)
{
    OnlineSoftmaxState new_state;
    new_state.row_max = fmaxf(old_state.row_max, block_max);

    float old_scale = expf(old_state.row_max - new_state.row_max);
    float block_scale = expf(block_max - new_state.row_max);
    new_state.row_sum = old_state.row_sum * old_scale + block_sum * block_scale;

    return new_state;
}

// Each CUDA block owns one Q tile for one (batch, head).
// blockIdx.x selects the Q tile along seq_len.
// blockIdx.y selects the attention head.
// blockIdx.z selects the batch.
//
// Q layout: [batch, num_heads, seq_len, head_dim]
//
// each thread block "owns" a BLOCK_M * HEAD_DIM chunk of Q 
// each thread block loads in a BLOCK_N * HEAD_DIM chunk of K, V per ITERATION 
// every iteration, flashAttention calculates (m * head_dim) @ (head_dim * n) chunk of Q @ K^T 
template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void flashattention_forward_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    FlashAttentionConfig cfg)
{
  int batch = blockIdx.z;
  int head = blockIdx.y;
  int q_block = blockIdx.x; // each block processes a BLOCK_M x HEAD_DIM tile of Q 
                            // because HEAD_DIM is usually small (128), we just set BLOCK_M = head_dim = 128 
                            // unlike in tiled GEMM where we have to tile both M and N dimensions, in flash attn 
                            // we only tile the M dimension (seq_len)

  int q_row_start = q_block * BLOCK_M;
  int q_row_end = std::min(q_row_start + BLOCK_M, cfg.seq_len);

  int tid = threadIdx.x;

  __shared__ float q_tile[BLOCK_M][HEAD_DIM];
  __shared__ float k_tile[BLOCK_N][HEAD_DIM];
  __shared__ float v_tile[BLOCK_N][HEAD_DIM];

  __shared__ float scores_tile[BLOCK_M][BLOCK_N];

  OnlineSoftmaxState softmax_state = init_online_softmax();

  // load Q tile once: each thread block is in charge of one Q tile
  for (int idx = tid; idx < BLOCK_M * HEAD_DIM; idx += blockDim.x) {
      int local_q_row = idx / HEAD_DIM;
      int col = idx % HEAD_DIM;
      int global_q_row = q_row_start + local_q_row;

      if (global_q_row < cfg.seq_len)
          // within a q_block: q_block_idx = (row * HEAD_DIM + col)
          // within a head: head_idx = head * seq_len * head_dim + (q_block_idx)
          // within a batch: batch_idx = batch * head * seq_len * head_dim + head_idx
          // within Q: flattened_idx = batch * head * seq_len * head_dim + (head * seq_len * head_dim + (row * head_dim + col))
          q_tile[local_q_row][col] =
              q[((batch * cfg.num_heads + head) * cfg.seq_len + global_q_row) * HEAD_DIM + col]; // factor common terms
  }

  // flash attention loop: each block loads a row of k, v vectors and computes softmax 
  // outer loop loops down rows of k matrix
  for (int k_block_start = 0; k_block_start < cfg.seq_len; k_block_start += BLOCK_N) {

    // inner loop loads an element from the k_block, strided by blockDim.x
    // load K tile
    for (int idx = tid; idx < BLOCK_N * HEAD_DIM; idx += blockDim.x) {
        int local_k_row = idx / HEAD_DIM;
        int col = idx % HEAD_DIM;
        int global_k_row = k_block_start + local_k_row;

        if (global_k_row < cfg.seq_len)
            k_tile[local_k_row][col] =
                k[((batch * cfg.num_heads + head) * cfg.seq_len + global_k_row) * HEAD_DIM + col];
    }

    // load V tile 
    for (int idx = tid; idx < BLOCK_N * HEAD_DIM; idx += blockDim.x) {
        int local_v_row = idx / HEAD_DIM;
        int col = idx % HEAD_DIM;
        int global_v_row = k_block_start + local_v_row;

        if (global_v_row < cfg.seq_len)
            v_tile[local_v_row][col] =
                v[((batch * cfg.num_heads + head) * cfg.seq_len + global_v_row) * HEAD_DIM + col];
    }

    __syncthreads();

    // compute Q @ K^T for a BLOCK_M x head_dim from Q and head_dim x BLOCK_N from K 
    // results in a BLOCK_M x BLOCK_N tile, part of a larger seq_len x seq_len "scores" matrix 
    // production-style mapping:
    // one warp owns one q_row
    // each lane owns multiple k_rows

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    int q_row = warp_id;  // one warp per Q row

    int valid_q_rows = q_row_end - q_row_start;
    int valid_k_rows = min(BLOCK_N, cfg.seq_len - k_block_start);

    if (q_row < valid_q_rows) {
        float local_max = -FLT_MAX;
        float scores[(BLOCK_N + 31) / 32];

        // each lane computes columns: lane, lane+32, lane+64, ...
        #pragma unroll
        for (int kk = lane; kk < BLOCK_N; kk += 32) {
            float score = -FLT_MAX;

            if (kk < valid_k_rows) {
                score = 0.0f;

                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d++) {
                    score += q_tile[q_row][d] * k_tile[kk][d];
                }

                score *= cfg.scale;
            }

            scores[kk / 32] = score;
            local_max = fmaxf(local_max, score);
        }

        // warp reduce max
        float row_max = local_max;
        for (int offset = 16; offset > 0; offset /= 2) {
            row_max = fmaxf(row_max, __shfl_down_sync(0xffffffff, row_max, offset));
        }

        // broadcast final max to all lanes
        row_max = __shfl_sync(0xffffffff, row_max, 0);

        // compute exp(score - row_max), local sum
        float local_sum = 0.0f;
        float probs[(BLOCK_N + 31) / 32];

        #pragma unroll
        for (int kk = lane; kk < BLOCK_N; kk += 32) {
            float p = 0.0f;

            if (kk < valid_k_rows) {
                p = expf(scores[kk / 32] - row_max);
            }

            probs[kk / 32] = p;
            local_sum += p;
        }

        // warp reduce sum
        float row_sum = local_sum;
        for (int offset = 16; offset > 0; offset /= 2) {
            row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
        }

        // broadcast final sum to all lanes
        row_sum = __shfl_sync(0xffffffff, row_sum, 0);

        // now each lane owns normalized softmax probs for its K columns
        #pragma unroll
        for (int kk = lane; kk < BLOCK_N; kk += 32) {
            if (kk < valid_k_rows) {
                float weight = probs[kk / 32] / row_sum;

                // weight = softmax(score[q_row][kk])
                // next step would be: accumulate weight * V[kk]
            }
        }
    }

    __syncthreads();

    // accumulate weighted V into output


  }

}

template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
void launch_flashattention_forward(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int batch_size,
    int num_heads,
    int seq_len,
    float scale,
    cudaStream_t stream = 0)
{
    FlashAttentionConfig cfg;
    cfg.batch_size = batch_size;
    cfg.num_heads = num_heads;
    cfg.seq_len = seq_len;
    cfg.head_dim = HEAD_DIM;
    cfg.scale = scale;

    dim3 block(BLOCK_M);
    dim3 grid(
        (seq_len + BLOCK_M - 1) / BLOCK_M,
        num_heads,
        batch_size
    );

    flashattention_forward_kernel<BLOCK_M, BLOCK_N, HEAD_DIM>
        <<<grid, block, 0, stream>>>(q, k, v, out, cfg);
}

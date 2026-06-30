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

    // slide across col dimension of K matrix 
    //
    for (int i = 0; i < )
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

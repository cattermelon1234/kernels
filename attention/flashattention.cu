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

// Each CUDA block owns one Q tile for one (batch, head).
// blockIdx.x selects the Q tile along seq_len.
// blockIdx.y selects the attention head.
// blockIdx.z selects the batch.
//
// Q layout: [batch, num_heads, seq_len, head_dim]
//
// each thread block owns a BLOCK_M * HEAD_DIM chunk of Q.
// each thread block loads a BLOCK_N * HEAD_DIM chunk of K, V per iteration.
// this educational version maps one warp to one Q row, so BLOCK_M <= 32.
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
    int q_block = blockIdx.x;
    int tid = threadIdx.x;
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    int q_row_start = q_block * BLOCK_M;
    int q_row_end = min(q_row_start + BLOCK_M, cfg.seq_len);
    int valid_q_rows = q_row_end - q_row_start;
    int q_row = warp_id;  // one warp owns one local Q row.

    __shared__ float q_tile[BLOCK_M][HEAD_DIM];
    __shared__ float k_tile[BLOCK_N][HEAD_DIM];
    __shared__ float v_tile[BLOCK_N][HEAD_DIM];

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    float acc[HEAD_DIM];

    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) {
        acc[d] = 0.0f;
    }

    // Load Q once for this block's query tile.
    for (int idx = tid; idx < BLOCK_M * HEAD_DIM; idx += blockDim.x) {
        int local_q_row = idx / HEAD_DIM;
        int col = idx % HEAD_DIM;
        int global_q_row = q_row_start + local_q_row;

        if (global_q_row < cfg.seq_len) {
            q_tile[local_q_row][col] =
                q[((batch * cfg.num_heads + head) * cfg.seq_len + global_q_row) * HEAD_DIM + col];
        }
    }

    __syncthreads();

    for (int k_block_start = 0; k_block_start < cfg.seq_len; k_block_start += BLOCK_N) {
        for (int idx = tid; idx < BLOCK_N * HEAD_DIM; idx += blockDim.x) {
            int local_k_row = idx / HEAD_DIM;
            int col = idx % HEAD_DIM;
            int global_k_row = k_block_start + local_k_row;

            if (global_k_row < cfg.seq_len) {
                k_tile[local_k_row][col] =
                    k[((batch * cfg.num_heads + head) * cfg.seq_len + global_k_row) * HEAD_DIM + col];
                v_tile[local_k_row][col] =
                    v[((batch * cfg.num_heads + head) * cfg.seq_len + global_k_row) * HEAD_DIM + col];
            }
        }

        __syncthreads();

        int valid_k_rows = min(BLOCK_N, cfg.seq_len - k_block_start);

        if (q_row < valid_q_rows) {
            float local_max = -FLT_MAX;
            float scores[(BLOCK_N + 31) / 32];

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

            float tile_max = local_max;
            for (int offset = 16; offset > 0; offset /= 2) {
                tile_max = fmaxf(tile_max, __shfl_down_sync(0xffffffff, tile_max, offset));
            }
            tile_max = __shfl_sync(0xffffffff, tile_max, 0);

            float local_sum = 0.0f;
            float probs[(BLOCK_N + 31) / 32];

            #pragma unroll
            for (int kk = lane; kk < BLOCK_N; kk += 32) {
                float p = 0.0f;

                if (kk < valid_k_rows) {
                    p = expf(scores[kk / 32] - tile_max);
                }

                probs[kk / 32] = p;
                local_sum += p;
            }

            float tile_sum = local_sum;
            for (int offset = 16; offset > 0; offset /= 2) {
                tile_sum += __shfl_down_sync(0xffffffff, tile_sum, offset);
            }
            tile_sum = __shfl_sync(0xffffffff, tile_sum, 0);

            float new_running_max = fmaxf(running_max, tile_max);
            float old_scale = expf(running_max - new_running_max);
            float tile_scale = expf(tile_max - new_running_max);
            float new_running_sum = running_sum * old_scale + tile_sum * tile_scale;

            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                float local_pv = 0.0f;

                #pragma unroll
                for (int kk = lane; kk < BLOCK_N; kk += 32) {
                    if (kk < valid_k_rows) {
                        local_pv += probs[kk / 32] * v_tile[kk][d];
                    }
                }

                for (int offset = 16; offset > 0; offset /= 2) {
                    local_pv += __shfl_down_sync(0xffffffff, local_pv, offset);
                }

                if (lane == 0) {
                    acc[d] = acc[d] * old_scale + local_pv * tile_scale;
                }
            }

            running_max = new_running_max;
            running_sum = new_running_sum;
        }

        __syncthreads();
    }

    if (q_row < valid_q_rows && lane == 0) {
        int global_q_row = q_row_start + q_row;

        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            out[((batch * cfg.num_heads + head) * cfg.seq_len + global_q_row) * HEAD_DIM + d] =
                acc[d] / running_sum;
        }
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
    static_assert(BLOCK_M * 32 <= 1024, "one-warp-per-row mapping requires BLOCK_M <= 32");

    FlashAttentionConfig cfg;
    cfg.batch_size = batch_size;
    cfg.num_heads = num_heads;
    cfg.seq_len = seq_len;
    cfg.head_dim = HEAD_DIM;
    cfg.scale = scale;

    dim3 block(BLOCK_M * 32);
    dim3 grid(
        (seq_len + BLOCK_M - 1) / BLOCK_M,
        num_heads,
        batch_size
    );

    flashattention_forward_kernel<BLOCK_M, BLOCK_N, HEAD_DIM>
        <<<grid, block, 0, stream>>>(q, k, v, out, cfg);
}

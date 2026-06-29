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

// each block is in charge of one "tile" of the q matrix of a single head of a batch 
// (q_block, head, batch)
template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void flashattention_forward_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    FlashAttentionConfig cfg)
{
    int q_block = blockIdx.x;
    int head = blockIdx.y;
    int batch = blockIdx.z;

    int tid = threadIdx.x;
    int q_row = q_block * BLOCK_M + tid;

    if (q_row >= cfg.seq_len) {
        return;
    }

    int bh = batch * cfg.num_heads + head;
    int base = bh * cfg.seq_len * cfg.head_dim;

    const float* q_bh = q + base;
    const float* k_bh = k + base;
    const float* v_bh = v + base;
    float* out_bh = out + base;

    OnlineSoftmaxState softmax = init_online_softmax();
    float acc[HEAD_DIM];

    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) {
        acc[d] = 0.0f;
    }

    for (int k_block = 0; k_block < cfg.seq_len; k_block += BLOCK_N) {
        float scores[BLOCK_N];
        float block_max = -FLT_MAX;

        #pragma unroll
        for (int j = 0; j < BLOCK_N; j++) {
            int k_col = k_block + j;
            float score = -FLT_MAX;

            if (k_col < cfg.seq_len) {
                score = 0.0f;

                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d++) {
                    float q_val = q_bh[q_row * cfg.head_dim + d];
                    float k_val = k_bh[k_col * cfg.head_dim + d];
                    score += q_val * k_val;
                }

                score *= cfg.scale;
            }

            scores[j] = score;
            block_max = fmaxf(block_max, score);
        }

        float block_sum = 0.0f;

        #pragma unroll
        for (int j = 0; j < BLOCK_N; j++) {
            block_sum += expf(scores[j] - block_max);
        }

        OnlineSoftmaxState next_softmax =
            update_online_softmax(softmax, block_max, block_sum);

        float old_out_scale = expf(softmax.row_max - next_softmax.row_max);
        float new_prob_scale = expf(block_max - next_softmax.row_max);

        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            acc[d] *= old_out_scale;
        }

        #pragma unroll
        for (int j = 0; j < BLOCK_N; j++) {
            int k_col = k_block + j;
            float prob = expf(scores[j] - block_max) * new_prob_scale;

            if (k_col < cfg.seq_len) {
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d++) {
                    acc[d] += prob * v_bh[k_col * cfg.head_dim + d];
                }
            }
        }

        softmax = next_softmax;
    }

    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) {
        out_bh[q_row * cfg.head_dim + d] = acc[d] / softmax.row_sum;
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

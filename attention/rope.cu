#include <cuda_runtime.h>
#include <math.h>

// Applies interleaved rotary position embeddings in place to Q and K.
//
// Q/K layout: [batch_size, num_heads, seq_len, HEAD_DIM]
// Rotated pairs: (0, 1), (2, 3), ... (rotary_dim - 2, rotary_dim - 1)
// Dimensions in [rotary_dim, HEAD_DIM) are left unchanged.
template<int HEAD_DIM>
__global__ void rope_kernel(
    float* q,
    float* k,
    int num_heads,
    int seq_len,
    int rotary_dim,
    int position_offset,
    float base)
{
    const int head = blockIdx.x;
    const int batch = blockIdx.y;
    const int pairs_per_token = rotary_dim / 2;
    const int total_pairs = seq_len * pairs_per_token;
    const size_t head_offset =
        (static_cast<size_t>(batch) * num_heads + head) * seq_len * HEAD_DIM;

    for (int idx = threadIdx.x; idx < total_pairs; idx += blockDim.x) {
        const int token_idx = idx / pairs_per_token;
        const int pair_idx = idx % pairs_per_token;
        const int dim_idx = pair_idx * 2;
        const int position = position_offset + token_idx;

        // pair_idx i uses frequency base^(-2i / rotary_dim).
        const float frequency =
            powf(base, -static_cast<float>(dim_idx) / static_cast<float>(rotary_dim));
        const float angle = static_cast<float>(position) * frequency;
        float sin_angle;
        float cos_angle;
        sincosf(angle, &sin_angle, &cos_angle);

        const size_t offset =
            head_offset + static_cast<size_t>(token_idx) * HEAD_DIM + dim_idx;

        const float q0 = q[offset];
        const float q1 = q[offset + 1];
        const float k0 = k[offset];
        const float k1 = k[offset + 1];

        q[offset] = q0 * cos_angle - q1 * sin_angle;
        q[offset + 1] = q0 * sin_angle + q1 * cos_angle;
        k[offset] = k0 * cos_angle - k1 * sin_angle;
        k[offset + 1] = k0 * sin_angle + k1 * cos_angle;
    }
}

template<int HEAD_DIM>
void launch_rope(
    float* q,
    float* k,
    int batch_size,
    int num_heads,
    int seq_len,
    int rotary_dim,
    int position_offset = 0,
    float base = 10000.0f,
    cudaStream_t stream = 0)
{
    static_assert(HEAD_DIM > 0, "HEAD_DIM must be positive");

    if (q == nullptr || k == nullptr || batch_size <= 0 || num_heads <= 0 ||
        seq_len <= 0 || rotary_dim <= 0 || rotary_dim > HEAD_DIM ||
        rotary_dim % 2 != 0 || position_offset < 0 || base <= 0.0f) {
        return;
    }

    constexpr int kBlockSize = 256;
    const dim3 grid(num_heads, batch_size);
    rope_kernel<HEAD_DIM><<<grid, kBlockSize, 0, stream>>>(
        q, k, num_heads, seq_len, rotary_dim, position_offset, base);
}

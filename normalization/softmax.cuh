#pragma once

#include <cuda_runtime.h>

#include <cfloat>

// Reusable temperature-scaled softmax. Each row is split into 256-element
// tiles, then the tile reductions are reduced once more for the whole row.
namespace softmax_cuda {

constexpr int kBlockSize = 256;

__device__ __forceinline__ float warp_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

template <int BLOCK_SIZE>
__device__ __forceinline__ float block_max(float value) {
    __shared__ float warp_values[BLOCK_SIZE / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_max(value);
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();
    value = threadIdx.x < BLOCK_SIZE / 32 ? warp_values[lane] : -FLT_MAX;
    if (warp == 0) value = warp_max(value);
    return value;
}

template <int BLOCK_SIZE>
__device__ __forceinline__ float block_sum(float value) {
    __shared__ float warp_values[BLOCK_SIZE / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();
    value = threadIdx.x < BLOCK_SIZE / 32 ? warp_values[lane] : 0.0f;
    if (warp == 0) value = warp_sum(value);
    return value;
}

template <int BLOCK_SIZE>
__global__ void partial_max_kernel(const float* logits, float* partial_max,
                                   int cols, float temperature) {
    const int row = blockIdx.y;
    const int tile = blockIdx.x;
    const int col = tile * BLOCK_SIZE + threadIdx.x;
    float value = col < cols ? logits[row * cols + col] / temperature : -FLT_MAX;
    value = block_max<BLOCK_SIZE>(value);
    if (threadIdx.x == 0) partial_max[row * gridDim.x + tile] = value;
}

template <int BLOCK_SIZE>
__global__ void finalize_max_kernel(const float* partial_max, float* row_max, int tiles) {
    const int row = blockIdx.x;
    float value = -FLT_MAX;
    for (int tile = threadIdx.x; tile < tiles; tile += BLOCK_SIZE) {
        value = fmaxf(value, partial_max[row * tiles + tile]);
    }
    value = block_max<BLOCK_SIZE>(value);
    if (threadIdx.x == 0) row_max[row] = value;
}

template <int BLOCK_SIZE>
__global__ void exp_partial_sum_kernel(const float* logits, const float* row_max,
                                       float* output, float* partial_sum,
                                       int cols, float temperature) {
    const int row = blockIdx.y;
    const int tile = blockIdx.x;
    const int col = tile * BLOCK_SIZE + threadIdx.x;
    float value = 0.0f;
    if (col < cols) {
        value = expf(logits[row * cols + col] / temperature - row_max[row]);
        output[row * cols + col] = value;
    }
    value = block_sum<BLOCK_SIZE>(value);
    if (threadIdx.x == 0) partial_sum[row * gridDim.x + tile] = value;
}

template <int BLOCK_SIZE>
__global__ void finalize_sum_kernel(const float* partial_sum, float* row_sum, int tiles) {
    const int row = blockIdx.x;
    float value = 0.0f;
    for (int tile = threadIdx.x; tile < tiles; tile += BLOCK_SIZE) {
        value += partial_sum[row * tiles + tile];
    }
    value = block_sum<BLOCK_SIZE>(value);
    if (threadIdx.x == 0) row_sum[row] = value;
}

__global__ void normalize_kernel(float* output, const float* row_sum, int cols) {
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) output[row * cols + col] /= row_sum[row];
}

// workspace needs rows * ceil(cols / kBlockSize) floats; row_max and row_sum
// each need rows floats.
inline void softmax_temperature(const float* logits, float* output, int rows, int cols,
                                float temperature, float* workspace,
                                float* row_max, float* row_sum) {
    if (rows <= 0 || cols <= 0 || temperature <= 0.0f) return;
    const int tiles = (cols + kBlockSize - 1) / kBlockSize;
    const dim3 tiled_grid(tiles, rows);
    partial_max_kernel<kBlockSize><<<tiled_grid, kBlockSize>>>(logits, workspace, cols, temperature);
    finalize_max_kernel<kBlockSize><<<rows, kBlockSize>>>(workspace, row_max, tiles);
    exp_partial_sum_kernel<kBlockSize><<<tiled_grid, kBlockSize>>>(logits, row_max, output, workspace, cols, temperature);
    finalize_sum_kernel<kBlockSize><<<rows, kBlockSize>>>(workspace, row_sum, tiles);
    normalize_kernel<<<tiled_grid, kBlockSize>>>(output, row_sum, cols);
}

}  // namespace softmax_cuda

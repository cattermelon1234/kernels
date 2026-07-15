#pragma once

#include <cuda_runtime.h>

#include <cfloat>

#include "../include/reductions.cuh"

// Reusable temperature-scaled softmax. Each row is split into 256-element
// tiles, then the tile reductions are reduced once more for the whole row.
namespace kernels {

constexpr int kSoftmaxBlockSize = 256;

template <int BLOCK_SIZE>
__global__ void compute_tile_maxes_kernel(const float* logits, float* tile_maxes,
                                          int cols, float temperature) {
    const int row = blockIdx.y;
    const int tile = blockIdx.x;
    const int col = tile * BLOCK_SIZE + threadIdx.x;
    float value = col < cols ? logits[row * cols + col] / temperature : -FLT_MAX;
    value = block_reduce_max<BLOCK_SIZE>(value);
    if (threadIdx.x == 0) tile_maxes[row * gridDim.x + tile] = value;
}

template <int BLOCK_SIZE>
__global__ void compute_exp_and_tile_sums_kernel(const float* logits, const float* row_maxes,
                                                 float* output, float* tile_sums,
                                                 int cols, float temperature) {
    const int row = blockIdx.y;
    const int tile = blockIdx.x;
    const int col = tile * BLOCK_SIZE + threadIdx.x;
    float value = 0.0f;
    if (col < cols) {
        value = expf(logits[row * cols + col] / temperature - row_maxes[row]);
        output[row * cols + col] = value;
    }
    value = block_reduce_sum<BLOCK_SIZE>(value);
    if (threadIdx.x == 0) tile_sums[row * gridDim.x + tile] = value;
}

__global__ void normalize_rows_kernel(float* output, const float* row_sums, int cols) {
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) output[row * cols + col] /= row_sums[row];
}

// workspace needs rows * ceil(cols / kSoftmaxBlockSize) floats; row_max and row_sum
// each need rows floats.
inline void softmax_temperature(const float* logits, float* output, int rows, int cols,
                                float temperature, float* workspace,
                                float* row_max, float* row_sum) {
    if (rows <= 0 || cols <= 0 || temperature <= 0.0f) return;
    const int tiles = (cols + kSoftmaxBlockSize - 1) / kSoftmaxBlockSize;
    const dim3 tiled_grid(tiles, rows);
    compute_tile_maxes_kernel<kSoftmaxBlockSize><<<tiled_grid, kSoftmaxBlockSize>>>(logits, workspace, cols, temperature);
    reduce_segment_maxes<kSoftmaxBlockSize><<<rows, kSoftmaxBlockSize>>>(
        workspace, row_max, tiles);
    compute_exp_and_tile_sums_kernel<kSoftmaxBlockSize><<<tiled_grid, kSoftmaxBlockSize>>>(logits, row_max, output, workspace, cols, temperature);
    reduce_segment_sums<kSoftmaxBlockSize><<<rows, kSoftmaxBlockSize>>>(
        workspace, row_sum, tiles);
    normalize_rows_kernel<<<tiled_grid, kSoftmaxBlockSize>>>(output, row_sum, cols);
}

}  // namespace kernels

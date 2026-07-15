#pragma once

#include <cuda_runtime.h>

#include "../include/reductions.cuh"

namespace kernels {

constexpr int kRmsNormBlockSize = 256;

template <int BLOCK_SIZE>
__global__ void rms_norm_kernel(const float* input,
                                const float* weight,
                                float* output,
                                int hidden_size,
                                float epsilon) {
    const int row = blockIdx.x;
    const float* row_input = input + row * hidden_size;
    float* row_output = output + row * hidden_size;

    float local_sum_of_squares = 0.0f;

    for (int col = threadIdx.x; col < hidden_size; col += BLOCK_SIZE) {
        const float value = row_input[col];
        local_sum_of_squares += value * value;
    }

    const float sum_of_squares =
        block_reduce_sum<BLOCK_SIZE>(local_sum_of_squares);

    __shared__ float inverse_rms;

    if (threadIdx.x == 0) {
        inverse_rms = rsqrtf(sum_of_squares / hidden_size + epsilon);
    }

    __syncthreads();

    for (int col = threadIdx.x; col < hidden_size; col += BLOCK_SIZE) {
        row_output[col] = row_input[col] * inverse_rms * weight[col];
    }
}

inline void rms_norm(const float* input,
                     const float* weight,
                     float* output,
                     int rows,
                     int hidden_size,
                     float epsilon,
                     cudaStream_t stream = 0) {
    if (rows <= 0 || hidden_size <= 0 || epsilon < 0.0f) return;

    rms_norm_kernel<kRmsNormBlockSize>
        <<<rows, kRmsNormBlockSize, 0, stream>>>(
            input, weight, output, hidden_size, epsilon);
}

}  // namespace kernels

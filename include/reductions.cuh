#pragma once

#include <cuda_runtime.h>

#include <cfloat>

namespace kernels {

__device__ __forceinline__ float warp_reduce_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_max(float value) {
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must contain whole warps");
    static_assert(BLOCK_SIZE <= 1024, "BLOCK_SIZE exceeds the CUDA block limit");
    __shared__ float warp_values[BLOCK_SIZE / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_reduce_max(value);
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();
    value = threadIdx.x < BLOCK_SIZE / 32 ? warp_values[lane] : -FLT_MAX;
    if (warp == 0) value = warp_reduce_max(value);
    __syncthreads();
    return value;
}

template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float value) {
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must contain whole warps");
    static_assert(BLOCK_SIZE <= 1024, "BLOCK_SIZE exceeds the CUDA block limit");
    __shared__ float warp_values[BLOCK_SIZE / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_reduce_sum(value);
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();
    value = threadIdx.x < BLOCK_SIZE / 32 ? warp_values[lane] : 0.0f;
    if (warp == 0) value = warp_reduce_sum(value);
    __syncthreads();
    return value;
}

// Reduces partial_results[segment][partial] to one sum per segment. Launch
// one block per segment; threads stride over that segment's partial results.
template <int BLOCK_SIZE>
__global__ void reduce_segment_sums(const float* partial_results,
                                    float* segment_results,
                                    int partials_per_segment) {
    const int segment = blockIdx.x;
    float value = 0.0f;

    for (int partial = threadIdx.x;
         partial < partials_per_segment;
         partial += BLOCK_SIZE) {
        value += partial_results[segment * partials_per_segment + partial];
    }

    value = block_reduce_sum<BLOCK_SIZE>(value);
    if (threadIdx.x == 0) segment_results[segment] = value;
}

// Reduces partial_results[segment][partial] to one maximum per segment. Launch
// one block per segment; threads stride over that segment's partial results.
template <int BLOCK_SIZE>
__global__ void reduce_segment_maxes(const float* partial_results,
                                     float* segment_results,
                                     int partials_per_segment) {
    const int segment = blockIdx.x;
    float value = -FLT_MAX;

    for (int partial = threadIdx.x;
         partial < partials_per_segment;
         partial += BLOCK_SIZE) {
        value = fmaxf(value,
                      partial_results[segment * partials_per_segment + partial]);
    }

    value = block_reduce_max<BLOCK_SIZE>(value);
    if (threadIdx.x == 0) segment_results[segment] = value;
}

}  // namespace kernels

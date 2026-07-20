#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

namespace kernels {

//--------------------------------------------------------------
// Device helpers
//--------------------------------------------------------------

__device__ __forceinline__
float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__
float silu(float x) {
  return sigmoid(x) * x;
}

//--------------------------------------------------------------
// CUDA kernel
//--------------------------------------------------------------

template<
    int BLOCK_SIZE,
    int VEC_SIZE = 4
>
__global__ void fused_swiglu_kernel(
    const float* gate,
    const float* up,
    float* out,
    int num_elements) {
  int 
}

//--------------------------------------------------------------
// Launcher
//--------------------------------------------------------------

template<
    int BLOCK_SIZE = 256,
    int VEC_SIZE = 4
>
void launch_fused_swiglu(
    const float* gate,
    const float* up,
    float* out,
    int num_elements,
    cudaStream_t stream = 0);

} // namespace kernels

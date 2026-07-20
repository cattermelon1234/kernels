#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <math.h>

namespace kernels {

__device__ __forceinline__
float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__
float silu(float x) {
  return sigmoid(x) * x;
}

template<int VEC_SIZE>
struct SwigluVector;

template<>
struct SwigluVector<1> {
  static constexpr int ALIGNMENT = alignof(float);

  __device__ __forceinline__
  static void apply(const float* gate, const float* up, float* out, int i) {
    out[i] = silu(gate[i]) * up[i];
  }
};

template<>
struct SwigluVector<2> {
  static constexpr int ALIGNMENT = alignof(float2);

  __device__ __forceinline__
  static void apply(const float* gate, const float* up, float* out, int i) {
    const float2 g = reinterpret_cast<const float2*>(gate)[i];
    const float2 u = reinterpret_cast<const float2*>(up)[i];
    reinterpret_cast<float2*>(out)[i] = make_float2(
        silu(g.x) * u.x,
        silu(g.y) * u.y);
  }
};

template<>
struct SwigluVector<4> {
  static constexpr int ALIGNMENT = alignof(float4);

  __device__ __forceinline__
  static void apply(const float* gate, const float* up, float* out, int i) {
    const float4 g = reinterpret_cast<const float4*>(gate)[i];
    const float4 u = reinterpret_cast<const float4*>(up)[i];
    reinterpret_cast<float4*>(out)[i] = make_float4(
        silu(g.x) * u.x,
        silu(g.y) * u.y,
        silu(g.z) * u.z,
        silu(g.w) * u.w);
  }
};

template<
    int BLOCK_SIZE,
    int VEC_SIZE = 4
>
__global__ void fused_swiglu_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int num_elements) {
  static_assert(VEC_SIZE == 1 || VEC_SIZE == 2 || VEC_SIZE == 4,
                "VEC_SIZE must be 1, 2, or 4");

  const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int thread_stride = blockDim.x * gridDim.x;
  const int num_vecs = num_elements / VEC_SIZE;

  for (int vec_idx = thread_idx; vec_idx < num_vecs;
       vec_idx += thread_stride) {
    SwigluVector<VEC_SIZE>::apply(gate, up, out, vec_idx);
  }

  // Process the at most VEC_SIZE - 1 elements that cannot form a full vector.
  const int tail_idx = num_vecs * VEC_SIZE + thread_idx;
  if (tail_idx < num_elements) {
    out[tail_idx] = silu(gate[tail_idx]) * up[tail_idx];
  }
}

template<int VEC_SIZE>
bool swiglu_pointers_are_aligned(
    const float* gate,
    const float* up,
    const float* out) {
  const uintptr_t alignment = SwigluVector<VEC_SIZE>::ALIGNMENT;
  return reinterpret_cast<uintptr_t>(gate) % alignment == 0 &&
         reinterpret_cast<uintptr_t>(up) % alignment == 0 &&
         reinterpret_cast<uintptr_t>(out) % alignment == 0;
}

template<int BLOCK_SIZE, int VEC_SIZE>
void launch_fused_swiglu_aligned(
    const float* gate,
    const float* up,
    float* out,
    int num_elements,
    cudaStream_t stream) {
  const int num_work_items =
      (num_elements + VEC_SIZE - 1) / VEC_SIZE;
  const int grid = (num_work_items + BLOCK_SIZE - 1) / BLOCK_SIZE;

  fused_swiglu_kernel<BLOCK_SIZE, VEC_SIZE>
      <<<grid, BLOCK_SIZE, 0, stream>>>(
      gate,
      up,
      out,
      num_elements);
}

template<int BLOCK_SIZE, int VEC_SIZE>
void launch_fused_swiglu_dispatch(
    const float* gate,
    const float* up,
    float* out,
    int num_elements,
    cudaStream_t stream) {
  if (swiglu_pointers_are_aligned<VEC_SIZE>(gate, up, out)) {
    launch_fused_swiglu_aligned<BLOCK_SIZE, VEC_SIZE>(
        gate, up, out, num_elements, stream);
  } else {
    launch_fused_swiglu_aligned<BLOCK_SIZE, 1>(
        gate, up, out, num_elements, stream);
  }
}

template<
    int BLOCK_SIZE = 256,
    int VEC_SIZE = 4
>
void launch_fused_swiglu(
    const float* gate,
    const float* up,
    float* out,
    int num_elements,
    cudaStream_t stream = 0) {

    if (gate == nullptr || up == nullptr || out == nullptr)
        return;

    if (num_elements <= 0)
        return;

    static_assert(BLOCK_SIZE > 0, "BLOCK_SIZE must be positive");
    static_assert(VEC_SIZE == 1 || VEC_SIZE == 2 || VEC_SIZE == 4,
                  "VEC_SIZE must be 1, 2, or 4");

    launch_fused_swiglu_dispatch<BLOCK_SIZE, VEC_SIZE>(
        gate, up, out, num_elements, stream);
}

} // namespace kernels

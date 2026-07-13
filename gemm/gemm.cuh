#pragma once

#include <cuda_runtime.h>

namespace kernels::gemm {

void launch_gemm(const float* A, const float* B, float* C,
                 int M, int N, int K, cudaStream_t stream = 0);

}

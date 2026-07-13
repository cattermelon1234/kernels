#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include "../tests/gemm_benchmark.cuh"

#define GEMM_DISABLE_STANDALONE_MAIN
#include "../gemm/gemm.cu"

void run_hyperoptimized_gemm_benchmark() {
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;
    const int iters = 50;

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);

    for (int i = 0; i < M * K; ++i) {
        h_A[i] = 1.0f;
    }

    for (int i = 0; i < K * N; ++i) {
        h_B[i] = 2.0f;
    }

    float* d_A;
    float* d_B;
    float* d_C;

    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    const double flops_per_iter = 2.0 * M * N * K;
    const double bytes_per_iter = 3.0 * M * N * sizeof(float);
    const size_t c_bytes = M * N * sizeof(float);

    dim3 block(32, WARPS_PER_BLOCK);
    dim3 grid((N + B_N - 1) / B_N, (M + B_M - 1) / B_M);

    RUN_BENCHMARK("Hyperoptimized GEMM", GEMM, block, grid);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return;
}

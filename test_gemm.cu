#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include "benchmark.cuh"
#include "gemm_kernels.cuh"

#define RUN_BENCHMARK(LABEL, KERNEL, BLOCK_CFG, GRID_CFG) do { \
    cudaMemset(d_C, 0, c_bytes); \
    KERNEL<<<GRID_CFG, BLOCK_CFG>>>(d_A, d_B, d_C, M, N, K); \
    cudaDeviceSynchronize(); \
    BenchmarkTimer timer; \
    timer.begin(); \
    for (int i = 0; i < iters; ++i) { \
        KERNEL<<<GRID_CFG, BLOCK_CFG>>>(d_A, d_B, d_C, M, N, K); \
    } \
    timer.end(); \
    std::cout << LABEL << "\n"; \
    benchmark_report(flops_per_iter, bytes_per_iter, iters, timer.elapsed_ms()); \
    std::cout << "\n"; \
} while (0)

int main() {
    const int M = 128;
    const int N = 128;
    const int K = 128;
    const int iters = 1000;

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);

    for (int i = 0; i < M * K; ++i) {
        h_A[i] = 1.0f;
    }

    for (int i = 0; i < K * N; ++i) {
        h_B[i] = 2.0f;
    }

    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    const double flops_per_iter = 2.0 * M * N * K;
    const double bytes_per_iter = 3.0 * M * N * sizeof(float);
    const size_t c_bytes = M * N * sizeof(float);

    dim3 block_naive(NAIVE_TILE, NAIVE_TILE);
    dim3 grid_naive((N + NAIVE_TILE - 1) / NAIVE_TILE, (M + NAIVE_TILE - 1) / NAIVE_TILE);

    dim3 block_opt(OPT_TILE / REG, OPT_TILE / REG);
    dim3 grid_opt((N + OPT_TILE - 1) / OPT_TILE, (M + OPT_TILE - 1) / OPT_TILE);

    dim3 block_tensor(32, TENSOR_WARPS_PER_BLOCK);
    dim3 grid_tensor((N + TENSOR_B_N - 1) / TENSOR_B_N, (M + TENSOR_B_M - 1) / TENSOR_B_M);

    RUN_BENCHMARK("Normal tiled GEMM", gemm_tiled_kernel, block_naive, grid_naive);
    RUN_BENCHMARK("Optimized GEMM", gemm_optimized_kernel, block_opt, grid_opt);
    RUN_BENCHMARK("Tensor core GEMM", gemm_tensor_core_kernel, block_tensor, grid_tensor);

    cudaMemcpy(h_C.data(), d_C, c_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

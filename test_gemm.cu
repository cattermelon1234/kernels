#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include "tests/gemm_benchmark.cuh"
#include "gemm_kernels.cuh"

int main() {
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    const int iters = 100;

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

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << " compute capability "
              << prop.major << "." << prop.minor << "\n\n";

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

    if (prop.major >= 7) {
        RUN_BENCHMARK("Tensor core GEMM", gemm_tensor_core_kernel, block_tensor, grid_tensor);
    } else {
        std::cout << "Tensor core GEMM\n";
        std::cout << "skipped: compute capability < 7.0\n\n";
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}

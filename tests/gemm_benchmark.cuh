#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#include "../include/benchmark.cuh"

inline void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << file << ":" << line << " CUDA error in " << expr
                  << ": " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

#define CUDA_CHECK(expr) cuda_check((expr), #expr, __FILE__, __LINE__)

#define RUN_BENCHMARK(LABEL, KERNEL, BLOCK_CFG, GRID_CFG) do { \
    CUDA_CHECK(cudaMemset(d_C, 0, c_bytes)); \
    KERNEL<<<GRID_CFG, BLOCK_CFG>>>(d_A, d_B, d_C, M, N, K); \
    CUDA_CHECK(cudaPeekAtLastError()); \
    CUDA_CHECK(cudaDeviceSynchronize()); \
    BenchmarkTimer timer; \
    timer.begin(); \
    for (int i = 0; i < iters; ++i) { \
        KERNEL<<<GRID_CFG, BLOCK_CFG>>>(d_A, d_B, d_C, M, N, K); \
        CUDA_CHECK(cudaPeekAtLastError()); \
    } \
    timer.end(); \
    std::cout << LABEL << "\n"; \
    benchmark_report(flops_per_iter, bytes_per_iter, iters, timer.elapsed_ms()); \
    std::cout << "\n"; \
} while (0)

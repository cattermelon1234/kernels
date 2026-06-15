#pragma once

#include <cuda_runtime.h>
#include <iostream>

struct BenchmarkTimer {
    BenchmarkTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~BenchmarkTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void begin() {
        cudaEventRecord(start);
    }

    void end() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    }

    float elapsed_ms() const {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }

    cudaEvent_t start;
    cudaEvent_t stop;
};

inline void benchmark_report(
    double flops_per_iter,
    double bytes_per_iter,
    int iters,
    float ms)
{
    double total_flops = flops_per_iter * iters;
    double total_bytes = bytes_per_iter * iters;

    std::cout << "GFLOP/s: "
              << (total_flops / (ms * 1e6))
              << "\n";
    std::cout << "GB/s: "
              << (total_bytes / (ms * 1e6))
              << "\n";
}

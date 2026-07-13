#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <cstdlib>

#define TILE 16

__global__ void transpose(const float* A, float* B, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        B[col * rows + row] = A[row * cols + col];
    }
}

__global__ void transposeTiled(const float* A, float* B, int M, int N) {
    __shared__ float tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (y < M && x < N) {
        tile[threadIdx.y][threadIdx.x] = A[y * N + x];
    }

    __syncthreads();

    int out_x = blockIdx.y * TILE + threadIdx.x;
    int out_y = blockIdx.x * TILE + threadIdx.y;

    if (out_y < N && out_x < M) {
        B[out_y * M + out_x] = tile[threadIdx.x][threadIdx.y];
    }
}

void demo_transpose() {
    constexpr int M = 4096;
    constexpr int N = 4096;

    size_t bytes = (size_t)M * N * sizeof(float);

    float* h_A = (float*)malloc(bytes);

    for (size_t i = 0; i < (size_t)M * N; ++i) {
        h_A[i] = static_cast<float>(i);
    }

    float *d_A, *d_B;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid(
        (N + TILE - 1) / TILE,
        (M + TILE - 1) / TILE
    );

    constexpr int ITERS = 100;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < 10; ++i) {
        transpose<<<grid, block>>>(d_A, d_B, M, N);
        transposeTiled<<<grid, block>>>(d_A, d_B, M, N);
    }

    cudaDeviceSynchronize();

    cudaEventRecord(start);

    for (int i = 0; i < ITERS; ++i) {
        transpose<<<grid, block>>>(d_A, d_B, M, N);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float naive_ms;
    cudaEventElapsedTime(&naive_ms, start, stop);
    naive_ms /= ITERS;

    cudaEventRecord(start);

    for (int i = 0; i < ITERS; ++i) {
        transposeTiled<<<grid, block>>>(d_A, d_B, M, N);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float tiled_ms;
    cudaEventElapsedTime(&tiled_ms, start, stop);
    tiled_ms /= ITERS;

    double gb_moved = (2.0 * bytes) / 1e9;

    double naive_bw =
        gb_moved / (naive_ms / 1000.0);

    double tiled_bw =
        gb_moved / (tiled_ms / 1000.0);

    std::cout << "Matrix: "
              << M << " x " << N << "\n\n";

    std::cout << "Naive transpose:\n";
    std::cout << "  Time      : "
              << naive_ms << " ms\n";
    std::cout << "  Bandwidth : "
              << naive_bw << " GB/s\n\n";

    std::cout << "Tiled transpose:\n";
    std::cout << "  Time      : "
              << tiled_ms << " ms\n";
    std::cout << "  Bandwidth : "
              << tiled_bw << " GB/s\n\n";

    std::cout << "Speedup: "
              << naive_ms / tiled_ms
              << "x\n";

    cudaFree(d_A);
    cudaFree(d_B);

    free(h_A);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return;
}

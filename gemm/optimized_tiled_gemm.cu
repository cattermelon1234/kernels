#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define TILE 64
#define BLOCK 16
#define REG 4

// multiplying MxK and KxN matrices
__global__ void GEMM(const float* A, const float* B, float* C,
    int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // block row, block col represent the real row and col idx of the start of our tile
  int block_row = blockIdx.y * TILE;
  int block_col = blockIdx.x * TILE;

  // base row and col represent the real row and col idx of the start of our 4x4 tile that we want to load into register
  int base_row = block_row + ty * REG; 
  int base_col = block_col + tx * REG;

  // note that these are indices of the final output tile we want to compute, 
  // not the indices of the tile from A or B we want to load into shared memory.

  float sum[REG][REG];

  #pragma unroll
  for (int i = 0; i < REG; i++) {
      #pragma unroll
      for (int j = 0; j < REG; j++) {
          sum[i][j] = 0.0f;
      }
  }


  // load the 64x64 tile of A and B into shared memory 
  for (int t = 0; t < (K + TILE - 1) / TILE; ++t) { 
    // load one 4x4 fragment of A and B per thread into shared memory
    for (int i = 0; i < REG; ++i) {
      for (int j = 0; j < REG; ++j) {
        int a_row = base_row + i; 
        int a_col = t * TILE + tx * REG + j;

        int b_col = base_col + j;
        int b_row = t * TILE + ty * REG + i;

        if (a_row < M && a_col < K) {
            As[ty * REG + i][tx * REG + j] = A[a_row * K + a_col];
        } else {
            As[ty * REG + i][tx * REG + j] = 0.0f;
        }

        if (b_row < K && b_col < N) {
            Bs[ty * REG + i][tx * REG + j] = B[b_row * N + b_col];
        } else {
            Bs[ty * REG + i][tx * REG + j] = 0.0f;
        }
      }
    }

    __syncthreads();

    for (int k_inner = 0; k_inner < TILE; k_inner++) {
        float a_frag[REG];
        float b_frag[REG];

        #pragma unroll
        for (int i = 0; i < REG; i++) {
            a_frag[i] = As[ty * REG + i][k_inner];
        }

        #pragma unroll
        for (int j = 0; j < REG; j++) {
            b_frag[j] = Bs[k_inner][tx * REG + j];
        }

        #pragma unroll
        for (int i = 0; i < REG; i++) {
            #pragma unroll
            for (int j = 0; j < REG; j++) {
                sum[i][j] += a_frag[i] * b_frag[j];
            }
        }
    }

    __syncthreads();
  }

  for (int i = 0; i < REG; ++i) {
    for (int j = 0; j < REG; ++j) {
      int row = base_row + i;
      int col = base_col + j;
      if (row < M && col < N) {
        C[row * N + col] = sum[i][j];
      }
    }
  }
}

#ifndef GEMM_DISABLE_STANDALONE_MAIN
void demo_optimized_gemm() {
    const int M = 64;
    const int K = 64;
    const int N = 64;

    // Host matrices
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);

    // Initialize A and B
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = 1.0f;
    }

    for (int i = 0; i < K * N; ++i) {
        h_B[i] = 2.0f;
    }

    // Device matrices
    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Copy inputs to GPU
    cudaMemcpy(
        d_A,
        h_A.data(),
        M * K * sizeof(float),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_B,
        h_B.data(),
        K * N * sizeof(float),
        cudaMemcpyHostToDevice
    );

    // Launch kernel
    dim3 block(TILE / REG, TILE / REG);

    dim3 grid(
        (N + TILE - 1) / TILE,
        (M + TILE - 1) / TILE
    );

    GEMM<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(
        h_C.data(),
        d_C,
        M * N * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    // Print a few values
    std::cout << "C[0][0] = " << h_C[0] << '\n';
    std::cout << "C[0][1] = " << h_C[1] << '\n';
    std::cout << "C[1][0] = " << h_C[N] << '\n';

    // Expected:
    // Each output = sum_{i=0}^{63} (1 * 2) = 128

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return;
}
#endif

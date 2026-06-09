#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define TILE 16

// multiplying MxK and KxN matrices
__global__ void GEMM(const float* A, const float* B, float* C,
    int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // blockIdx.y is the y index of the block we are on. blockIdx.y * TILE is the real row idx of the start of our tile
  // ty is the offset within a block, so we add ty to get the actual row idx. 
  // C[row][col] represents the index of the final output
  int row = blockIdx.y * TILE + ty;
  int col = blockIdx.x * TILE + tx;

  float sum = 0.0f;

  // number of tiles in K (shared dimension)
  // for A, we are sliding our tiled window col-wise. for B, we are sliding our tiled window row-wise.
  // one iteration of this loop calculates the partial product for one TILE, summing up all the tiles in the shared dimension leads to the final result.
  for (int t = 0; t < (K + TILE - 1) / TILE; ++t) { 
    // C[i][j] = sum_{k=0}^{k=K} A[i][k] * B[k][j]
    // Notice the row of A remains the same, and the col of B remains the same
    int a_col = t * TILE + tx; // we are currently on the t'th tile, with offset tx inside that tile
    int b_row = t * TILE + ty;

    // since A and B are flattened:
    // A[row][a_col] = A[row * K + a_col], K is the col dim. A[row][a_col] represents the element of A corresponding to A[ty][tx] that we want to load into our shared tile
    // B[b_row][col] = B[b_row * N + col], N is the col dim. B[b_row][col] represents the element of B corresponding to B[ty][tx] that we want to load into our shared tile
    if (row < M && a_col < K) {
        As[ty][tx] = A[row * K + a_col];
    } else {
        As[ty][tx] = 0.0f;
    }

    if (b_row < K && col < N) {
        Bs[ty][tx] = B[b_row * N + col];
    } else {
        Bs[ty][tx] = 0.0f;
    }
    // we sync threads here, waiting for every one of the 16x16=256 threads in our tile to load
    __syncthreads();

    // calculate the partial result of C[row][col]
    // The result is partial because we're essentially doing C[i][j] = Σ_t Σ_k A[i][t*TILE + k] * B[t*TILE + k][j]
    // each iteration of the big loop calculates one inner summation
    
    // this loop calculates one dot product of a 1x16 * 16x1 vector
    for (int k = 0; k < TILE; ++k) {
      sum += As[ty][k] * Bs[k][tx];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    // C is also flattened, so we translate row, col to 1d
    C[row * N + col] = sum;
  }

  // once all threads in this block finish, we will have the finalized 16x16 tile of the final output. 
  // other blocks will independently compute other 16x16 tiles in our matmul
}

int main() {
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
    dim3 block(TILE, TILE);

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

    return 0;
}

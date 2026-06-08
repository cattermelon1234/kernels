#include <cuda_runtime.h>

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


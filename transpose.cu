#define TILE 16

// naive transpose kernel
__global__ void transpose(const float* A, float* B, int rows, int cols) {
    // say blockDim.x = 16, blockDim.y = 16
    // then threadIdx.x goes from 0 to 15, threadIdx.y goes from 0 to 15

    // within a block: thread (i, j) handles the element 
    // at (row, col) = (blockIdx.y * blockDim.y + threadIdx.y, blockIdx.x * blockDim.x + threadIdx.x)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        // reads: 
        // A[row * cols + col] 
        B[col * rows + row] = A[row * cols + col];
    }
}

__global__ void transposeTiled(const float* A, float* B, int M, int N) {
    __shared__ float tile[TILE][TILE + 1]; // +1 avoids shared memory bank conflicts

    int x = blockIdx.x * TILE + threadIdx.x; // col in A
    int y = blockIdx.y * TILE + threadIdx.y; // row in A

    if (y < M && x < N) {
        tile[threadIdx.y][threadIdx.x] = A[y * N + x];
    }

    __syncthreads();

    // swap block coordinates for output
    int out_x = blockIdx.y * TILE + threadIdx.x; // col in B
    int out_y = blockIdx.x * TILE + threadIdx.y; // row in B

    if (out_y < N && out_x < M) {
        B[out_y * M + out_x] = tile[threadIdx.x][threadIdx.y];
    }
}

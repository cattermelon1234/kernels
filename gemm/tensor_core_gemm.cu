#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

#include <iostream>
#include <vector>

using namespace nvcuda;

constexpr int B_M = 128;
constexpr int B_N = 128;
constexpr int B_K = 8;

constexpr int MMA_M = 16;
constexpr int MMA_N = 16;
constexpr int MMA_K = 8;

constexpr int WARP_TILE = 32;

constexpr int WARP_M = WARP_TILE / MMA_M; // 2
constexpr int WARP_N = WARP_TILE / MMA_N; // 2

constexpr int WARPS_M = B_M / WARP_TILE; // 4
constexpr int WARPS_N = B_N / WARP_TILE; // 4
constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N; // 16

__global__ void GEMM(const float* A, const float* B, float* C,
          int M, int N, int K);

using AccFrag = wmma::fragment<
    wmma::accumulator,
    MMA_M, MMA_N, MMA_K,
    float>;

using AFrag = wmma::fragment<
    wmma::matrix_a,
    MMA_M, MMA_N, MMA_K,
    wmma::precision::tf32,
    wmma::row_major>;

using BFrag = wmma::fragment<
    wmma::matrix_b,
    MMA_M, MMA_N, MMA_K,
    wmma::precision::tf32,
    wmma::row_major>;

__device__ __forceinline__
void mma_accumulate(AccFrag& c, AFrag& a, BFrag& b) {
    wmma::mma_sync(c, a, b, c);
}

__device__ __forceinline__
void outer_product(
    AccFrag c[WARP_M][WARP_N],
    AFrag a[WARP_M],
    BFrag b[WARP_N])
{
#pragma unroll
    for (int i = 0; i < WARP_M; i++) {
#pragma unroll
        for (int j = 0; j < WARP_N; j++) {
            mma_accumulate(c[i][j], a[i], b[j]);
        }
    }
}

// A: M x K
// B: K x N
// C: M x N
//
// Teaching version assumption:
// M and N are multiples of 128.
// K is handled with zero-padding.
__global__
void GEMM(const float* A, const float* B, float* C,
          int M, int N, int K)
{
    __shared__ float As[B_M][B_K];
    __shared__ float Bs[B_K][B_N];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int warp_id = threadIdx.y; // because block is dim3(32, 16)
    int lane_id = threadIdx.x;

    int block_row = blockIdx.y * B_M;
    int block_col = blockIdx.x * B_N;

    int warp_m = warp_id / WARPS_N;
    int warp_n = warp_id % WARPS_N;

    int warp_row = warp_m * WARP_TILE;
    int warp_col = warp_n * WARP_TILE;

    AccFrag c[WARP_M][WARP_N];

#pragma unroll
    for (int i = 0; i < WARP_M; i++) {
#pragma unroll
        for (int j = 0; j < WARP_N; j++) {
            wmma::fill_fragment(c[i][j], 0.0f);
        }
    }

    for (int k0 = 0; k0 < K; k0 += B_K) {

        // Load A tile: B_M x B_K
        for (int idx = tid; idx < B_M * B_K; idx += blockDim.x * blockDim.y) {
            int row = idx / B_K;
            int col = idx % B_K;

            int global_row = block_row + row;
            int global_col = k0 + col;

            if (global_row < M && global_col < K) {
                As[row][col] = A[global_row * K + global_col];
            } else {
                As[row][col] = 0.0f;
            }
        }

        // Load B tile: B_K x B_N
        for (int idx = tid; idx < B_K * B_N; idx += blockDim.x * blockDim.y) {
            int row = idx / B_N;
            int col = idx % B_N;

            int global_row = k0 + row;
            int global_col = block_col + col;

            if (global_row < K && global_col < N) {
                Bs[row][col] = B[global_row * N + global_col];
            } else {
                Bs[row][col] = 0.0f;
            }
        }

        __syncthreads();

        AFrag a[WARP_M];
        BFrag b[WARP_N];

#pragma unroll
        for (int i = 0; i < WARP_M; i++) {
            wmma::load_matrix_sync(
                a[i],
                &As[warp_row + i * MMA_M][0],
                B_K
            );
        }

#pragma unroll
        for (int j = 0; j < WARP_N; j++) {
            wmma::load_matrix_sync(
                b[j],
                &Bs[0][warp_col + j * MMA_N],
                B_N
            );
        }

        outer_product(c, a, b);

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < WARP_M; i++) {
#pragma unroll
        for (int j = 0; j < WARP_N; j++) {
            int out_row = block_row + warp_row + i * MMA_M;
            int out_col = block_col + warp_col + j * MMA_N;

            wmma::store_matrix_sync(
                &C[out_row * N + out_col],
                c[i][j],
                N,
                wmma::mem_row_major
            );
        }
    }
}

#ifndef GEMM_DISABLE_STANDALONE_MAIN
int main() {
    const int M = 128;
    const int K = 128;
    const int N = 128;

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);

    for (int i = 0; i < M * K; i++) {
        h_A[i] = 1.0f;
    }

    for (int i = 0; i < K * N; i++) {
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

    dim3 block(32, WARPS_PER_BLOCK);

    dim3 grid(
        (N + B_N - 1) / B_N,
        (M + B_M - 1) / B_M
    );

    GEMM<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "C[0][0] = " << h_C[0] << "\n";
    std::cout << "C[0][1] = " << h_C[1] << "\n";
    std::cout << "C[1][0] = " << h_C[N] << "\n";

    std::cout << "Expected = " << K * 2.0f << "\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
#endif

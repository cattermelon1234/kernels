#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

constexpr int NAIVE_TILE = 16;
constexpr int OPT_TILE = 64;
constexpr int REG = 4;

constexpr int TENSOR_B_M = 128;
constexpr int TENSOR_B_N = 128;
constexpr int TENSOR_B_K = 16;
constexpr int TENSOR_MMA_M = 16;
constexpr int TENSOR_MMA_N = 16;
constexpr int TENSOR_MMA_K = 16;
constexpr int TENSOR_WARP_TILE = 64;
constexpr int TENSOR_WARP_M = TENSOR_WARP_TILE / TENSOR_MMA_M;
constexpr int TENSOR_WARP_N = TENSOR_WARP_TILE / TENSOR_MMA_N;
constexpr int TENSOR_WARPS_M = TENSOR_B_M / TENSOR_WARP_TILE;
constexpr int TENSOR_WARPS_N = TENSOR_B_N / TENSOR_WARP_TILE;
constexpr int TENSOR_WARPS_PER_BLOCK = TENSOR_WARPS_M * TENSOR_WARPS_N;

using TensorAccFrag = wmma::fragment<
    wmma::accumulator,
    TENSOR_MMA_M, TENSOR_MMA_N, TENSOR_MMA_K,
    float>;

using TensorAFrag = wmma::fragment<
    wmma::matrix_a,
    TENSOR_MMA_M, TENSOR_MMA_N, TENSOR_MMA_K,
    half,
    wmma::row_major>;

using TensorBFrag = wmma::fragment<
    wmma::matrix_b,
    TENSOR_MMA_M, TENSOR_MMA_N, TENSOR_MMA_K,
    half,
    wmma::row_major>;

__device__ __forceinline__
void tensor_mma_accumulate(TensorAccFrag& c, TensorAFrag& a, TensorBFrag& b) {
    wmma::mma_sync(c, a, b, c);
}

__device__ __forceinline__
void tensor_outer_product(
    TensorAccFrag c[TENSOR_WARP_M][TENSOR_WARP_N],
    TensorAFrag a[TENSOR_WARP_M],
    TensorBFrag b[TENSOR_WARP_N])
{
#pragma unroll
    for (int i = 0; i < TENSOR_WARP_M; i++) {
#pragma unroll
        for (int j = 0; j < TENSOR_WARP_N; j++) {
            tensor_mma_accumulate(c[i][j], a[i], b[j]);
        }
    }
}

__global__
inline void gemm_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[NAIVE_TILE][NAIVE_TILE];
    __shared__ float Bs[NAIVE_TILE][NAIVE_TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * NAIVE_TILE + ty;
    int col = blockIdx.x * NAIVE_TILE + tx;
    float sum = 0.0f;

    for (int t = 0; t < (K + NAIVE_TILE - 1) / NAIVE_TILE; ++t) {
        int a_col = t * NAIVE_TILE + tx;
        int b_row = t * NAIVE_TILE + ty;

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

        __syncthreads();

        for (int k = 0; k < NAIVE_TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__
inline void gemm_optimized_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[OPT_TILE][OPT_TILE];
    __shared__ float Bs[OPT_TILE][OPT_TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int block_row = blockIdx.y * OPT_TILE;
    int block_col = blockIdx.x * OPT_TILE;
    int base_row = block_row + ty * REG;
    int base_col = block_col + tx * REG;

    float sum[REG][REG];

#pragma unroll
    for (int i = 0; i < REG; i++) {
#pragma unroll
        for (int j = 0; j < REG; j++) {
            sum[i][j] = 0.0f;
        }
    }

    for (int t = 0; t < (K + OPT_TILE - 1) / OPT_TILE; ++t) {
        for (int i = 0; i < REG; ++i) {
            for (int j = 0; j < REG; ++j) {
                int a_row = base_row + i;
                int a_col = t * OPT_TILE + tx * REG + j;
                int b_row = t * OPT_TILE + ty * REG + i;
                int b_col = base_col + j;

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

        for (int k_inner = 0; k_inner < OPT_TILE; k_inner++) {
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

__global__
inline void gemm_tensor_core_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ half As[TENSOR_B_M][TENSOR_B_K];
    __shared__ half Bs[TENSOR_B_K][TENSOR_B_N];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = threadIdx.y;

    int block_row = blockIdx.y * TENSOR_B_M;
    int block_col = blockIdx.x * TENSOR_B_N;

    int warp_m = warp_id / TENSOR_WARPS_N;
    int warp_n = warp_id % TENSOR_WARPS_N;
    int warp_row = warp_m * TENSOR_WARP_TILE;
    int warp_col = warp_n * TENSOR_WARP_TILE;

    TensorAccFrag c[TENSOR_WARP_M][TENSOR_WARP_N];

#pragma unroll
    for (int i = 0; i < TENSOR_WARP_M; i++) {
#pragma unroll
        for (int j = 0; j < TENSOR_WARP_N; j++) {
            wmma::fill_fragment(c[i][j], 0.0f);
        }
    }

    for (int k0 = 0; k0 < K; k0 += TENSOR_B_K) {
        for (int idx = tid; idx < TENSOR_B_M * TENSOR_B_K; idx += blockDim.x * blockDim.y) {
            int row = idx / TENSOR_B_K;
            int col = idx % TENSOR_B_K;
            int global_row = block_row + row;
            int global_col = k0 + col;

            if (global_row < M && global_col < K) {
                As[row][col] = __float2half(A[global_row * K + global_col]);
            } else {
                As[row][col] = __float2half(0.0f);
            }
        }

        for (int idx = tid; idx < TENSOR_B_K * TENSOR_B_N; idx += blockDim.x * blockDim.y) {
            int row = idx / TENSOR_B_N;
            int col = idx % TENSOR_B_N;
            int global_row = k0 + row;
            int global_col = block_col + col;

            if (global_row < K && global_col < N) {
                Bs[row][col] = __float2half(B[global_row * N + global_col]);
            } else {
                Bs[row][col] = __float2half(0.0f);
            }
        }

        __syncthreads();

        TensorAFrag a[TENSOR_WARP_M];
        TensorBFrag b[TENSOR_WARP_N];

#pragma unroll
        for (int i = 0; i < TENSOR_WARP_M; i++) {
            wmma::load_matrix_sync(
                a[i],
                &As[warp_row + i * TENSOR_MMA_M][0],
                TENSOR_B_K
            );
        }

#pragma unroll
        for (int j = 0; j < TENSOR_WARP_N; j++) {
            wmma::load_matrix_sync(
                b[j],
                &Bs[0][warp_col + j * TENSOR_MMA_N],
                TENSOR_B_N
            );
        }

        tensor_outer_product(c, a, b);
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TENSOR_WARP_M; i++) {
#pragma unroll
        for (int j = 0; j < TENSOR_WARP_N; j++) {
            int out_row = block_row + warp_row + i * TENSOR_MMA_M;
            int out_col = block_col + warp_col + j * TENSOR_MMA_N;

            wmma::store_matrix_sync(
                &C[out_row * N + out_col],
                c[i][j],
                N,
                wmma::mem_row_major
            );
        }
    }
}

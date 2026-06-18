#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/pipeline>
#include <mma.h>
#include <cooperative_groups.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
using namespace nvcuda;
namespace cg = cooperative_groups;
#endif

constexpr int NAIVE_TILE = 16;
constexpr int OPT_TILE = 64;
constexpr int REG = 4;

constexpr int TENSOR_B_M = 128;
constexpr int TENSOR_B_N = 128;
constexpr int TENSOR_B_K = 8;
constexpr int TENSOR_MMA_M = 16;
constexpr int TENSOR_MMA_N = 16;
constexpr int TENSOR_MMA_K = 8;
constexpr int TENSOR_WARP_TILE = 32;
constexpr int TENSOR_WARP_M = TENSOR_WARP_TILE / TENSOR_MMA_M;
constexpr int TENSOR_WARP_N = TENSOR_WARP_TILE / TENSOR_MMA_N;
constexpr int TENSOR_WARPS_M = TENSOR_B_M / TENSOR_WARP_TILE;
constexpr int TENSOR_WARPS_N = TENSOR_B_N / TENSOR_WARP_TILE;
constexpr int TENSOR_WARPS_PER_BLOCK = TENSOR_WARPS_M * TENSOR_WARPS_N;

constexpr int HYPER_B_M = 128;
constexpr int HYPER_B_N = 128;
constexpr int HYPER_B_K = 32;
constexpr int HYPER_MMA_M = 16;
constexpr int HYPER_MMA_N = 16;
constexpr int HYPER_MMA_K = 8;
constexpr int HYPER_WARP_TILE = 32;
constexpr int HYPER_WARP_M = HYPER_WARP_TILE / HYPER_MMA_M;
constexpr int HYPER_WARP_N = HYPER_WARP_TILE / HYPER_MMA_N;
constexpr int HYPER_WARPS_M = HYPER_B_M / HYPER_WARP_TILE;
constexpr int HYPER_WARPS_N = HYPER_B_N / HYPER_WARP_TILE;
constexpr int HYPER_WARPS_PER_BLOCK = HYPER_WARPS_M * HYPER_WARPS_N;
constexpr int HYPER_STAGES = 2;

__global__ void gemm_tensor_core_kernel(const float* A, const float* B, float* C, int M, int N, int K);

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

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

using TensorAccFrag = wmma::fragment<
    wmma::accumulator,
    TENSOR_MMA_M, TENSOR_MMA_N, TENSOR_MMA_K,
    float>;

using TensorAFrag = wmma::fragment<
    wmma::matrix_a,
    TENSOR_MMA_M, TENSOR_MMA_N, TENSOR_MMA_K,
    wmma::precision::tf32,
    wmma::row_major>;

using TensorBFrag = wmma::fragment<
    wmma::matrix_b,
    TENSOR_MMA_M, TENSOR_MMA_N, TENSOR_MMA_K,
    wmma::precision::tf32,
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
inline void gemm_tensor_core_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TENSOR_B_M][TENSOR_B_K];
    __shared__ float Bs[TENSOR_B_K][TENSOR_B_N];

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
                As[row][col] = A[global_row * K + global_col];
            } else {
                As[row][col] = 0.0f;
            }
        }

        for (int idx = tid; idx < TENSOR_B_K * TENSOR_B_N; idx += blockDim.x * blockDim.y) {
            int row = idx / TENSOR_B_N;
            int col = idx % TENSOR_B_N;
            int global_row = k0 + row;
            int global_col = block_col + col;

            if (global_row < K && global_col < N) {
                Bs[row][col] = B[global_row * N + global_col];
            } else {
                Bs[row][col] = 0.0f;
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
#else

__global__
inline void gemm_tensor_core_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    (void)A;
    (void)B;
    (void)C;
    (void)M;
    (void)N;
    (void)K;
}

#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

using HyperAccFrag = wmma::fragment<
    wmma::accumulator,
    HYPER_MMA_M, HYPER_MMA_N, HYPER_MMA_K,
    float>;

using HyperAFrag = wmma::fragment<
    wmma::matrix_a,
    HYPER_MMA_M, HYPER_MMA_N, HYPER_MMA_K,
    wmma::precision::tf32,
    wmma::row_major>;

using HyperBFrag = wmma::fragment<
    wmma::matrix_b,
    HYPER_MMA_M, HYPER_MMA_N, HYPER_MMA_K,
    wmma::precision::tf32,
    wmma::row_major>;

__device__ __forceinline__
void hyper_mma_accumulate(HyperAccFrag& c, HyperAFrag& a, HyperBFrag& b) {
    wmma::mma_sync(c, a, b, c);
}

__device__ __forceinline__
void hyper_outer_product(
    HyperAccFrag c[HYPER_WARP_M][HYPER_WARP_N],
    HyperAFrag a[HYPER_WARP_M],
    HyperBFrag b[HYPER_WARP_N])
{
#pragma unroll
    for (int i = 0; i < HYPER_WARP_M; i++) {
#pragma unroll
        for (int j = 0; j < HYPER_WARP_N; j++) {
            hyper_mma_accumulate(c[i][j], a[i], b[j]);
        }
    }
}

template <typename Pipeline>
__device__ __forceinline__
void hyper_load_a_tile_async(
    cg::thread_block block,
    const float* A,
    float (&As)[HYPER_STAGES][HYPER_B_M][HYPER_B_K],
    int stage,
    int block_row,
    int k0,
    int M,
    int K,
    Pipeline& pipe)
{
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;
    constexpr int CHUNKS_PER_ROW = HYPER_B_K / 4;

    for (int idx = tid; idx < HYPER_B_M * CHUNKS_PER_ROW; idx += num_threads) {
        int row = idx / CHUNKS_PER_ROW;
        int chunk = idx % CHUNKS_PER_ROW;
        int global_row = block_row + row;
        int global_col = k0 + chunk * 4;

        if (global_row < M && global_col + 3 < K) {
            cuda::memcpy_async(
                block,
                reinterpret_cast<void*>(&As[stage][row][chunk * 4]),
                reinterpret_cast<const void*>(&A[global_row * K + global_col]),
                sizeof(float4),
                pipe
            );
        } else {
            for (int i = 0; i < 4 && chunk * 4 + i < HYPER_B_K; ++i) {
                As[stage][row][chunk * 4 + i] = 0.0f;
            }
        }
    }
}

template <typename Pipeline>
__device__ __forceinline__
void hyper_load_b_tile_async(
    cg::thread_block block,
    const float* B,
    float (&Bs)[HYPER_STAGES][HYPER_B_K][HYPER_B_N],
    int stage,
    int block_col,
    int k0,
    int K,
    int N,
    Pipeline& pipe)
{
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;
    constexpr int CHUNKS_PER_ROW = HYPER_B_N / 4;

    for (int idx = tid; idx < HYPER_B_K * CHUNKS_PER_ROW; idx += num_threads) {
        int row = idx / CHUNKS_PER_ROW;
        int chunk = idx % CHUNKS_PER_ROW;
        int global_row = k0 + row;
        int global_col = block_col + chunk * 4;

        if (global_row < K && global_col + 3 < N) {
            cuda::memcpy_async(
                block,
                reinterpret_cast<void*>(&Bs[stage][row][chunk * 4]),
                reinterpret_cast<const void*>(&B[global_row * N + global_col]),
                sizeof(float4),
                pipe
            );
        } else {
            for (int i = 0; i < 4 && chunk * 4 + i < HYPER_B_N; ++i) {
                Bs[stage][row][chunk * 4 + i] = 0.0f;
            }
        }
    }
}

__global__
inline void gemm_hyperoptimized_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[HYPER_STAGES][HYPER_B_M][HYPER_B_K];
    __shared__ float Bs[HYPER_STAGES][HYPER_B_K][HYPER_B_N];

    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope_block,
        HYPER_STAGES
    > pipe_state;

    cg::thread_block block = cg::this_thread_block();
    auto pipe = cuda::make_pipeline(block, &pipe_state);

    int warp_id = threadIdx.y;

    int block_row = blockIdx.y * HYPER_B_M;
    int block_col = blockIdx.x * HYPER_B_N;

    int warp_m = warp_id / HYPER_WARPS_N;
    int warp_n = warp_id % HYPER_WARPS_N;
    int warp_row = warp_m * HYPER_WARP_TILE;
    int warp_col = warp_n * HYPER_WARP_TILE;

    HyperAccFrag c[HYPER_WARP_M][HYPER_WARP_N];

#pragma unroll
    for (int i = 0; i < HYPER_WARP_M; i++) {
#pragma unroll
        for (int j = 0; j < HYPER_WARP_N; j++) {
            wmma::fill_fragment(c[i][j], 0.0f);
        }
    }

    int stage = 0;

    pipe.producer_acquire();
    hyper_load_a_tile_async(block, A, As, stage, block_row, 0, M, K, pipe);
    hyper_load_b_tile_async(block, B, Bs, stage, block_col, 0, K, N, pipe);
    pipe.producer_commit();

    pipe.consumer_wait();
    __syncthreads();

    for (int k0 = 0; k0 < K; k0 += HYPER_B_K) {
        int next_k0 = k0 + HYPER_B_K;
        int next_stage = stage ^ 1;

        if (next_k0 < K) {
            pipe.producer_acquire();
            hyper_load_a_tile_async(block, A, As, next_stage, block_row, next_k0, M, K, pipe);
            hyper_load_b_tile_async(block, B, Bs, next_stage, block_col, next_k0, K, N, pipe);
            pipe.producer_commit();
        }

        for (int kk = 0; kk < HYPER_B_K; kk += HYPER_MMA_K) {
            HyperAFrag a[HYPER_WARP_M];
            HyperBFrag b[HYPER_WARP_N];

#pragma unroll
            for (int i = 0; i < HYPER_WARP_M; i++) {
                wmma::load_matrix_sync(
                    a[i],
                    &As[stage][warp_row + i * HYPER_MMA_M][kk],
                    HYPER_B_K
                );
            }

#pragma unroll
            for (int j = 0; j < HYPER_WARP_N; j++) {
                wmma::load_matrix_sync(
                    b[j],
                    &Bs[stage][kk][warp_col + j * HYPER_MMA_N],
                    HYPER_B_N
                );
            }

            hyper_outer_product(c, a, b);
        }

        pipe.consumer_release();

        if (next_k0 < K) {
            pipe.consumer_wait();
            __syncthreads();
        }

        stage = next_stage;
    }

#pragma unroll
    for (int i = 0; i < HYPER_WARP_M; i++) {
#pragma unroll
        for (int j = 0; j < HYPER_WARP_N; j++) {
            int out_row = block_row + warp_row + i * HYPER_MMA_M;
            int out_col = block_col + warp_col + j * HYPER_MMA_N;

            wmma::store_matrix_sync(
                &C[out_row * N + out_col],
                c[i][j],
                N,
                wmma::mem_row_major
            );
        }
    }
}

#else

__global__
inline void gemm_hyperoptimized_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    (void)A;
    (void)B;
    (void)C;
    (void)M;
    (void)N;
    (void)K;
}

#endif

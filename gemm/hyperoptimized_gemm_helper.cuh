#pragma once

#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>

namespace hyperoptimized_gemm {
using namespace nvcuda;
namespace cg = cooperative_groups;

constexpr int B_M = 128;
constexpr int B_N = 128;
constexpr int B_K = 16;

constexpr int MMA_M = 16;
constexpr int MMA_N = 16;
constexpr int MMA_K = 8;

constexpr int WARP_TILE = 32;

constexpr int WARP_M = WARP_TILE / MMA_M; // 2
constexpr int WARP_N = WARP_TILE / MMA_N; // 2

constexpr int WARPS_M = B_M / WARP_TILE; // 4
constexpr int WARPS_N = B_N / WARP_TILE; // 4
constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N; // 16

constexpr int STAGES = 2;

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

struct StoreFragment {
    __device__ __forceinline__
    void operator()(float* C, int out_row, int out_col, int M, int N, const AccFrag& frag) const {
        float tile[MMA_M][MMA_N];
        wmma::store_matrix_sync(&tile[0][0], frag, MMA_N, wmma::mem_row_major);

        #pragma unroll
        for (int i = 0; i < MMA_M; ++i) {
            #pragma unroll
            for (int j = 0; j < MMA_N; ++j) {
                int row = out_row + i;
                int col = out_col + j;
                if (row < M && col < N) {
                    C[row * N + col] = tile[i][j];
                }
            }
        }
    }
};

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

template <typename Pipeline>
__device__ __forceinline__
void load_a_tile_async(
    cg::thread_block block,
    const float* A,
    float (&As)[STAGES][B_M][B_K],
    int stage,
    int block_row,
    int k0,
    int M,
    int K,
    Pipeline& pipe)
{
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;
    constexpr int CHUNKS_PER_ROW = B_K / 4;

    for (int idx = tid; idx < B_M * CHUNKS_PER_ROW; idx += num_threads) {
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
            for (int i = 0; i < 4 && chunk * 4 + i < B_K; ++i) {
                As[stage][row][chunk * 4 + i] = 0.0f;
            }
        }
    }
}

template <typename Pipeline>
__device__ __forceinline__
void load_b_tile_async(
    cg::thread_block block,
    const float* B,
    float (&Bs)[STAGES][B_K][B_N],
    int stage,
    int block_col,
    int k0,
    int K,
    int N,
    Pipeline& pipe)
{
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;
    constexpr int CHUNKS_PER_ROW = B_N / 4;

    for (int idx = tid; idx < B_K * CHUNKS_PER_ROW; idx += num_threads) {
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
            for (int i = 0; i < 4 && chunk * 4 + i < B_N; ++i) {
                Bs[stage][row][chunk * 4 + i] = 0.0f;
            }
        }
    }
}

template <typename StoreOp>
__device__ __forceinline__
void hyperoptimized_gemm_block(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    StoreOp store_op)
{
    __shared__ float As[STAGES][B_M][B_K];
    __shared__ float Bs[STAGES][B_K][B_N];

    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope_block,
        STAGES
    > pipe_state;

    cg::thread_block block = cg::this_thread_block();
    auto pipe = cuda::make_pipeline(block, &pipe_state);

    int warp_id = threadIdx.y;

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

    int stage = 0;

    pipe.producer_acquire();
    load_a_tile_async(block, A, As, stage, block_row, 0, M, K, pipe);
    load_b_tile_async(block, B, Bs, stage, block_col, 0, K, N, pipe);
    pipe.producer_commit();

    pipe.consumer_wait();
    __syncthreads();

    for (int k0 = 0; k0 < K; k0 += B_K) {
        int next_k0 = k0 + B_K;
        int next_stage = stage ^ 1;

        if (next_k0 < K) {
            pipe.producer_acquire();
            load_a_tile_async(block, A, As, next_stage, block_row, next_k0, M, K, pipe);
            load_b_tile_async(block, B, Bs, next_stage, block_col, next_k0, K, N, pipe);
            pipe.producer_commit();
        }

        for (int kk = 0; kk < B_K; kk += MMA_K) {
            AFrag a[WARP_M];
            BFrag b[WARP_N];

#pragma unroll
            for (int i = 0; i < WARP_M; i++) {
                wmma::load_matrix_sync(
                    a[i],
                    &As[stage][warp_row + i * MMA_M][kk],
                    B_K
                );
            }

#pragma unroll
            for (int j = 0; j < WARP_N; j++) {
                wmma::load_matrix_sync(
                    b[j],
                    &Bs[stage][kk][warp_col + j * MMA_N],
                    B_N
                );
            }

            outer_product(c, a, b);
        }

        pipe.consumer_release();

        if (next_k0 < K) {
            pipe.consumer_wait();
            __syncthreads();
        }

        stage = next_stage;
    }

#pragma unroll
    for (int i = 0; i < WARP_M; i++) {
#pragma unroll
        for (int j = 0; j < WARP_N; j++) {
            int out_row = block_row + warp_row + i * MMA_M;
            int out_col = block_col + warp_col + j * MMA_N;
            store_op(C, out_row, out_col, M, N, c[i][j]);
        }
    }
}

template <typename StoreOp = StoreFragment>
__device__ __forceinline__
void hyperoptimized_gemm_block(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    hyperoptimized_gemm_block(A, B, C, M, N, K, StoreOp{});
}

} // namespace hyperoptimized_gemm

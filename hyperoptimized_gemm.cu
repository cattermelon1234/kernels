#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>

#include <iostream>
#include <vector>

using namespace nvcuda;
namespace cg = cooperative_groups;

constexpr int B_M = 128;
constexpr int B_N = 128;
constexpr int B_K = 32;

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

__global__
void GEMM(const float* A, const float* B, float* C,
          int M, int N, int K)
{
    __shared__ float As[STAGES][B_M][B_K];
    __shared__ float Bs[STAGES][B_K][B_N];

    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope_block,
        STAGES
    > pipe_state;

    cg::thread_block block = cg::this_thread_block();
    auto pipe = cuda::make_pipeline(block, &pipe_state);

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = threadIdx.y; // block is dim3(32, 16)

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

            wmma::store_matrix_sync(
                &C[out_row * N + out_col],
                c[i][j],
                N,
                wmma::mem_row_major
            );
        }
    }
}

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

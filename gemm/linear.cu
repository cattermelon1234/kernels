#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "gemm_helper.cuh"
#include "../epilogues/linear.cuh"

// Fused linear layer:
// Y = epilogue(X * W, row, col)
//
// The GEMM mainloop comes from hyperoptimized_gemm_helper.cuh.
// This file only adapts the accumulator tile to a per-element epilogue.

namespace linear_impl {
using namespace hyperoptimized_gemm;

template <typename Epilogue>
struct TileStore {
    Epilogue epilogue;

    __device__ __forceinline__
    void operator()(float* Y, int out_row, int out_col, int M, int N, const AccFrag& frag) const {
        float tile[MMA_M][MMA_N];
        wmma::store_matrix_sync(&tile[0][0], frag, MMA_N, wmma::mem_row_major);

#pragma unroll
        for (int i = 0; i < MMA_M; ++i) {
#pragma unroll
            for (int j = 0; j < MMA_N; ++j) {
                int row = out_row + i;
                int col = out_col + j;
                if (row < M && col < N) {
                    float x = tile[i][j];
                    x = epilogue(x, row, col);
                    Y[row * N + col] = x;
                }
            }
        }
    }
};
} // namespace linear_impl

template <typename Epilogue>
__global__
void linear_kernel(
    const float* X,
    const float* W,
    float* Y,
    int M,
    int N,
    int K,
    Epilogue epilogue)
{
    linear_impl::TileStore<Epilogue> store_op{epilogue};
    hyperoptimized_gemm::hyperoptimized_gemm_block(X, W, Y, M, N, K, store_op);
}

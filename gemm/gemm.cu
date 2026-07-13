#include <iostream>
#include <vector>
#include "gemm_helper.cuh"
#include "gemm.cuh"

namespace kernels::gemm {

using namespace hyperoptimized_gemm;

__global__
void gemm_kernel(const float* A, const float* B, float* C,
                 int M, int N, int K)
{
    hyperoptimized_gemm_block(A, B, C, M, N, K);
}

void launch_gemm(const float* A, const float* B, float* C,
                 int M, int N, int K, cudaStream_t stream)
{
    dim3 block(32, WARPS_PER_BLOCK);
    dim3 grid(
        (N + B_N - 1) / B_N,
        (M + B_M - 1) / B_M
    );
    gemm_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

#ifndef GEMM_DISABLE_STANDALONE_MAIN
void demo_hyperoptimized_gemm() {
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

    gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "C[0][0] = " << h_C[0] << "\n";
    std::cout << "C[0][1] = " << h_C[1] << "\n";
    std::cout << "C[1][0] = " << h_C[N] << "\n";

    std::cout << "Expected = " << K * 2.0f << "\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return;
}
#endif

}

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <ctime>
#include <cstdlib>

#include "benchmark.cuh"

#define WARP_SIZE 32

template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer(size_t n) {
        cudaMalloc(&ptr_, n * sizeof(T));
    }

    ~DeviceBuffer() {
        cudaFree(ptr_);
    }

    T* data() {
        return ptr_;
    }

private:
    T* ptr_;
};

__device__ __forceinline__
float warp_reduce_sum(float x) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        x += __shfl_down_sync(0xffffffff, x, offset);
    }
    return x;
}

template<int BLOCK_SIZE>
__device__ __forceinline__
float block_reduce_sum(float x) {
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp = tid / WARP_SIZE;
    __shared__ float warp_vals[BLOCK_SIZE / WARP_SIZE];
    x = warp_reduce_sum(x);
    __syncthreads();

    if (lane == 0) {
        warp_vals[warp] = x;
    }
    __syncthreads();

    x = (tid < BLOCK_SIZE / WARP_SIZE) ? warp_vals[lane] : 0.0f;

    if (warp == 0) {
        x = warp_reduce_sum(x);
    }

    return x;
}

template<int BLOCK_SIZE>
__global__ void layernorm(const float* x, float* out, int rows, int cols, float eps) {
    int tid = threadIdx.x;
    int row = blockIdx.x;

    if (row >= rows) {
        return;
    }

    const float* row_x = x + row * cols;
    float* row_out = out + row * cols;
    const float4* row_x4 = reinterpret_cast<const float4*>(row_x);
    float4* row_out4 = reinterpret_cast<float4*>(row_out);
    int vec_cols = cols / 4;

    float thread_sum = 0.0f;
    float thread_sumsq = 0.0f;

    for (int col = tid; col < vec_cols; col += BLOCK_SIZE) {
        float4 v = row_x4[col];
        thread_sum += v.x + v.y + v.z + v.w;
        thread_sumsq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (int col = vec_cols * 4 + tid; col < cols; col += BLOCK_SIZE) {
        float v = row_x[col];
        thread_sum += v;
        thread_sumsq += v * v;
    }

    float row_sum = block_reduce_sum<BLOCK_SIZE>(thread_sum);
    float row_sumsq = block_reduce_sum<BLOCK_SIZE>(thread_sumsq);

    __shared__ float shared_sum;
    __shared__ float shared_sumsq;
    if (tid == 0) {
        shared_sum = row_sum;
        shared_sumsq = row_sumsq;
    }
    __syncthreads();

    float mean = shared_sum / cols;
    float var = shared_sumsq / cols - mean * mean;
    float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < vec_cols; col += BLOCK_SIZE) {
        float4 v = row_x4[col];
        row_out4[col] = make_float4(
            (v.x - mean) * inv_std,
            (v.y - mean) * inv_std,
            (v.z - mean) * inv_std,
            (v.w - mean) * inv_std
        );
    }
    for (int col = vec_cols * 4 + tid; col < cols; col += BLOCK_SIZE) {
        row_out[col] = (row_x[col] - mean) * inv_std;
    }
}

int main() {
    constexpr int BLOCK_SIZE = 256;

    const int rows = 4;
    const int cols = 1024;
    const float eps = 1e-5f;

    srand(time(0));

    std::vector<float> h_x(rows * cols);
    std::vector<float> h_out(rows * cols);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            h_x[r * cols + c] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }

    DeviceBuffer<float> d_x(rows * cols);
    DeviceBuffer<float> d_out(rows * cols);

    cudaMemcpy(
        d_x.data(),
        h_x.data(),
        rows * cols * sizeof(float),
        cudaMemcpyHostToDevice
    );

    dim3 block(BLOCK_SIZE);
    dim3 grid(rows);

    BenchmarkTimer timer;

    layernorm<BLOCK_SIZE><<<grid, block>>>(d_x.data(), d_out.data(), rows, cols, eps);
    cudaDeviceSynchronize();

    const int iters = 1000;

    timer.begin();

    for (int i = 0; i < iters; i++) {
        layernorm<BLOCK_SIZE><<<grid, block>>>(d_x.data(), d_out.data(), rows, cols, eps);
    }

    timer.end();

    float ms = timer.elapsed_ms();

    cudaMemcpy(
        h_out.data(),
        d_out.data(),
        rows * cols * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    benchmark_report(
        static_cast<double>(rows) *
            static_cast<double>(cols) *
            4.0,
        static_cast<double>(rows) *
            static_cast<double>(cols) *
            sizeof(float) *
            3.0,
        iters,
        ms
    );
    std::cout << "\n";

    std::cout << "First 10 outputs of row 0:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << h_out[i] << " ";
    }
    std::cout << "\n\n";

    std::cout << "Last 10 outputs of row 0:\n";
    for (int i = cols - 10; i < cols; i++) {
        std::cout << h_out[i] << " ";
    }
    std::cout << "\n";

    return 0;
}

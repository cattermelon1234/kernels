#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <math.h>
#include <vector>
#include <iostream>

#include "benchmark.cuh"

#define TILE 16
#define WARP_SIZE 32

// warp reductions: 
// a warp naturally executes simultaneously in lockstep, no synchronization needed!
// each iter, the # of threads halves, and each thread contains the max of 2->4->8->16->32 elements

__device__ __forceinline__
float warp_reduce_max(float thread_max) {
  for (int offset = 16; offset > 0; offset/= 2) {
    thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
  }
  return thread_max;
}

__device__ __forceinline__
float warp_reduce_sum(float x) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        x += __shfl_down_sync(0xffffffff, x, offset);
    }
    return x;
}

template<int BLOCK_SIZE>
__device__ __forceinline__
float block_reduce_max(float x) {
  int tid = threadIdx.x;
  int lane = tid % WARP_SIZE;
  int warp = tid / WARP_SIZE;
  __shared__ float warp_vals[BLOCK_SIZE / WARP_SIZE];
  x = warp_reduce_max(x);
  __syncthreads();

  // each thread's x register now holds the warp local max 
  if (lane == 0) {
    warp_vals[warp] = x;
  }
  __syncthreads();
  
  // reduce warp_vals 
  // our goal: thread 0 has warp_vals[0], thread 1 has warp_vals[1], etc.
  x = (tid < BLOCK_SIZE / WARP_SIZE) ? warp_vals[lane] : -FLT_MAX;

  // now, 1 warp holds all the reduction values, we just need to reduce this warp
  if (warp == 0) {
    x = warp_reduce_max(x);
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

  // each thread's x register now holds the warp sum 
  if (lane == 0) {
    warp_vals[warp] = x;
  }
  __syncthreads();
  
  // reduce warp_vals 
  // our goal: thread 0 has warp_vals[0], thread 1 has warp_vals[1], etc.
  x = (tid < BLOCK_SIZE / WARP_SIZE) ? warp_vals[lane] : 0;

  // now, 1 warp holds all the reduction values, we just need to reduce this warp
  if (warp == 0) {
    x = warp_reduce_sum(x);
  }
  
  return x;
}

template<int BLOCK_SIZE>
__global__ void softmax(float* x, float* out, int rows, int cols) {
  
  int tid = threadIdx.x;
  int row = blockIdx.x;

  if (row >= rows) {
    return; 
  }

  const float* row_x = x + row * cols;
  float* row_out = out + row * cols;
  const float4* row_x4 = reinterpret_cast<const float4*>(row_x);
  int vec_cols = cols / 4;

  // reduce over one row 
  float thread_max = -FLT_MAX;

  // BLOCK_SIZE threads per row, so each thread read is strided by BLOCK_SIZE 
  // each thread has to read multiple elements in the row
  for (int col = tid; col < vec_cols; col += BLOCK_SIZE) {
      float4 v = row_x4[col];
      thread_max = fmaxf(thread_max, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
  }
  for (int col = vec_cols * 4 + tid; col < cols; col += BLOCK_SIZE) {
      thread_max = fmaxf(thread_max, row_x[col]);
  }

  // do block reduce 
  // notice that thread_max is only correct for the first thread in each row due to the nature
  // of how our block_reduce works!
  float row_max = block_reduce_max<BLOCK_SIZE>(thread_max);

  // share row_max per thread in the row
  __shared__ float shared_max;
  if (tid == 0) {
    shared_max = row_max;
  }
  __syncthreads();
  row_max = shared_max;

  // compute exp(x - max)
  float thread_sum = 0.0f;

  // accumulate all the sums that tid is responsible for
  for (int col = tid; col < vec_cols; col += BLOCK_SIZE) {
    float4 v = row_x4[col];
    float4 e = make_float4(
        expf(v.x - row_max),
        expf(v.y - row_max),
        expf(v.z - row_max),
        expf(v.w - row_max)
    );
    thread_sum += e.x + e.y + e.z + e.w;
    reinterpret_cast<float4*>(row_out)[col] = e;
  }
  for (int col = vec_cols * 4 + tid; col < cols; col += BLOCK_SIZE) {
    float e = expf(row_x[col] - row_max);
    thread_sum += e;
    row_out[col] = e;
  }

  // compute sum reduction
  float row_sum = block_reduce_sum<BLOCK_SIZE>(thread_sum);

  // Make row_sum visible to all threads.
  __shared__ float shared_sum;
  if (tid == 0) {
      shared_sum = row_sum;
  }
  __syncthreads();

  row_sum = shared_sum;

  // Normalize.
  for (int col = tid; col < cols; col += BLOCK_SIZE) {
      row_out[col] /= row_sum;
  }
}

int main() {
    constexpr int BLOCK_SIZE = 256;

    const int rows = 4;
    const int cols = 1024;

    std::vector<float> h_x(rows * cols);
    std::vector<float> h_out(rows * cols);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            h_x[r * cols + c] = static_cast<float>(c);
        }
    }

    float* d_x;
    float* d_out;

    cudaMalloc(&d_x, rows * cols * sizeof(float));
    cudaMalloc(&d_out, rows * cols * sizeof(float));

    cudaMemcpy(
        d_x,
        h_x.data(),
        rows * cols * sizeof(float),
        cudaMemcpyHostToDevice
    );

    dim3 block(BLOCK_SIZE);
    dim3 grid(rows);

    BenchmarkTimer timer;

    softmax<BLOCK_SIZE><<<grid, block>>>(d_x, d_out, rows, cols);
    cudaDeviceSynchronize();

    const int iters = 1000;

    timer.begin();

    for (int i = 0; i < iters; i++) {
        softmax<BLOCK_SIZE><<<grid, block>>>(d_x, d_out, rows, cols);
    }

    timer.end();

    float ms = timer.elapsed_ms();

    cudaMemcpy(
        h_out.data(),
        d_out,
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
    std::cout << "\n\n";

    float sum = 0.0f;
    float max_prob = 0.0f;

    for (int c = 0; c < cols; c++) {
        sum += h_out[c];
        max_prob = std::max(max_prob, h_out[c]);
    }

    std::cout << "Row 0 sum = " << sum << "\n";
    std::cout << "Max prob = " << max_prob << "\n";

    cudaFree(d_x);
    cudaFree(d_out);

    return 0;
}

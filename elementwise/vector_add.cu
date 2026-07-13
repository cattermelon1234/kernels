#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>

__global__ void add(const int *a, const int *b, int *c, int numElements) {
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  // add the correct element for this thread 
  if (thread_id < numElements) {
    c[thread_id] = a[thread_id] + b[thread_id];
  }
}

void demo_vector_add() {
  // pointers for vectors on host (cpu) and device (gpu) 
  std::vector<int> hostA;
  std::vector<int> hostB;
  std::vector<int> hostResult;
  int *deviceA, *deviceB, *deviceC;

  int numElements = 1024;

  // create and allocate a, b and result vectors
  hostA = std::vector<int>(numElements);
  hostB = std::vector<int>(numElements);
  hostResult = std::vector<int>(numElements);

  // initialize input vectors
  for (int i = 0; i < numElements; ++i) {
    hostA[i] = i;
    hostB[i] = 2 * i;
  }

  // allocate tensors on gpu 
  int size = sizeof(int) * numElements;
  cudaMalloc((void**)&deviceA, size); 
  cudaMalloc((void**)&deviceB, size); 
  cudaMalloc((void**)&deviceC, size); 

  // memcpy vectors to GPU
  cudaMemcpy(deviceA, hostA.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB.data(), size, cudaMemcpyHostToDevice);

  // define grid and block dims 
  int threadsPerBlock = 256; // usually 128-512
  // blocksPerGrid ( ensure there are enough to cover numElements) 
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

  // call kernel
  add<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, deviceC, numElements);

  // copy result back to host
  cudaMemcpy(hostResult.data(), deviceC, size, cudaMemcpyDeviceToHost);

  // print results
  for (int i = 0; i < numElements; ++i) {
    std::cout << hostA[i] << " + " << hostB[i] << " = " << hostResult[i] << std::endl;
  }

  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
}

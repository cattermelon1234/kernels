#include <cuda_runtime.h>
#include <iostream>
#include <device_launch_parameters.h>
#include <ctime>

__global__ void dot(const float* a, const float* b, float* out, int n) {
    __shared__ float partial[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float local_sum = 0.0f;

    // each thread processes many elements
    while (idx < n) {
        local_sum += a[idx] * b[idx];
        idx += blockDim.x * gridDim.x;
    }

    partial[tid] = local_sum;
    __syncthreads();

    // reduction inside block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    // one atomic per block instead of per element
    if (tid == 0) {
        atomicAdd(out, partial[0]);
    }
}

float randomFloat(int randMax = 1000) {
  return static_cast<float>(rand()) / static_cast<float>(randMax);
}
// Main function
void demo_vector_dot_blocked(){
    // Seed the random number generator
    srand(time(0));

    // Define the number of elements in the vectors
    int numElements = 300;

    // Calculate the size of the vectors in bytes
    size_t size = numElements * sizeof(float);
    
    // Declare pointers for host and device vectors
    float *hostA, *hostB, *hostC;
    float *deviceA, *deviceB, *deviceC;

    // Allocate memory for host vectors
    hostA = (float*) malloc(size);
    hostB = (float*) malloc(size);
    hostC = (float*) malloc(sizeof(float)); // Only need space for one float for the result

    // Initialize host vectors with random floats
    for (int idx = 0; idx < numElements; idx++){
        hostA[idx] = randomFloat();
        hostB[idx] = randomFloat();
    }

    // Allocate memory for device vectors
    cudaMalloc((void**)&deviceA, size);
    cudaMalloc((void**)&deviceB, size);
    cudaMalloc((void**)&deviceC, sizeof(float)); // Only need space for one float for the result

    // Copy host vectors to device
    cudaMemcpy(deviceA, hostA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, size, cudaMemcpyHostToDevice);

    // Define the number of threads per block and the number of blocks per grid
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the vector dot product kernel
    dot<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, deviceC, numElements);

    // Copy the result from device to host
    cudaMemcpy(hostC, deviceC, sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result of the dot product
    std::cout << "Dot product: " << *hostC << std::endl;

    // Free the memory allocated for host vectors
    free(hostA);
    free(hostB);
    free(hostC);

    // Free the memory allocated for device vectors
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
}


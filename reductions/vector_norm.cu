#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cmath>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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

__global__
void vectorNorm(const float* vector,
                float* norm,
                int p,
                int numElements)
{
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    __shared__ float partial[256];

    if (idx < numElements) {
        partial[tid] = powf(fabsf(vector[idx]), p);
    } else {
        partial[tid] = 0.0f;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(norm, partial[0]);
    }
}

float randomFloat(int randMax = 1000) {
    return static_cast<float>(rand()) /
           static_cast<float>(randMax);
}

int main() {
    srand(time(0));

    int numElements = 4096;
    int p = 2;

    size_t size = numElements * sizeof(float);

    float* hostVector =
        (float*)malloc(size);

    float* hostNorm =
        (float*)malloc(sizeof(float));

    for (int i = 0; i < numElements; i++) {
        hostVector[i] = randomFloat();
    }

    DeviceBuffer<float> deviceVector(numElements);
    DeviceBuffer<float> deviceNorm(1);

    cudaMemcpy(
        deviceVector.data(),
        hostVector,
        size,
        cudaMemcpyHostToDevice
    );

    float zero = 0.0f;

    cudaMemcpy(
        deviceNorm.data(),
        &zero,
        sizeof(float),
        cudaMemcpyHostToDevice
    );

    int threadsPerBlock = 256;

    int blocksPerGrid =
        (numElements + threadsPerBlock - 1)
        / threadsPerBlock;

    vectorNorm<<<blocksPerGrid, threadsPerBlock>>>(
        deviceVector.data(),
        deviceNorm.data(),
        p,
        numElements
    );

    cudaDeviceSynchronize();

    cudaMemcpy(
        hostNorm,
        deviceNorm.data(),
        sizeof(float),
        cudaMemcpyDeviceToHost
    );

    *hostNorm = powf(*hostNorm, 1.0f / p);

    std::cout << "Vector Norm: "
              << *hostNorm
              << std::endl;

    free(hostVector);
    free(hostNorm);

    return 0;
}

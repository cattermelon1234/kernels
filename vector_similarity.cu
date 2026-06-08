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
void similarity(
    const float* a,
    const float* b,
    float* dot,
    float* normA,
    float* normB,
    int numElements)
{
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    extern __shared__ float shared[];

    float* dotShared = shared;
    float* normAShared = shared + blockDim.x;
    float* normBShared = shared + 2 * blockDim.x;

    float localDot = 0.0f;
    float localNormA = 0.0f;
    float localNormB = 0.0f;

    while (idx < numElements) {
        float av = a[idx];
        float bv = b[idx];

        localDot += av * bv;
        localNormA += av * av;
        localNormB += bv * bv;

        idx += blockDim.x * gridDim.x;
    }

    dotShared[tid] = localDot;
    normAShared[tid] = localNormA;
    normBShared[tid] = localNormB;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            dotShared[tid] += dotShared[tid + stride];
            normAShared[tid] += normAShared[tid + stride];
            normBShared[tid] += normBShared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(dot, dotShared[0]);
        atomicAdd(normA, normAShared[0]);
        atomicAdd(normB, normBShared[0]);
    }
}

float randomFloat(int randMax = 1000) {
    return static_cast<float>(rand()) /
           static_cast<float>(randMax);
}

int main() {
    srand(time(0));

    int numElements = 4096;
    size_t size = numElements * sizeof(float);

    float* hostA = (float*)malloc(size);
    float* hostB = (float*)malloc(size);

    for (int i = 0; i < numElements; i++) {
        hostA[i] = randomFloat();
        hostB[i] = randomFloat();
    }

    DeviceBuffer<float> deviceA(numElements);
    DeviceBuffer<float> deviceB(numElements);

    DeviceBuffer<float> deviceDot(1);
    DeviceBuffer<float> deviceNormA(1);
    DeviceBuffer<float> deviceNormB(1);

    cudaMemcpy(
        deviceA.data(),
        hostA,
        size,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        deviceB.data(),
        hostB,
        size,
        cudaMemcpyHostToDevice
    );

    float zero = 0.0f;

    cudaMemcpy(
        deviceDot.data(),
        &zero,
        sizeof(float),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        deviceNormA.data(),
        &zero,
        sizeof(float),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        deviceNormB.data(),
        &zero,
        sizeof(float),
        cudaMemcpyHostToDevice
    );

    int threadsPerBlock = 256;

    int blocksPerGrid =
        (numElements + threadsPerBlock - 1)
        / threadsPerBlock;

    size_t sharedMemSize =
        3 * threadsPerBlock * sizeof(float);

    similarity<<<
        blocksPerGrid,
        threadsPerBlock,
        sharedMemSize
    >>>(
        deviceA.data(),
        deviceB.data(),
        deviceDot.data(),
        deviceNormA.data(),
        deviceNormB.data(),
        numElements
    );

    cudaDeviceSynchronize();

    float dot;
    float normA;
    float normB;

    cudaMemcpy(
        &dot,
        deviceDot.data(),
        sizeof(float),
        cudaMemcpyDeviceToHost
    );

    cudaMemcpy(
        &normA,
        deviceNormA.data(),
        sizeof(float),
        cudaMemcpyDeviceToHost
    );

    cudaMemcpy(
        &normB,
        deviceNormB.data(),
        sizeof(float),
        cudaMemcpyDeviceToHost
    );

    float cosine =
        dot / (sqrtf(normA) * sqrtf(normB));

    std::cout << "Dot Product: " << dot << std::endl;
    std::cout << "Norm A: " << sqrtf(normA) << std::endl;
    std::cout << "Norm B: " << sqrtf(normB) << std::endl;
    std::cout << "Cosine Similarity: " << cosine << std::endl;

    free(hostA);
    free(hostB);

    return 0;
}

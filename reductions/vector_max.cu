#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cfloat>

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
void vector_max(const float* input,
                float* blockMaxes,
                int n)
{
    __shared__ float s[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float localMax = -FLT_MAX;

    while (idx < n) {
        localMax = fmaxf(localMax, input[idx]);
        idx += blockDim.x * gridDim.x;
    }

    s[tid] = localMax;
    __syncthreads();

    for (int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2)
    {
        if (tid < stride) {
            s[tid] = fmaxf(s[tid], s[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        blockMaxes[blockIdx.x] = s[0];
    }
}

float randomFloat(int randMax = 1000)
{
    return static_cast<float>(rand()) /
           static_cast<float>(randMax);
}

void demo_vector_max()
{
    srand(time(0));

    int numElements = 300;
    size_t size = numElements * sizeof(float);

    float* hostA =
        static_cast<float*>(malloc(size));

    for (int i = 0; i < numElements; ++i) {
        hostA[i] = randomFloat();
    }

    float* deviceA;
    cudaMalloc(&deviceA, size);

    cudaMemcpy(deviceA,
               hostA,
               size,
               cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid =
        (numElements + threadsPerBlock - 1)
        / threadsPerBlock;

    float* deviceBlockMaxes;
    cudaMalloc(&deviceBlockMaxes,
               blocksPerGrid * sizeof(float));

    vector_max<<<blocksPerGrid,
                 threadsPerBlock>>>(
        deviceA,
        deviceBlockMaxes,
        numElements);

    float* hostBlockMaxes =
        static_cast<float*>(
            malloc(blocksPerGrid * sizeof(float)));

    cudaMemcpy(hostBlockMaxes,
               deviceBlockMaxes,
               blocksPerGrid * sizeof(float),
               cudaMemcpyDeviceToHost);

    float maxVal = -FLT_MAX;

    for (int i = 0; i < blocksPerGrid; ++i) {
        maxVal = std::max(maxVal,
                          hostBlockMaxes[i]);
    }

    std::cout << "Max value: "
              << maxVal
              << std::endl;

    free(hostA);
    free(hostBlockMaxes);

    cudaFree(deviceA);
    cudaFree(deviceBlockMaxes);

    return;
}

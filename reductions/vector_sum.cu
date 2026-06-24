#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t n) {
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
float warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

template <int BLOCK_SIZE>
__device__ __forceinline__
float block_reduce_sum(float value) {
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of warp size");

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    __shared__ float warp_sums[BLOCK_SIZE / 32];

    value = warp_reduce_sum(value);

    if (lane == 0) {
        warp_sums[warp] = value;
    }

    __syncthreads();

    value = tid < (BLOCK_SIZE / 32) ? warp_sums[lane] : 0.0f;

    if (warp == 0) {
        value = warp_reduce_sum(value);
    }

    return value;
}

template <int BLOCK_SIZE>
__global__
void vectorSum(const float* __restrict__ vector,
               float* __restrict__ sum,
               int numElements)
{
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float localSum = 0.0f;

    while (idx < numElements) {
        localSum += vector[idx];
        idx += stride;
    }

    float blockSum = block_reduce_sum<BLOCK_SIZE>(localSum);

    if (tid == 0) {
        atomicAdd(sum, blockSum);
    }
}

float randomFloat(int randMax = 1000) {
    return static_cast<float>(rand()) /
           static_cast<float>(randMax);
}

int main() {
    srand(time(0));

    constexpr int numElements = 4096;
    constexpr int threadsPerBlock = 256;

    size_t size = numElements * sizeof(float);

    float* hostVector = static_cast<float*>(malloc(size));
    float hostExpected = 0.0f;

    for (int i = 0; i < numElements; ++i) {
        hostVector[i] = randomFloat();
        hostExpected += hostVector[i];
    }

    DeviceBuffer<float> deviceVector(numElements);
    DeviceBuffer<float> deviceSum(1);

    cudaMemcpy(
        deviceVector.data(),
        hostVector,
        size,
        cudaMemcpyHostToDevice
    );

    cudaMemset(deviceSum.data(), 0, sizeof(float));

    int blocksPerGrid =
        (numElements + threadsPerBlock - 1) /
        threadsPerBlock;

    vectorSum<threadsPerBlock><<<blocksPerGrid, threadsPerBlock>>>(
        deviceVector.data(),
        deviceSum.data(),
        numElements
    );

    cudaDeviceSynchronize();

    float hostSum = 0.0f;

    cudaMemcpy(
        &hostSum,
        deviceSum.data(),
        sizeof(float),
        cudaMemcpyDeviceToHost
    );

    std::cout << "Vector Sum: " << hostSum << std::endl;
    std::cout << "CPU Sum: " << hostExpected << std::endl;
    std::cout << "Abs Error: " << fabsf(hostSum - hostExpected) << std::endl;

    free(hostVector);

    return 0;
}

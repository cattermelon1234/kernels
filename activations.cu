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

__global__ void relu(float* a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float4* a4 = reinterpret_cast<float4*>(a);
    int vec_n = n / 4;

    for (int i = idx; i < vec_n; i += stride) {
        float4 v = a4[i];
        v.x = fmaxf(v.x, 0.0f);
        v.y = fmaxf(v.y, 0.0f);
        v.z = fmaxf(v.z, 0.0f);
        v.w = fmaxf(v.w, 0.0f);
        a4[i] = v;
    }

    for (int i = vec_n * 4 + idx; i < n; i += stride) {
        a[i] = fmaxf(a[i], 0.0f);
    }
}

__global__ void relu_back(float* a, float* grad_in, float* grad_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float4* grad_out4 = reinterpret_cast<float4*>(grad_out);
    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* grad_in4 = reinterpret_cast<const float4*>(grad_in);
    int vec_n = n / 4;

    for (int i = idx; i < vec_n; i += stride) {
        float4 av = a4[i];
        float4 gv = grad_in4[i];
        grad_out4[i] = make_float4(
            av.x > 0.0f ? gv.x : 0.0f,
            av.y > 0.0f ? gv.y : 0.0f,
            av.z > 0.0f ? gv.z : 0.0f,
            av.w > 0.0f ? gv.w : 0.0f
        );
    }

    for (int i = vec_n * 4 + idx; i < n; i += stride) {
      grad_out[i] = (a[i] > 0.0f) ? grad_in[i] : 0.0f;
    }
}

__global__ void dot_backward(
    const float* a,
    const float* b,
    float grad_out,   
    float* grad_a,
    float* grad_b,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        grad_a[idx] = grad_out * b[idx];
        grad_b[idx] = grad_out * a[idx];
    }
}

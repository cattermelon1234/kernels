#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define TILE 16

__global__
void matrixBroadcast(const float* a, float* c, int width, int height, float scalar) {
    /*
    Broadcasting is a technique used in matrix operations to apply an operation to all elements in a matrix.
    For example, to multiply all elements in a matrix by a scalar.
    */
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        c[row * width + col] = a[row * width + col] * scalar;
    }
}

__global__
void optimizedBroadcast(const float* x, const float* bias, float* out, int batch, int hidden) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // important! col is strided by 4 to prevent duplicate work
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (row < batch) {

        // vectorized load of 4 elements at row, col
        if (col + 3 < hidden) {

            // compiler thinks we are loading a float4 type (16 bytes) from bias and from x
            float4 x4 = reinterpret_cast<const float4*>(&x[row * hidden + col])[0];
            float4 b4 = reinterpret_cast<const float4*>(&bias[col])[0];

            // add
            x4.x += b4.x;
            x4.y += b4.y;
            x4.z += b4.z;
            x4.w += b4.w;

            // compiler thinks we are modifying a float4 type (16 bytes) at row * hidden + col
            reinterpret_cast<float4*>(&out[row * hidden + col])[0] = x4;
        }
        else {

            // tail handling for hidden dimensions not divisible by 4
            for (int i = col; i < hidden; ++i) {
                out[row * hidden + i] =
                    x[row * hidden + i] + bias[i];
            }
        }
    }

    // note: this requires 16-byte aligned memory, which cudaMalloc supports.
}

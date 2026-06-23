#pragma once

#include <cuda_runtime.h>

namespace epilogues {

struct Gelu {
    __device__ __forceinline__
    float operator()(float x, int /*row*/, int /*col*/) const {
        constexpr float kAlpha = 0.7978845608028654f; // sqrt(2 / pi)
        constexpr float kBeta = 0.044715f;
        float x3 = x * x * x;
        return 0.5f * x * (1.0f + tanhf(kAlpha * (x + kBeta * x3)));
    }
};

} // namespace epilogues

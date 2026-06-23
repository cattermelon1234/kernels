#pragma once

#include "bias.cuh"
#include "gelu.cuh"

namespace epilogues {

struct BiasGelu {
    const float* bias;

    __device__ __forceinline__
    float operator()(float x, int row, int col) const {
        BiasAdd add_bias{bias};
        Gelu gelu{};
        return gelu(add_bias(x, row, col), row, col);
    }
};

} // namespace epilogues

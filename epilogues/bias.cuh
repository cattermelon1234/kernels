#pragma once

namespace epilogues {

struct BiasAdd {
    const float* bias;

    __device__ __forceinline__
    float operator()(float x, int /*row*/, int col) const {
        return x + bias[col];
    }
};

} // namespace epilogues

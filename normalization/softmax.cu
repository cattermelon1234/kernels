#include "softmax.cuh"

#include <iostream>
#include <vector>

void demo_softmax() {
    constexpr int rows = 4;
    constexpr int cols = 4099;
    constexpr float temperature = 0.7f;
    const int tiles =
        (cols + kernels::kSoftmaxBlockSize - 1) / kernels::kSoftmaxBlockSize;

    std::vector<float> h_logits(rows * cols);
    std::vector<float> h_output(rows * cols);
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < cols; ++col) h_logits[row * cols + col] = 0.001f * col;

    float *d_logits, *d_output, *d_workspace, *d_max, *d_sum;
    cudaMalloc(&d_logits, h_logits.size() * sizeof(float));
    cudaMalloc(&d_output, h_output.size() * sizeof(float));
    cudaMalloc(&d_workspace, rows * tiles * sizeof(float));
    cudaMalloc(&d_max, rows * sizeof(float));
    cudaMalloc(&d_sum, rows * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), h_logits.size() * sizeof(float), cudaMemcpyHostToDevice);

    kernels::softmax_temperature(d_logits, d_output, rows, cols, temperature,
                                 d_workspace, d_max, d_sum);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    float probability_sum = 0.0f;
    for (int col = 0; col < cols; ++col) probability_sum += h_output[col];
    std::cout << "row 0 probability sum: " << probability_sum << '\n';

    cudaFree(d_logits); cudaFree(d_output); cudaFree(d_workspace);
    cudaFree(d_max); cudaFree(d_sum);
    return;
}

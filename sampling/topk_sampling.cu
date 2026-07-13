#include "../normalization/softmax.cuh"

#include <cfloat>
#include <climits>
#include <iostream>
#include <vector>

// Intentionally straightforward decoding pipeline:
// logits -> softmax(logits / temperature) -> top-k probabilities -> sample.
constexpr int kTopKBlockSize = 256;
constexpr int kMaxTopK = 32;

struct Candidate {
    float probability;
    int index;
};

__device__ __forceinline__ bool better(Candidate a, Candidate b) {
    return a.probability > b.probability ||
           (a.probability == b.probability && a.index < b.index);
}

template <int K>
__device__ __forceinline__ void insert(Candidate (&topk)[K], Candidate value) {
    if (!better(value, topk[K - 1])) return;
    topk[K - 1] = value;
    for (int i = K - 1; i > 0 && better(topk[i], topk[i - 1]); --i) {
        Candidate tmp = topk[i]; topk[i] = topk[i - 1]; topk[i - 1] = tmp;
    }
}

template <int K>
__device__ __forceinline__ void merge(Candidate (&dst)[K], const Candidate (&src)[K]) {
    #pragma unroll
    for (int i = 0; i < K; ++i) insert(dst, src[i]);
}

// Reduces K sorted candidates per thread to K candidates for the whole block.
template <int K>
__device__ __forceinline__ void block_topk(Candidate (&topk)[K], Candidate* warp_topk) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        Candidate other[K];
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            other[i].probability = __shfl_down_sync(0xffffffff, topk[i].probability, offset);
            other[i].index = __shfl_down_sync(0xffffffff, topk[i].index, offset);
        }
        if (lane < offset) merge(topk, other);
    }
    if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < K; ++i) warp_topk[warp * K + i] = topk[i];
    }
    __syncthreads();

    if (warp == 0) {
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            topk[i] = lane < kTopKBlockSize / 32 ? warp_topk[lane * K + i]
                                                  : Candidate{-FLT_MAX, INT_MAX};
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            Candidate other[K];
            #pragma unroll
            for (int i = 0; i < K; ++i) {
                other[i].probability = __shfl_down_sync(0xffffffff, topk[i].probability, offset);
                other[i].index = __shfl_down_sync(0xffffffff, topk[i].index, offset);
            }
            if (lane < offset) merge(topk, other);
        }
    }
}

// First reduction: each vocabulary tile becomes K candidates.
template <int K>
__global__ void partial_topk_kernel(const float* probabilities, Candidate* partial_topk,
                                    int vocab_size) {
    const int row = blockIdx.y;
    const int tile = blockIdx.x;
    Candidate topk[K];
    #pragma unroll
    for (int i = 0; i < K; ++i) topk[i] = {-FLT_MAX, INT_MAX};
    const int col = tile * blockDim.x + threadIdx.x;
    if (col < vocab_size) insert(topk, {probabilities[row * vocab_size + col], col});

    __shared__ Candidate warp_topk[(kTopKBlockSize / 32) * K];
    block_topk<K>(topk, warp_topk);
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < K; ++i) partial_topk[(row * gridDim.x + tile) * K + i] = topk[i];
    }
}

// Second reduction: all tile candidates become the row's final top K.
template <int K>
__global__ void finalize_topk_kernel(const Candidate* partial_topk, Candidate* topk,
                                     int tiles) {
    const int row = blockIdx.x;
    Candidate local[K];
    #pragma unroll
    for (int i = 0; i < K; ++i) local[i] = {-FLT_MAX, INT_MAX};
    for (int i = threadIdx.x; i < tiles * K; i += blockDim.x) {
        insert(local, partial_topk[row * tiles * K + i]);
    }
    __shared__ Candidate warp_topk[(kTopKBlockSize / 32) * K];
    block_topk<K>(local, warp_topk);
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < K; ++i) topk[row * K + i] = local[i];
    }
}

// Sample from the selected probabilities after renormalizing their mass.
template <int K>
__global__ void sample_topk_kernel(const Candidate* topk, int* sampled_token,
                                   const float* uniforms) {
    const int row = blockIdx.x;
    if (threadIdx.x != 0) return;
    float mass = 0.0f;
    for (int i = 0; i < K; ++i) mass += topk[row * K + i].probability;
    const float u = uniforms[row] * mass;  // uniforms[row] must be in [0, 1).
    float cumulative = 0.0f;
    int chosen = topk[row * K + K - 1].index;
    for (int i = 0; i < K; ++i) {
        cumulative += topk[row * K + i].probability;
        if (u < cumulative) { chosen = topk[row * K + i].index; break; }
    }
    sampled_token[row] = chosen;
}

// Workspace sizes:
// softmax_workspace: rows * ceil(vocab_size / 256) floats
// partial_topk: rows * ceil(vocab_size / 256) * K Candidates
template <int K>
void topk_sample(const float* logits, int* sampled_token, int rows, int vocab_size,
                 float temperature, const float* uniforms, float* probabilities,
                 float* softmax_workspace, float* row_max, float* row_sum,
                 Candidate* partial_topk, Candidate* topk) {
    static_assert(K > 0 && K <= kMaxTopK, "K must be in [1, 32]");
    if (rows <= 0 || vocab_size < K || temperature <= 0.0f) return;
    const int tiles = (vocab_size + kTopKBlockSize - 1) / kTopKBlockSize;
    softmax_cuda::softmax_temperature(logits, probabilities, rows, vocab_size, temperature,
                                      softmax_workspace, row_max, row_sum);
    partial_topk_kernel<K><<<dim3(tiles, rows), kTopKBlockSize>>>(probabilities, partial_topk, vocab_size);
    finalize_topk_kernel<K><<<rows, kTopKBlockSize>>>(partial_topk, topk, tiles);
    sample_topk_kernel<K><<<rows, 1>>>(topk, sampled_token, uniforms);
}

void demo_topk_sampling() {
    constexpr int rows = 3, vocab_size = 4099, K = 8;
    constexpr float temperature = 0.8f;
    const int tiles = (vocab_size + kTopKBlockSize - 1) / kTopKBlockSize;
    std::vector<float> h_logits(rows * vocab_size);
    const std::vector<float> h_uniforms{0.15f, 0.42f, 0.77f};
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < vocab_size; ++col) h_logits[row * vocab_size + col] = 0.01f * ((col * 17 + row) % 997);

    float *d_logits, *d_uniforms, *d_probabilities, *d_softmax_workspace, *d_max, *d_sum;
    int* d_tokens; Candidate *d_partial_topk, *d_topk;
    cudaMalloc(&d_logits, h_logits.size() * sizeof(float)); cudaMalloc(&d_tokens, rows * sizeof(int));
    cudaMalloc(&d_uniforms, rows * sizeof(float));
    cudaMalloc(&d_probabilities, h_logits.size() * sizeof(float)); cudaMalloc(&d_softmax_workspace, rows * tiles * sizeof(float));
    cudaMalloc(&d_max, rows * sizeof(float)); cudaMalloc(&d_sum, rows * sizeof(float));
    cudaMalloc(&d_partial_topk, rows * tiles * K * sizeof(Candidate)); cudaMalloc(&d_topk, rows * K * sizeof(Candidate));
    cudaMemcpy(d_logits, h_logits.data(), h_logits.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uniforms, h_uniforms.data(), rows * sizeof(float), cudaMemcpyHostToDevice);
    topk_sample<K>(d_logits, d_tokens, rows, vocab_size, temperature, d_uniforms, d_probabilities,
                   d_softmax_workspace, d_max, d_sum, d_partial_topk, d_topk);

    std::vector<int> h_tokens(rows); cudaMemcpy(h_tokens.data(), d_tokens, rows * sizeof(int), cudaMemcpyDeviceToHost);
    for (int row = 0; row < rows; ++row) std::cout << "sampled token for row " << row << ": " << h_tokens[row] << '\n';
    cudaFree(d_logits); cudaFree(d_tokens); cudaFree(d_uniforms); cudaFree(d_probabilities); cudaFree(d_softmax_workspace);
    cudaFree(d_max); cudaFree(d_sum); cudaFree(d_partial_topk); cudaFree(d_topk);
    return;
}

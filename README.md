kernels from scratch

## repo

- `elementwise/`: independent per-element kernels such as add, broadcast, and activations.
- `reductions/`: kernels that reduce many inputs into fewer outputs, such as sum, dot, norm, max, and similarity.
- `normalization/`: normalization-style kernels such as softmax and layernorm.
- `sampling/`: decoding kernels, including temperature-scaled exact top-k sampling.
- `layout/`: memory layout transforms such as transpose.
- `gemm/`: GEMM, tensor core GEMM, linear, and GEMM helper code.
- `epilogues/`: reusable output epilogues for GEMM-style kernels.
- `benchmarks/`: benchmark entry points.
- `tests/`: benchmark/test scripts and shared test helpers.
- `include/`: shared utility headers.
- `docs/`: notes and writeups

## decoding examples

`normalization/softmax.cu` implements `softmax_temperature`: a stable,
temperature-scaled row softmax whose max and sum reductions are tiled for rows
larger than one block. `sampling/topk_sampling.cu` implements
`topk_sample<K>` for `1 <= K <= 32`. It calls `softmax_temperature`, reduces
each probability row to its exact top-k, then samples from that top-k mass.

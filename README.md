kernels from scratch

## repo

- `elementwise/`: independent per-element kernels such as add, broadcast, and activations.
- `reductions/`: kernels that reduce many inputs into fewer outputs, such as sum, dot, norm, max, and similarity.
- `normalization/`: normalization-style kernels such as softmax and layernorm.
- `layout/`: memory layout transforms such as transpose.
- `gemm/`: GEMM, tensor core GEMM, linear, and GEMM helper code.
- `epilogues/`: reusable output epilogues for GEMM-style kernels.
- `benchmarks/`: benchmark entry points.
- `tests/`: benchmark/test scripts and shared test helpers.
- `include/`: shared utility headers.
- `docs/`: notes and writeups

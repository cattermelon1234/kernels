#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$(mktemp -d "${TMPDIR:-/tmp}/gemm-bench.XXXXXX")"

compile_and_run() {
  local src="$1"
  local exe="$2"
  shift 2

  echo "==> ${src}"
  nvcc -std=c++17 -I"${root_dir}" "$@" "${root_dir}/${src}" -o "${build_dir}/${exe}"
  "${build_dir}/${exe}"
  echo
}

compile_and_run benchmarks/bench_gemm.cu bench_gemm
compile_and_run benchmarks/bench_optimized_gemm.cu bench_optimized_gemm
compile_and_run benchmarks/bench_tensor_core_gemm.cu bench_tensor_core_gemm -arch=sm_80
compile_and_run benchmarks/bench_hyperoptimized_gemm.cu bench_hyperoptimized_gemm -arch=sm_80

echo "build artifacts: ${build_dir}"

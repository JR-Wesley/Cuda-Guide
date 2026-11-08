#!/usr/bin/env bash
set -euo pipefail
source ../record.sh
[[ $# == 0 ]] || { echo 'Usage: CUDA_ARCH=sm_XX bash run.sh' >&2; exit 2; }
arch=${CUDA_ARCH:?Set CUDA_ARCH=sm_XX explicitly}
[[ "$arch" =~ ^sm_[0-9]+$ ]] || { echo 'CUDA_ARCH must be sm_XX' >&2; exit 2; }
compute_arch=${arch/sm_/compute_}
binary=../../build/$arch/vector_transpose
mkdir -p "../../build/$arch"
rm -f -- "$binary"
rc=0
record build "${NVCC:-nvcc}" -std=c++17 -O3 -lineinfo -cudart=shared -Xptxas=-v \
  -gencode "arch=$compute_arch,code=$arch" \
  -gencode "arch=$compute_arch,code=$compute_arch" \
  ../../examples/vector_transpose.cu -o "$binary" || rc=$?
if ((rc != 0)); then rm -f -- "$binary"; exit "$rc"; fi
require_file "$binary"

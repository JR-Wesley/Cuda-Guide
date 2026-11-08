#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${CUDA_ARCH:-sm_89}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid CUDA_ARCH' >&2; exit 2; }
nvcc=${NVCC:-nvcc}
source "$root/../../tools/lab-common.sh"
lab_init "${arch}-release" main

bin="$out/bin/eviction_priority"
step compiler "$nvcc" --version
step build "$nvcc" -std=c++17 -O3 -lineinfo \
    -gencode "arch=${arch/sm_/compute_},code=$arch" \
    -gencode "arch=${arch/sm_/compute_},code=${arch/sm_/compute_}" \
    "$root/main.cu" -o "$bin"
step ptx "$nvcc" -std=c++17 -O3 -lineinfo -arch="$arch" -ptx \
    "$root/main.cu" -o "$out/eviction_priority.ptx"

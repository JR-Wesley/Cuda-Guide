#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${CUDA_ARCH:-sm_89}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid CUDA_ARCH' >&2; exit 2; }
nvcc=${NVCC:-nvcc}
source "$root/../../tools/lab-common.sh"
lab_init "${arch}-release" main

step compiler "$nvcc" --version
flags=(-std=c++17 -O3 -lineinfo -Xcompiler -pthread -ldl
    -gencode "arch=${arch/sm_/compute_},code=$arch"
    -gencode "arch=${arch/sm_/compute_},code=${arch/sm_/compute_}")
for name in pipeline events_graph errors; do
    step "build_$name" "$nvcc" "${flags[@]}" --default-stream legacy "$root/$name.cu" -o "$out/bin/$name"
done
step build_legacy "$nvcc" "${flags[@]}" --default-stream legacy -DLAB_PER_THREAD=0 "$root/ordering.cu" -o "$out/bin/ordering_legacy"
step build_per_thread "$nvcc" "${flags[@]}" --default-stream per-thread -DLAB_PER_THREAD=1 "$root/ordering.cu" -o "$out/bin/ordering_per_thread"

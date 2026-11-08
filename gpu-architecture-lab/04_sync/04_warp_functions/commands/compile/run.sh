#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${CUDA_ARCH:-sm_80}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid CUDA_ARCH' >&2; exit 2; }
nvcc=${NVCC:-nvcc}
mode=${BUILD_MODE:-release}
[[ "$mode" == release || "$mode" == debug ]] || { echo "BUILD_MODE must be release or debug" >&2; exit 2; }
source "$root/../../tools/lab-common.sh"
lab_init "${arch}-$mode" main

step compiler "$nvcc" --version
flags=(-std=c++17 -O3 -lineinfo -Xcompiler=-Wall,-Wextra
    -gencode "arch=${arch/sm_/compute_},code=$arch"
    -gencode "arch=${arch/sm_/compute_},code=${arch/sm_/compute_}")
if [[ "$mode" == debug ]]; then
    flags=(-std=c++17 -O0 -g -G -arch="$arch")
fi
step build "$nvcc" "${flags[@]}" "$root/main.cu" -o "$out/bin/warp_functions"
if [[ "$mode" == release ]]; then
step ptx "$nvcc" -std=c++17 -O3 -lineinfo -arch="$arch" -ptx "$root/main.cu" -o "$out/main.ptx"
fi

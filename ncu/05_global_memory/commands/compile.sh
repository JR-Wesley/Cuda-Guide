#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mkdir -p log
exec > >(tee log/compile.log) 2>&1
trap 'printf "exit_code=%s\n" "$?"' EXIT
set -x

: "${CUDA_ARCH:?Set CUDA_ARCH explicitly, e.g. sm_89}"
mkdir -p build
rm -f build/lesson05
nvcc -std=c++17 -O3 -lineinfo -arch="$CUDA_ARCH" kernels/main.cu -o build/lesson05

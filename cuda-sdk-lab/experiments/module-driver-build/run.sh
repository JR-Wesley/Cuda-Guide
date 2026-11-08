#!/usr/bin/env bash
set -euo pipefail
source ../record.sh
[[ $# == 0 ]] || { echo 'Usage: bash run.sh' >&2; exit 2; }
arch=${CUDA_ARCH:?Set CUDA_ARCH=sm_XX explicitly}
[[ "$arch" =~ ^sm_[0-9]+$ ]] || { echo 'CUDA_ARCH must look like sm_89' >&2; exit 2; }
output=../../build/$arch/module-loading/driver
mkdir -p "../../build/$arch/module-loading"
rm -f -- "$output"
rc=0
record build "${NVCC:-nvcc}" -std=c++17 -O2 -cudart=none \
  ../module-loading/driver.cpp -L"${CUDA_STUB_DIR:-/opt/cuda/lib64/stubs}" \
  -lcuda -o "$output" || rc=$?
if ((rc != 0)); then rm -f -- "$output"; exit "$rc"; fi
require_file "$output"

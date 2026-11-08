#!/usr/bin/env bash
set -euo pipefail
arch=${CUDA_ARCH:?Set CUDA_ARCH=sm_XX explicitly}
[[ "$arch" =~ ^sm_[0-9]+$ ]] || { echo 'Expected CUDA_ARCH=sm_89 (or another numeric SM)' >&2; exit 2; }
compute=${arch/sm_/compute_}
source ../record.sh
rm -f -- ./log/input.log
require_file ../compile/compiler_probe.cu
out=../../build/$arch/compile/artifacts
mkdir -p "$out"
# --dryrun prints the proposed pipeline without creating probe_dryrun.
record dryrun timeout 60s "${NVCC:-nvcc}" -std=c++17 -O3 -lineinfo \
  -gencode "arch=$compute,code=$arch" -gencode "arch=$compute,code=$compute" \
  -cudart=shared --dryrun ../compile/compiler_probe.cu -o "$out/probe_dryrun"

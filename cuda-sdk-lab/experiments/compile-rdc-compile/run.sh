#!/usr/bin/env bash
set -euo pipefail
arch=${CUDA_ARCH:?Set CUDA_ARCH=sm_XX explicitly}
[[ "$arch" =~ ^sm_[0-9]+$ ]] || { echo 'Expected CUDA_ARCH=sm_89 (or another numeric SM)' >&2; exit 2; }
compute=${arch/sm_/compute_}
source ../record.sh
rm -f -- ./log/input.log ./log/compile_device.{sh,log} ./log/compile_main.{sh,log}
out=../../build/$arch/compile/artifacts
mkdir -p "$out"
outputs=("$out/rdc_device.o" "$out/rdc_main.o")
rm -f -- "${outputs[@]}"
# Treat the pair as one compilation action; a failure invalidates both outputs.
trap 'rc=$?; if ((rc != 0)); then rm -f -- "${outputs[@]}"; fi; exit "$rc"' EXIT
require_file ../compile/rdc_device.cu
require_file ../compile/rdc_main.cu
flags=(-std=c++17 -O3 -lineinfo -gencode "arch=$compute,code=$arch" -dc)
record compile_device timeout 60s "${NVCC:-nvcc}" "${flags[@]}" \
  ../compile/rdc_device.cu -o "${outputs[0]}"
record compile_main timeout 60s "${NVCC:-nvcc}" "${flags[@]}" \
  ../compile/rdc_main.cu -o "${outputs[1]}"

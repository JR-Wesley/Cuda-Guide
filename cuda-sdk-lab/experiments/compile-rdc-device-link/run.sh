#!/usr/bin/env bash
set -euo pipefail
arch=${CUDA_ARCH:?Set CUDA_ARCH=sm_XX explicitly}
[[ "$arch" =~ ^sm_[0-9]+$ ]] || { echo 'Expected CUDA_ARCH=sm_89 (or another numeric SM)' >&2; exit 2; }
compute=${arch/sm_/compute_}
source ../record.sh
rm -f -- ./log/input.log
out=../../build/$arch/compile/artifacts
mkdir -p "$out"
output=$out/rdc_dlink.o
rm -f -- "$output"
trap 'rc=$?; if ((rc != 0)); then rm -f -- "$output"; fi; exit "$rc"' EXIT
require_file "$out/rdc_device.o"
require_file "$out/rdc_main.o"
record device_link timeout 60s "${NVCC:-nvcc}" -std=c++17 -O3 -lineinfo \
  -gencode "arch=$compute,code=$arch" --device-link \
  "$out/rdc_device.o" "$out/rdc_main.o" -o "$output"

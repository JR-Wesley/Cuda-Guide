#!/usr/bin/env bash
set -euo pipefail
arch=${CUDA_ARCH:?Set CUDA_ARCH=sm_XX explicitly}
[[ "$arch" =~ ^sm_[0-9]+$ ]] || { echo 'Expected CUDA_ARCH=sm_89 (or another numeric SM)' >&2; exit 2; }

source ../record.sh
rm -f -- ./log/input.log
out=../../build/$arch/compile/artifacts
mkdir -p "$out"
output=$out/rdc_lto_dlink.o
rm -f -- "$output"
trap 'rc=$?; if ((rc != 0)); then rm -f -- "$output"; fi; exit "$rc"' EXIT
require_file "$out/rdc_device_lto.o"
require_file "$out/rdc_main_lto.o"
record device_link timeout 60s "${NVCC:-nvcc}" -std=c++17 -O3 -lineinfo \
  -arch="$arch" -dlto --device-link \
  "$out/rdc_device_lto.o" "$out/rdc_main_lto.o" -o "$output"

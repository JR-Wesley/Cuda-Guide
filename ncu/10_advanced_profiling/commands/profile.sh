#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mkdir -p log
exec > >(tee log/profile.log) 2>&1
trap 'printf "exit_code=%s\n" "$?"' EXIT
set -x

[[ -x build/lesson10 ]] || { echo 'Run CUDA_ARCH=sm_XX bash commands/compile.sh first.'; exit 1; }
[[ ! -e log/profile.ncu-rep ]] || { echo 'Preserve log/profile.ncu-rep under a meaningful name before profiling again.'; exit 1; }
# Warmup precedes cudaProfilerStart and the NVTX push/pop range measure.
ncu --profile-from-start off --nvtx --nvtx-include 'measure/' \
    --set basic --kernel-name-base function -k vector_mul -c 1 \
    --replay-mode kernel --cache-control all --clock-control base \
    -o log/profile.ncu-rep ./build/lesson10

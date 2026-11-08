#!/usr/bin/env bash
set -euo pipefail
source ../record.sh
[[ $# == 0 ]] || { echo 'Usage: CUDA_ARCH=sm_XX bash run.sh' >&2; exit 2; }
arch=${CUDA_ARCH:?Set CUDA_ARCH=sm_XX explicitly}
[[ "$arch" =~ ^sm_[0-9]+$ ]] || { echo 'CUDA_ARCH must be sm_XX' >&2; exit 2; }
binary=../../build/$arch/vector_transpose
require_file "$binary"
# Explicitly replace only this action's fixed report and profiler intermediate.
rm -f -- ./log/timeline.nsys-rep ./log/timeline.qdstrm
record profile "${NSYS:-nsys}" profile --trace=cuda --sample=none --cpuctxsw=none \
  --force-overwrite=true --output=./log/timeline \
  "$binary" --case vector --n 257 --repeat 1
require_file ./log/timeline.nsys-rep
record correctness grep -F 'correct=yes, errors=0,' ./log/profile.log

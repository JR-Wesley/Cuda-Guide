#!/usr/bin/env bash
set -euo pipefail
source ../record.sh
[[ $# == 0 ]] || { echo 'Usage: bash run.sh' >&2; exit 2; }
arch=${CUDA_ARCH:?Set CUDA_ARCH=sm_XX explicitly}
[[ "$arch" =~ ^sm_[0-9]+$ ]] || { echo 'CUDA_ARCH must look like sm_89' >&2; exit 2; }
binary=../../build/$arch/vector_transpose
require_file "$binary"
record racecheck "${COMPUTE_SANITIZER:-compute-sanitizer}" --tool racecheck \
  --error-exitcode 86 "$binary" --case tiled --n 65 --repeat 1
record correctness grep -F 'correct=yes, errors=0,' ./log/racecheck.log
record summary grep -F 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' ./log/racecheck.log

#!/usr/bin/env bash
set -euo pipefail
source ../record.sh
[[ $# == 0 ]] || { echo 'Usage: CUDA_ARCH=sm_XX N=257 REPEAT=1 bash run.sh' >&2; exit 2; }
arch=${CUDA_ARCH:?Set CUDA_ARCH=sm_XX explicitly}
[[ "$arch" =~ ^sm_[0-9]+$ ]] || { echo 'CUDA_ARCH must be sm_XX' >&2; exit 2; }
binary=../../build/$arch/vector_transpose
require_file "$binary"
record vector "$binary" --case vector --n "${N:-257}" --repeat "${REPEAT:-1}"
record correctness grep -F 'correct=yes, errors=0,' ./log/vector.log

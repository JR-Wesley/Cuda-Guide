#!/usr/bin/env bash
set -euo pipefail
source ../record.sh
[[ $# == 0 ]] || { echo 'Usage: bash run.sh' >&2; exit 2; }
arch=${CUDA_ARCH:?Set CUDA_ARCH=sm_XX explicitly}
[[ "$arch" =~ ^sm_[0-9]+$ ]] || { echo 'CUDA_ARCH must look like sm_89' >&2; exit 2; }
binary=../../build/$arch/sanitizer/cases
require_file "$binary"
rc=0
record initcheck "${COMPUTE_SANITIZER:-compute-sanitizer}" --tool initcheck \
  --error-exitcode 86 "$binary" initcheck || rc=$?
# Keep the raw nonzero status; only the matching diagnostic proves detection.
record expected_exit test "$rc" -eq 86
record diagnostic grep -E 'Uninitialized __global__ memory read' ./log/initcheck.log
record summary grep -E 'ERROR SUMMARY: [1-9][0-9]* errors?' ./log/initcheck.log

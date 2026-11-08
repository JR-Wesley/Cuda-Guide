#!/usr/bin/env bash
set -euo pipefail
source ../record.sh
[[ $# == 0 ]] || { echo 'Usage: CUDA_ARCH=sm_XX bash run.sh' >&2; exit 2; }
arch=${CUDA_ARCH:?Set CUDA_ARCH=sm_XX explicitly}
[[ "$arch" =~ ^sm_[0-9]+$ ]] || { echo 'CUDA_ARCH must be sm_XX' >&2; exit 2; }
binary=../../build/$arch/vector_transpose
require_file "$binary"
# Rerunning this action intentionally replaces its own report, never another action's.
rm -f -- ./log/basic.ncu-repz
record profile "${NCU:-ncu}" --export ./log/basic --force-overwrite \
  --kernel-name vector_add --launch-skip 1 --launch-count 1 \
  --section LaunchStats --section SpeedOfLight \
  "$binary" --case vector --n 257 --repeat 1
require_file ./log/basic.ncu-repz
record correctness grep -F 'correct=yes, errors=0,' ./log/profile.log

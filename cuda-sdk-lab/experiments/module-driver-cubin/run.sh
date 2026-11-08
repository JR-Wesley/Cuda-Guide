#!/usr/bin/env bash
set -euo pipefail
source ../record.sh
[[ $# == 0 ]] || { echo 'Usage: bash run.sh' >&2; exit 2; }
arch=${CUDA_ARCH:?Set CUDA_ARCH=sm_XX explicitly}
[[ "$arch" =~ ^sm_[0-9]+$ ]] || { echo 'CUDA_ARCH must look like sm_89' >&2; exit 2; }
build_dir=../../build/$arch/module-loading
require_file "$build_dir/driver"
require_file "$build_dir/image.cubin"
record driver_cubin "$build_dir/driver" "$build_dir/image.cubin"

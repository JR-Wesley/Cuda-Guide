#!/usr/bin/env bash
set -euo pipefail

arch=${CUDA_ARCH:?Set CUDA_ARCH=sm_XX explicitly}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'CUDA_ARCH must be sm_XX (optional a/f suffix)' >&2; exit 2; }
[[ $# -le 1 ]] || { echo "Usage: bash $0 [input cubin]" >&2; exit 2; }
input=${1:-../../build/$arch/compile/artifacts/probe.cubin}
nvdisasm=${NVDISASM:-nvdisasm}
source ../record.sh
require_file "$input"
record tool_version "$nvdisasm" --version
record live_ranges "$nvdisasm" --print-code --print-life-ranges "$input"

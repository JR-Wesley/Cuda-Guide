#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${CUDA_ARCH:-sm_89}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid CUDA_ARCH' >&2; exit 2; }
nvcc=${NVCC:-nvcc}
arch=sm_90
source "$root/../../tools/lab-common.sh"
lab_init "${arch}-release" cluster
require_artifacts "$out/bin/cluster90" "$out/cluster90.ptx"

step cluster90 timeout 60 "$out/bin/cluster90"

#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${CUDA_ARCH:-sm_89}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid CUDA_ARCH' >&2; exit 2; }
nvcc=${NVCC:-nvcc}
source "$root/../../tools/lab-common.sh"
lab_init "${arch}-release" main
require_artifacts "$out/bin/intrinsics"

n=${ELEMENTS:-4099}
step gpu nvidia-smi
step run "$out/bin/intrinsics" "$n"
if [[ "$n" != 1 ]]; then step edge "$out/bin/intrinsics" 1; fi

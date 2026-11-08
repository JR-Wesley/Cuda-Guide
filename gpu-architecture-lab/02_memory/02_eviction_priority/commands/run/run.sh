#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${CUDA_ARCH:-sm_89}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid CUDA_ARCH' >&2; exit 2; }
nvcc=${NVCC:-nvcc}
source "$root/../../tools/lab-common.sh"
lab_init "${arch}-release" main
require_artifacts "$out/bin/eviction_priority"

samples=${SAMPLES:-11}; noise=${NOISE_KIB:-256}; hot=${HOT_KIB:-16}
bin="$out/bin/eviction_priority"
step run "$bin" "$samples" "$noise" "$hot"

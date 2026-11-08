#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${CUDA_ARCH:-sm_89}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid CUDA_ARCH' >&2; exit 2; }
nvcc=${NVCC:-nvcc}
source "$root/../../tools/lab-common.sh"
lab_init "${arch}-release" main
require_artifacts "$out/bin/eviction_priority"

samples=${SAMPLES:-1}; noise=${NOISE_KIB:-8}; hot=${HOT_KIB:-1}
bin="$out/bin/eviction_priority"
step sanitizer_version "${COMPUTE_SANITIZER:-compute-sanitizer}" --version
step memcheck "${COMPUTE_SANITIZER:-compute-sanitizer}" --tool memcheck \
    --error-exitcode 1 "$bin" "$samples" "$noise" "$hot"

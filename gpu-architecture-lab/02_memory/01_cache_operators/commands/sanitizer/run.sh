#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${CUDA_ARCH:-sm_89}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid CUDA_ARCH' >&2; exit 2; }
nvcc=${NVCC:-nvcc}
source "$root/../../tools/lab-common.sh"
lab_init "${arch}-release" main
require_artifacts "$out/bin/cache_operators"

mib=${STREAM_MIB:-1}; samples=${SAMPLES:-1}; passes=${REUSE_PASSES:-2}
bin="$out/bin/cache_operators"
step sanitizer_version "${COMPUTE_SANITIZER:-compute-sanitizer}" --version
step memcheck "${COMPUTE_SANITIZER:-compute-sanitizer}" --tool memcheck \
    --error-exitcode 1 "$bin" "$mib" "$samples" "$passes"

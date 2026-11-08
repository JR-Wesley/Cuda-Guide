#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${CUDA_ARCH:-sm_89}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid CUDA_ARCH' >&2; exit 2; }
nvcc=${NVCC:-nvcc}
source "$root/../../tools/lab-common.sh"
lab_init "${arch}-release" main
require_artifacts "$out/bin/memory_model"

rounds=${ROUNDS:-1000}; spin_limit=${SPIN_LIMIT:-100000}
bin="$out/bin/memory_model"
step memcheck "${COMPUTE_SANITIZER:-compute-sanitizer}" \
    --tool memcheck --error-exitcode 1 "$bin" "$rounds" "$spin_limit"
step synccheck "${COMPUTE_SANITIZER:-compute-sanitizer}" \
    --tool synccheck --error-exitcode 1 "$bin" "$rounds" "$spin_limit"

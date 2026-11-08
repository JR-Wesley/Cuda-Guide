#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${CUDA_ARCH:-sm_89}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid CUDA_ARCH' >&2; exit 2; }
nvcc=${NVCC:-nvcc}
source "$root/../../tools/lab-common.sh"
lab_init "${arch}-release" main
require_artifacts "$out/bin/main"

san=${COMPUTE_SANITIZER:-compute-sanitizer}
step sanitizer_version "$san" --version
echo 'SUBSET ONLY: exclude explicit parity and custom completion from racecheck; exclude parity from synccheck'
overall=0
step racecheck_rest timeout 120 "$san" --tool racecheck --error-exitcode 86 \
  --kernel-name-exclude 'kns=8exchangeILi2E' --kernel-name-exclude 'kns=completion' "$out/bin/main" 9 || overall=$?
step synccheck_rest timeout 120 "$san" --tool synccheck --error-exitcode 86 \
  --kernel-name-exclude 'kns=8exchangeILi2E' "$out/bin/main" 9 || overall=$?
exit "$overall"

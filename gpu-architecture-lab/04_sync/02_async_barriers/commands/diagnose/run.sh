#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${CUDA_ARCH:-sm_89}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid CUDA_ARCH' >&2; exit 2; }
nvcc=${NVCC:-nvcc}
source "$root/../../tools/lab-common.sh"
lab_init "${arch}-release" main
require_artifacts "$out/bin/main" "$out/bin/api_parity"

san=${COMPUTE_SANITIZER:-compute-sanitizer}
step sanitizer_version "$san" --version
overall=0
step raw_parity timeout 120 "$san" --tool racecheck --error-exitcode 86 --kernel-name 'kns=8exchangeILi2E' \
  --racecheck-trace-sync yes "$out/bin/main" 1 || overall=$?
step api_parity timeout 120 "$san" --tool racecheck --error-exitcode 86 --kernel-name 'kns=8exchangeILi2E' \
  --racecheck-trace-sync yes "$out/bin/api_parity" 1 || overall=$?
step raw_parity_9 timeout 120 "$san" --tool racecheck --error-exitcode 86 --kernel-name 'kns=8exchangeILi2E' \
  --racecheck-trace-sync yes "$out/bin/main" 9 || overall=$?
step api_parity_9 timeout 120 "$san" --tool racecheck --error-exitcode 86 --kernel-name 'kns=8exchangeILi2E' \
  --racecheck-trace-sync yes "$out/bin/api_parity" 9 || overall=$?
step completion_only timeout 120 "$san" --tool racecheck --error-exitcode 86 --kernel-name 'kns=completion' \
  "$out/bin/main" 1 || overall=$?
exit "$overall"

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
overall=0
for tool in memcheck racecheck synccheck; do
  step "$tool" timeout 120 "$san" --tool "$tool" --error-exitcode 86 "$out/bin/main" 9 || overall=$?
done
exit "$overall"

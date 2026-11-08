#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${CUDA_ARCH:-sm_80}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid CUDA_ARCH' >&2; exit 2; }
nvcc=${NVCC:-nvcc}
source "$root/../../tools/lab-common.sh"
lab_init "${arch}-release" main
require_artifacts "$out/bin/warp_functions"

# Metadata only: the CUDA runtime query below is authoritative for visible device 0.
step device nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader || true
san=${COMPUTE_SANITIZER:-compute-sanitizer}
step sanitizer_version "$san" --version
overall=0
for tool in memcheck racecheck synccheck initcheck; do
    step "$tool" timeout 120 "$san" --tool "$tool" --error-exitcode 86 \
        "$out/bin/warp_functions" quick || overall=$?
done
exit "$overall"

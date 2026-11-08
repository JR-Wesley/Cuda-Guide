#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${CUDA_ARCH:-sm_89}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid CUDA_ARCH' >&2; exit 2; }
nvcc=${NVCC:-nvcc}
source "$root/../../tools/lab-common.sh"
lab_init "${arch}-release" main
require_artifacts "$out/bin/scopes" "$out/bin/copies" "$out/bin/carveout"

step byte_count timeout 60 "$out/bin/copies" debug
step missing_optin timeout 60 "$out/bin/carveout" debug
# Negative controls are bounded, race-free logic/API errors, not UB kernels.

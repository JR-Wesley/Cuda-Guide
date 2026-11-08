#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${CUDA_ARCH:-sm_89}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid CUDA_ARCH' >&2; exit 2; }
nvcc=${NVCC:-nvcc}
source "$root/../../tools/lab-common.sh"
lab_init "${arch}-release" main
require_artifacts "$out/bin/scopes" "$out/bin/copies" "$out/bin/carveout"

step scopes timeout 60 "$out/bin/scopes" 7
step scopes_one timeout 60 "$out/bin/scopes" 1
step copies timeout 60 "$out/bin/copies"
step carveout timeout 60 "$out/bin/carveout"

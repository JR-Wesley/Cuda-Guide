#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${CUDA_ARCH:-sm_89}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid CUDA_ARCH' >&2; exit 2; }
nvcc=${NVCC:-nvcc}
source "$root/../../tools/lab-common.sh"
lab_init "${arch}-release" main

# Compile-only contract probes; intentionally rerun positive and negative controls.
step compiler "$nvcc" --version
step types python3 "$root/check_types.py" "$nvcc" "$arch" "$root/type_probe.cu" "$out" "$work"

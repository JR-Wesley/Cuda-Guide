#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${CUDA_ARCH:-sm_80}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid CUDA_ARCH' >&2; exit 2; }
nvcc=${NVCC:-nvcc}
source "$root/../../tools/lab-common.sh"
lab_init "${arch}-release" main
require_artifacts "$out/bin/warp_functions" "$out/main.ptx"

step sass "${CUOBJDUMP:-cuobjdump}" --dump-sass "$out/bin/warp_functions"
step static_check python3 "$root/check_static.py" "$out/main.ptx" "$work/sass.log"

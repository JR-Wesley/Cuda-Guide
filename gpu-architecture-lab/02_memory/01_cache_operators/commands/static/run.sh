#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${CUDA_ARCH:-sm_89}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid CUDA_ARCH' >&2; exit 2; }
nvcc=${NVCC:-nvcc}
source "$root/../../tools/lab-common.sh"
lab_init "${arch}-release" main
require_artifacts "$out/bin/cache_operators" "$out/cache_operators.ptx"

bin="$out/bin/cache_operators"
step sass bash -c '"$1" --dump-sass "$2" > "$3"' bash \
    "${CUOBJDUMP:-cuobjdump}" "$bin" "$out/cache_operators.sass"
step inspect python3 "$root/inspect_sass.py" "$out/cache_operators.sass"

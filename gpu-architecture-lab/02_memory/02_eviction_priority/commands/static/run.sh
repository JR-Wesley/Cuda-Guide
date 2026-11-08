#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${CUDA_ARCH:-sm_89}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid CUDA_ARCH' >&2; exit 2; }
nvcc=${NVCC:-nvcc}
source "$root/../../tools/lab-common.sh"
lab_init "${arch}-release" main
require_artifacts "$out/bin/eviction_priority" "$out/eviction_priority.ptx"

bin="$out/bin/eviction_priority"
step sass bash -c '"$1" --dump-sass "$2" > "$3"' bash \
    "${CUOBJDUMP:-cuobjdump}" "$bin" "$out/eviction_priority.sass"
step inspect python3 "$root/inspect.py" "$out/eviction_priority.sass" "$out/eviction_priority.ptx"

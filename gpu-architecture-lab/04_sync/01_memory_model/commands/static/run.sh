#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${CUDA_ARCH:-sm_89}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid CUDA_ARCH' >&2; exit 2; }
nvcc=${NVCC:-nvcc}
source "$root/../../tools/lab-common.sh"
lab_init "${arch}-release" main
require_artifacts "$out/bin/memory_model" "$out/memory_model.ptx"

bin="$out/bin/memory_model"
# Keep the full disassembly in an artifact rather than printing it.
step sass bash -c '"$1" --dump-sass "$2" > "$3"' bash \
    "${CUOBJDUMP:-cuobjdump}" "$bin" "$out/memory_model.sass"

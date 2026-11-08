#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${CUDA_ARCH:-sm_89}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid CUDA_ARCH' >&2; exit 2; }
nvcc=${NVCC:-nvcc}
source "$root/../../tools/lab-common.sh"
lab_init "${arch}-release" main
require_artifacts "$out/bin/scopes" "$out/bin/copies" "$out/scopes.ptx" "$out/copies.ptx"

for name in scopes copies; do
    step "${name}_sass" "${CUOBJDUMP:-cuobjdump}" --dump-sass "$out/bin/$name"
done
step static_check python3 "$root/check_static.py" "$out/scopes.ptx" "$out/copies.ptx" "$work/copies_sass.log"

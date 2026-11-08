#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${CUDA_ARCH:-sm_80}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid CUDA_ARCH' >&2; exit 2; }
nvcc=${NVCC:-nvcc}
source "$root/../../tools/lab-common.sh"
lab_init "${arch}-debug" main
require_artifacts "$out/bin/warp_functions"

# Metadata only: the CUDA runtime query below is authoritative for visible device 0.
step device nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader || true
step gdb_version "${CUDA_GDB:-cuda-gdb}" --version
step gdb timeout 60 "${CUDA_GDB:-cuda-gdb}" --batch \
    -ex 'set pagination off' -ex 'set cuda break_on_launch application' \
    -ex 'run debug' -ex 'info cuda kernels' -ex 'info cuda threads' \
    -ex 'continue' "$out/bin/warp_functions"
step gdb_check python3 -c \
    'import pathlib,sys; s=pathlib.Path(sys.argv[1]).read_text(); assert "width_debug_kernel" in s and "PASS checked_fields=" in s and "exited normally" in s, "debugger did not complete a kernel stop and validation"' \
    "$work/gdb.log"

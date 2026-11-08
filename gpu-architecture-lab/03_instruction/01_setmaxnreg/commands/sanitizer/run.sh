#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${CUDA_ARCH:-sm_90a}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid CUDA_ARCH' >&2; exit 2; }
nvcc=${NVCC:-nvcc}
case "$arch" in
    sm_90a|sm_100a|sm_110a|sm_120a) ;;
    *) echo 'This lab accepts sm_90a, sm_100a, sm_110a or sm_120a; see README for ISA support' >&2; exit 2 ;;
esac
dynamic=1
cc=${arch#sm_}; cc=${cc%a}
source "$root/../../tools/lab-common.sh"
lab_init "${arch}-dynamic" main
require_artifacts "$out/bin/setmaxnreg"

# Device inventory failure is recorded; the executable performs the decisive probe.
step device nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader || true
# Check support before invoking a tool that might report zero errors after a skip.
step preflight "$out/bin/setmaxnreg" 1 1 1 0
sanitizer=${COMPUTE_SANITIZER:-compute-sanitizer}
step sanitizer_version "$sanitizer" --version
step memcheck "$sanitizer" --tool memcheck --error-exitcode 86 \
    "$out/bin/setmaxnreg" "${BLOCKS:-2}" "${ROUNDS:-3}" "${STEPS:-2}" 0
step synccheck "$sanitizer" --tool synccheck --error-exitcode 86 \
    "$out/bin/setmaxnreg" "${BLOCKS:-2}" "${ROUNDS:-3}" "${STEPS:-2}" 0

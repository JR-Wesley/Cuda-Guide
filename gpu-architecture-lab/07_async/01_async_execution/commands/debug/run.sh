#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${CUDA_ARCH:-sm_89}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid CUDA_ARCH' >&2; exit 2; }
nvcc=${NVCC:-nvcc}
source "$root/../../tools/lab-common.sh"
lab_init "${arch}-release" main
require_artifacts "$out/bin/errors"

step launch_error "$out/bin/errors" launch
step execution_fault "$out/bin/errors" fault
step blocking_fault env CUDA_LAUNCH_BLOCKING=1 "$out/bin/errors" fault
step sanitizer_version "${COMPUTE_SANITIZER:-compute-sanitizer}" --version
rc=0
step expected_memcheck_fault "${COMPUTE_SANITIZER:-compute-sanitizer}" --tool memcheck \
    --error-exitcode 86 "$out/bin/errors" fault || rc=$?
if ((rc == 0)) || ! rg -q 'Invalid __global__ write|Invalid global write' "$work/expected_memcheck_fault.log"; then
    echo 'FAIL expected sanitizer diagnostic missing' >&2; exit 1
fi
printf 'expected_fault_diagnostic\t0\n' >> "$work/status.tsv"
echo 'PASS expected failure detected; nonzero sanitizer exit is intentional'

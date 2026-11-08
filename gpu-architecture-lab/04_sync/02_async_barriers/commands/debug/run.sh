#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${CUDA_ARCH:-sm_89}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid CUDA_ARCH' >&2; exit 2; }
nvcc=${NVCC:-nvcc}
source "$root/../../tools/lab-common.sh"
lab_init "${arch}-release" main
require_artifacts "$out/bin/debug"

step count timeout 60 "$out/bin/debug" count
san=${COMPUTE_SANITIZER:-compute-sanitizer}
step sanitizer_version "$san" --version
rc=0
step expected_race timeout 120 "$san" --tool racecheck --error-exitcode 86 "$out/bin/debug" race || rc=$?
if ((rc != 86)) || ! rg -q 'Race reported|[RW]A[RW] hazard' "$work/expected_race.log"; then
  echo 'FAIL: expected racecheck diagnostic/exit 86 not observed'; exit 1
fi
printf 'expected_race_verified\t0\n' >> "$work/status.tsv"
echo 'PASS expected shared-memory race detected'

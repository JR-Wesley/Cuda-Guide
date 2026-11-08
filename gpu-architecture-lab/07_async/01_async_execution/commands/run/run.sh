#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${CUDA_ARCH:-sm_89}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid CUDA_ARCH' >&2; exit 2; }
nvcc=${NVCC:-nvcc}
source "$root/../../tools/lab-common.sh"
lab_init "${arch}-release" main
require_artifacts "$out/bin/pipeline" "$out/bin/events_graph" "$out/bin/ordering_legacy" "$out/bin/ordering_per_thread" "$out/bin/errors"

step pipeline "$out/bin/pipeline" "${SAMPLES:-5}" "${ELEMENTS:-1048613}"
step events_graph "$out/bin/events_graph"
step ordering_legacy "$out/bin/ordering_legacy"
step ordering_per_thread "$out/bin/ordering_per_thread"
step launch_error "$out/bin/errors" launch

#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${CUDA_ARCH:-sm_89}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid CUDA_ARCH' >&2; exit 2; }
nvcc=${NVCC:-nvcc}
source "$root/../../tools/lab-common.sh"
lab_init "${arch}-release" main
require_artifacts "$out/bin/pipeline" "$out/bin/events_graph" "$out/bin/ordering_legacy"

sanitizer=${COMPUTE_SANITIZER:-compute-sanitizer}
step sanitizer_version "$sanitizer" --version
step pipeline_memcheck "$sanitizer" --tool memcheck --error-exitcode 86 "$out/bin/pipeline" 1 65573
step graph_memcheck "$sanitizer" --tool memcheck --error-exitcode 86 "$out/bin/events_graph"
step ordering_memcheck "$sanitizer" --tool memcheck --error-exitcode 86 "$out/bin/ordering_legacy"

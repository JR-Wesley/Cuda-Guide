#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${CUDA_ARCH:-sm_89}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid CUDA_ARCH' >&2; exit 2; }
nvcc=${NVCC:-nvcc}
source "$root/../../tools/lab-common.sh"
lab_init "${arch}-release" main
require_artifacts "$out/bin/pipeline"

report="$command_dir/${lab_variant}-timeline"
step nsys_version "${NSYS:-nsys}" --version
step profile "${NSYS:-nsys}" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --export=sqlite --force-overwrite=true -o "$report" "$out/bin/pipeline" 2 1048613
step trace_analysis python3 "$root/analyze_trace.py" "$report.sqlite"

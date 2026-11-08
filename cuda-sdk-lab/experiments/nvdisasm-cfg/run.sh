#!/usr/bin/env bash
set -euo pipefail

arch=${CUDA_ARCH:?Set CUDA_ARCH=sm_XX explicitly}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'CUDA_ARCH must be sm_XX (optional a/f suffix)' >&2; exit 2; }
[[ $# -le 1 ]] || { echo "Usage: bash $0 [input cubin]" >&2; exit 2; }
input=${1:-../../build/$arch/compile/artifacts/probe.cubin}
nvdisasm=${NVDISASM:-nvdisasm}
source ../record.sh
require_file "$input"
record tool_version "$nvdisasm" --version

# DOT must contain stdout only; diagnostics belong in a separate log.
printf '%q ' "$nvdisasm" --output-control-flow-graph-with-basic-blocks "$input" > ./log/cfg.sh
printf '> ./log/cfg.dot 2> ./log/cfg.log\n' >> ./log/cfg.sh
rc=0
"$nvdisasm" --output-control-flow-graph-with-basic-blocks "$input" > ./log/cfg.dot 2> ./log/cfg.log || rc=$?
printf 'cfg\t%d\n' "$rc" >> ./log/status.tsv
cat ./log/cfg.log >&2
exit "$rc"

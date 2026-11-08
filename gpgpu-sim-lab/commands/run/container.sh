#!/usr/bin/env bash
# Upstream setup_environment reads optional unset variables; do not enable nounset.
set -eo pipefail
trap 'rc=$?; printf "\ncontainer_exit_code=%s\n" "$rc"' EXIT
set -x
cd /work/repo/gpgpu-sim_distribution
export CUDA_INSTALL_PATH=/usr/local/cuda
export PATH="$CUDA_INSTALL_PATH/bin:$PATH"
export PAGER=cat GIT_PAGER=cat
source ./setup_environment release
linkage=$(ldd /work/build/vector_add)
printf '%s\n' "$linkage"
grep -E 'libcudart.so.*=> /work/repo/gpgpu-sim_distribution/lib/' <<<"$linkage"
cd /work/commands/run
timeout 300 /work/build/vector_add

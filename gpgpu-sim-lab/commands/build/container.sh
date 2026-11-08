#!/usr/bin/env bash
# Upstream setup_environment reads optional unset variables; do not enable nounset.
set -eo pipefail
trap 'rc=$?; printf "\ncontainer_exit_code=%s\n" "$rc"' EXIT
set -x
cd /work/repo/gpgpu-sim_distribution
export CUDA_INSTALL_PATH=/usr/local/cuda
export PATH="$CUDA_INSTALL_PATH/bin:$PATH"
export PAGER=cat GIT_PAGER=cat
cat /etc/os-release
nvcc --version
gcc --version
bison --version
flex --version
git --no-pager rev-parse HEAD
source ./setup_environment release
make -j2
mkdir -p /work/build
nvcc -O2 -lineinfo -gencode arch=compute_70,code=compute_70 --cudart shared \
  /work/vector_add.cu -o /work/build/vector_add

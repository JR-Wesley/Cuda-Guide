#!/usr/bin/env bash
# Upstream setup scripts read optional unset variables; do not enable nounset.
set -eo pipefail
trap 'rc=$?; printf "\ncontainer_exit_code=%s\n" "$rc"' EXIT
set -x
cd /work/repo/accel-sim-framework
export CUDA_INSTALL_PATH=/usr/local/cuda
export PATH="$CUDA_INSTALL_PATH/bin:$PATH"
export PAGER=cat GIT_PAGER=cat
# Prevent upstream setup from fetching missing dependencies implicitly.
test -f gpu-simulator/gpgpu-sim/setup_environment
test -f gpu-simulator/extern/pybind11/CMakeLists.txt
cat /etc/os-release
uname -a
g++ --version
nvcc --version
cmake --version
git --no-pager rev-parse HEAD
source gpu-simulator/setup_environment.sh release
cmake -S gpu-simulator -B gpu-simulator/build/release
cmake --build gpu-simulator/build/release --target accel-sim.out cudart -j2
cmake --install gpu-simulator/build/release

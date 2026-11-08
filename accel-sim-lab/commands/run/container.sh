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
source gpu-simulator/setup_environment.sh release
trace_path=$1
cd /work/commands/run
/work/repo/accel-sim-framework/gpu-simulator/bin/release/accel-sim.out \
  -trace "$trace_path" \
  -config gpgpusim.config \
  -config trace.config

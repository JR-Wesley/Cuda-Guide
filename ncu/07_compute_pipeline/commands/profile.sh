#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mkdir -p log
exec > >(tee log/profile.log) 2>&1
trap 'printf "exit_code=%s\n" "$?"' EXIT
set -x

[[ -x build/lesson07 ]] || { echo 'Run CUDA_ARCH=sm_XX bash commands/compile.sh first.'; exit 1; }
[[ ! -e log/profile.ncu-rep ]] || { echo 'Preserve log/profile.ncu-rep under a meaningful name before profiling again.'; exit 1; }
ncu --section LaunchStats --section SpeedOfLight \
    --section ComputeWorkloadAnalysis --section InstructionStats --section SchedulerStats \
    --kernel-name-base function -k fma_chain \
    -s 10 -c 1 --replay-mode kernel --cache-control all --clock-control base \
    -o log/profile.ncu-rep ./build/lesson07

#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${CUDA_ARCH:-sm_90a}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid CUDA_ARCH' >&2; exit 2; }
nvcc=${NVCC:-nvcc}
case "$arch" in
    sm_90a|sm_100a|sm_110a|sm_120a) ;;
    *) echo 'This lab accepts sm_90a, sm_100a, sm_110a or sm_120a; see README for ISA support' >&2; exit 2 ;;
esac
dynamic=1
cc=${arch#sm_}; cc=${cc%a}
source "$root/../../tools/lab-common.sh"
lab_init "${arch}-dynamic" main

step compile_probes python3 "$root/debug_compile.py" "$out" "${PTXAS:-ptxas}" "$work"

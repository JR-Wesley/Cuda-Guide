#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${CUDA_ARCH:-sm_90a}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid CUDA_ARCH' >&2; exit 2; }
nvcc=${NVCC:-nvcc}
mode=${BUILD_MODE:-dynamic}
if [[ "$mode" == baseline ]]; then
    arch=${CUDA_ARCH:-sm_89}
[[ "$arch" =~ ^sm_[0-9]+$ ]] || { echo 'Baseline requires sm_XX' >&2; exit 2; }
dynamic=0
cc=${arch#sm_}
elif [[ "$mode" == dynamic ]]; then
case "$arch" in
    sm_90a|sm_100a|sm_110a|sm_120a) ;;
    *) echo 'This lab accepts sm_90a, sm_100a, sm_110a or sm_120a; see README for ISA support' >&2; exit 2 ;;
esac
dynamic=1
cc=${arch#sm_}; cc=${cc%a}
else
    echo 'BUILD_MODE must be dynamic or baseline' >&2; exit 2
fi
source "$root/../../tools/lab-common.sh"
lab_init "${arch}-$mode" main

step compiler "$nvcc" --version
flags=(-std=c++17 -O3 -lineinfo --maxrregcount=128 -Xptxas=-v,-warn-spills
    -gencode "arch=${arch/sm_/compute_},code=$arch"
    -DLAB_DYNAMIC="$dynamic" -DLAB_TARGET_CC="$cc")
step build "$nvcc" "${flags[@]}" "$root/main.cu" -o "$out/bin/setmaxnreg"
if ((dynamic)); then
step ptx "$nvcc" "${flags[@]}" -ptx "$root/main.cu" -o "$out/setmaxnreg.ptx"
fi

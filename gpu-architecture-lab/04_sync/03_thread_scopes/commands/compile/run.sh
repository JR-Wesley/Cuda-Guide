#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${CUDA_ARCH:-sm_89}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid CUDA_ARCH' >&2; exit 2; }
nvcc=${NVCC:-nvcc}
target=${TARGET:-standard}
case "$target" in
    standard) group=main ;;
    cluster) arch=sm_90; group=cluster ;;
    *) echo 'TARGET must be standard or cluster' >&2; exit 2 ;;
esac
source "$root/../../tools/lab-common.sh"
lab_init "${arch}-release" "$group"

step compiler "$nvcc" --version
flags=(-std=c++17 -O3 -lineinfo -Xcompiler -pthread)
if [[ "$target" == cluster ]]; then
    step build_cluster90 "$nvcc" "${flags[@]}" -arch="$arch" "$root/cluster90.cu" -o "$out/bin/cluster90"
    step ptx_cluster90 "$nvcc" "${flags[@]}" -arch="$arch" -ptx "$root/cluster90.cu" -o "$out/cluster90.ptx"
else
    for name in scopes copies carveout system; do
        step "build_$name" "$nvcc" "${flags[@]}" -arch="$arch" "$root/$name.cu" -o "$out/bin/$name"
    done
    for name in scopes copies system; do
        step "ptx_$name" "$nvcc" "${flags[@]}" -arch="$arch" -ptx "$root/$name.cu" -o "$out/$name.ptx"
    done
fi

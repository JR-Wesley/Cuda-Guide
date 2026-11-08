#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${CUDA_ARCH:-sm_89}
[[ "$arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid CUDA_ARCH' >&2; exit 2; }
nvcc=${NVCC:-nvcc}
target=${TARGET:-standard}
case "$target" in
    standard) group=main ;;
    tx90) arch=sm_90; group=tx90 ;;
    *) echo 'TARGET must be standard or tx90' >&2; exit 2 ;;
esac
source "$root/../../tools/lab-common.sh"
lab_init "${arch}-release" "$group"

step compiler "$nvcc" --version
flags=(-std=c++17 -O3 -lineinfo -Xcompiler -Wall)
if [[ "$target" == tx90 ]]; then
step build_tx90 "$nvcc" "${flags[@]}" -arch=sm_90 "$root/tx90.cu" -o "$out/bin/tx90"
step ptx_tx90 "$nvcc" "${flags[@]}" -arch=sm_90 -ptx "$root/tx90.cu" -o "$out/tx90.ptx"
else
for name in main debug; do
  step "build_$name" "$nvcc" "${flags[@]}" -arch="$arch" "$root/$name.cu" -o "$out/bin/$name"
done
step ptx "$nvcc" "${flags[@]}" -arch="$arch" -ptx "$root/main.cu" -o "$out/main.ptx"
step build_api_parity "$nvcc" "${flags[@]}" -arch="$arch" -DLAB_API_PARITY "$root/main.cu" -o "$out/bin/api_parity"
fi

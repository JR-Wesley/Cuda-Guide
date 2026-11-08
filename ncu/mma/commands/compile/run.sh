#!/usr/bin/env bash
set -euo pipefail
action_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$action_dir/../.." && pwd)
printf 'Log: %s/output.log\n' "$action_dir"
exec >"$action_dir/output.log" 2>&1
trap 'rc=$?; printf "\nexit_code=%s\n" "$rc"' EXIT
set -x
[[ $# -le 1 ]] || { echo 'Usage: CUDA_ARCH=sm_89 bash commands/compile/run.sh [Release|Debug]' >&2; exit 2; }
build_type=${1:-Release}
[[ "$build_type" == Release || "$build_type" == Debug ]] || { echo 'Expected Release or Debug' >&2; exit 2; }
: "${CUDA_ARCH:?Set CUDA_ARCH explicitly; the current CMakeLists.txt requires sm_89}"
# Keep the requested architecture consistent with the unchanged CMake flags.
[[ "$CUDA_ARCH" == sm_89 ]] || { echo 'CMakeLists.txt hardcodes sm_89; other architectures are not supported by this entry' >&2; exit 2; }
export CXX=${CXX:-clang++}
TORCH_PATH=${TORCH_PATH:-/home/mikasa/.venvs/cuda_ops/lib/python3.13/site-packages/torch/share/cmake}
build_dir="$project_dir/build/$CUDA_ARCH/$build_type"
cmake -S "$project_dir" -B "$build_dir" -G 'Unix Makefiles' \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_PREFIX_PATH="$TORCH_PATH" \
  -DCMAKE_BUILD_TYPE="$build_type" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH#sm_}"
cmake --build "$build_dir" --config "$build_type" --parallel "${JOBS:-2}"

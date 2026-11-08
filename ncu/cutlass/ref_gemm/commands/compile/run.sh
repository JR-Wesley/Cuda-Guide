#!/usr/bin/env bash
set -euo pipefail
action_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$action_dir/../.." && pwd)
printf 'Log: %s/output.log\n' "$action_dir"
exec >"$action_dir/output.log" 2>&1
trap 'rc=$?; printf "\nexit_code=%s\n" "$rc"' EXIT
set -x
[[ $# -le 1 ]] || { echo 'Usage: CUDA_ARCH=sm_XX bash commands/compile/run.sh [Release|Debug]' >&2; exit 2; }
build_type=${1:-Release}
[[ "$build_type" == Release || "$build_type" == Debug ]] || { echo 'Expected Release or Debug' >&2; exit 2; }
: "${CUDA_ARCH:?Set CUDA_ARCH explicitly, for example sm_89 if supported by your GPU and Toolkit}"
[[ "$CUDA_ARCH" =~ ^sm_[0-9]+[a-z]?$ ]] || { echo 'CUDA_ARCH must use sm_XX format' >&2; exit 2; }
export CXX=${CXX:-clang++}
build_dir="$project_dir/build/$CUDA_ARCH/$build_type"
cmake -S "$project_dir" -B "$build_dir" -G 'Unix Makefiles' \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_BUILD_TYPE="$build_type" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH#sm_}"
cmake --build "$build_dir" --config "$build_type" --parallel "${JOBS:-2}"

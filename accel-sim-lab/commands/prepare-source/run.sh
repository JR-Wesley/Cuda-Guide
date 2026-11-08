#!/usr/bin/env bash
set -euo pipefail
action_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lab_root=$(cd "$action_dir/../.." && pwd)
printf 'Log: %s/output.log\n' "$action_dir"
exec >"$action_dir/output.log" 2>&1
trap 'rc=$?; printf "\nexit_code=%s\n" "$rc"' EXIT
set -x
cd "$lab_root"
[[ $# -eq 0 ]] || { echo 'Usage: bash commands/prepare-source/run.sh' >&2; exit 2; }

clone_pinned() {
  local url=$1 dest=$2 commit=$3
  if [[ ! -e "$dest/.git" ]]; then
    git clone --no-checkout "$url" "$dest"
    git -C "$dest" checkout --detach "$commit"
  fi
  # Existing clones are checked, never reset or checked out again.
  test "$(git --no-pager -C "$dest" rev-parse HEAD)" = "$commit"
  git --no-pager -C "$dest" --no-optional-locks status --short
}
mkdir -p repo
clone_pinned https://github.com/accel-sim/accel-sim-framework.git repo/accel-sim-framework 1c9cc831c5d948f89fa9dc8f21f6c87952f5f2ea
clone_pinned https://github.com/accel-sim/gpgpu-sim_distribution.git repo/accel-sim-framework/gpu-simulator/gpgpu-sim c24e8858eee552b3bc2919b147f14e2a88571775
clone_pinned https://github.com/pybind/pybind11.git repo/accel-sim-framework/gpu-simulator/extern/pybind11 fdec8c350aa0c7fbb552ec70d8176f8a13eb0561

#!/usr/bin/env bash
set -euo pipefail
action_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lab_root=$(cd "$action_dir/../.." && pwd)
printf 'Log: %s/output.log\n' "$action_dir"
exec >"$action_dir/output.log" 2>&1
trap 'rc=$?; printf "\nexit_code=%s\n" "$rc"' EXIT
set -x
cd "$lab_root"
[[ $# -eq 0 ]] || { echo 'Usage: bash commands/run/run.sh' >&2; exit 2; }
image=${SIM_IMAGE:-$(cat image.txt)}
trace_rel=$(cat case.txt)
test -f "$trace_rel"
[[ -x repo/accel-sim-framework/gpu-simulator/bin/release/accel-sim.out ]] || {
  echo 'Missing simulator: first run bash commands/build/run.sh' >&2
  exit 1
}
cp repo/accel-sim-framework/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM7_QV100/* "$action_dir/"
cp repo/accel-sim-framework/gpu-simulator/configs/tested-cfgs/SM7_QV100/trace.config "$action_dir/"
docker image inspect --format '{{.Id}}' "$image"
docker run --rm --pull never --network none \
  --hostname accel-sim-lab --add-host accel-sim-lab:127.0.0.1 \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  --mount "type=bind,src=$lab_root,dst=/work" --workdir /work \
  --entrypoint bash "$image" /work/commands/run/container.sh "/work/$trace_rel"
printf 'docker_exit_code=0\n'
python3 "$action_dir/check_run.py" "$action_dir/output.log" "$lab_root/$trace_rel"

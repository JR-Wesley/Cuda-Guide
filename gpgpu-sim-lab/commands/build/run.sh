#!/usr/bin/env bash
set -euo pipefail
action_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lab_root=$(cd "$action_dir/../.." && pwd)
printf 'Log: %s/output.log\n' "$action_dir"
exec >"$action_dir/output.log" 2>&1
trap 'rc=$?; printf "\nexit_code=%s\n" "$rc"' EXIT
set -x
cd "$lab_root"
[[ $# -eq 0 ]] || { echo 'Usage: bash commands/build/run.sh' >&2; exit 2; }
image=${SIM_IMAGE:-$(cat image.txt)}
docker image inspect --format '{{.Id}}' "$image"
docker run --rm --pull never --network none \
  --hostname gpgpu-sim-lab --add-host gpgpu-sim-lab:127.0.0.1 \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  --mount "type=bind,src=$lab_root,dst=/work" --workdir /work \
  --entrypoint bash "$image" /work/commands/build/container.sh

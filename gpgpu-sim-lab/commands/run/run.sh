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
[[ -x build/vector_add ]] || {
  echo 'Missing executable: first run bash commands/build/run.sh' >&2
  exit 1
}
cp repo/gpgpu-sim_distribution/configs/tested-cfgs/SM7_QV100/* "$action_dir/"
docker image inspect --format '{{.Id}}' "$image"
docker run --rm --pull never --network none \
  --hostname gpgpu-sim-lab --add-host gpgpu-sim-lab:127.0.0.1 \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  --mount "type=bind,src=$lab_root,dst=/work" --workdir /work \
  --entrypoint bash "$image" /work/commands/run/container.sh
printf 'docker_exit_code=0\n'
grep -qFx 'PASS vector_add n=257 blocks=3 threads=128 checked=257 first=1 last=769' "$action_dir/output.log"
grep -qE '^gpu_tot_sim_cycle = [1-9][0-9]*[[:space:]]*$' "$action_dir/output.log"
grep -qE '^gpu_tot_sim_insn = [1-9][0-9]*[[:space:]]*$' "$action_dir/output.log"
printf 'Numerical result and simulation statistics: PASS\n'

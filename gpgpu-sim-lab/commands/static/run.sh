#!/usr/bin/env bash
set -euo pipefail
action_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lab_root=$(cd "$action_dir/../.." && pwd)
printf 'Log: %s/output.log\n' "$action_dir"
exec >"$action_dir/output.log" 2>&1
trap 'rc=$?; printf "\nexit_code=%s\n" "$rc"' EXIT
set -x
cd "$lab_root"
[[ $# -eq 0 ]] || { echo 'Usage: bash commands/static/run.sh' >&2; exit 2; }
image=${SIM_IMAGE:-$(cat image.txt)}
[[ -x build/vector_add ]] || {
  echo 'Missing executable: first run bash commands/build/run.sh' >&2
  exit 1
}
docker image inspect --format '{{.Id}}' "$image"
docker run --rm --pull never --network none \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  --mount "type=bind,src=$lab_root,dst=/work,readonly" --workdir /work \
  --entrypoint /usr/local/cuda/bin/cuobjdump "$image" --dump-ptx /work/build/vector_add

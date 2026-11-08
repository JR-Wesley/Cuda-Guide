#!/usr/bin/env bash
set -euo pipefail
action_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lab_root=$(cd "$action_dir/../.." && pwd)
printf 'Log: %s/output.log\n' "$action_dir"
exec >"$action_dir/output.log" 2>&1
trap 'rc=$?; printf "\nexit_code=%s\n" "$rc"' EXIT
set -x
cd "$lab_root"
[[ $# -eq 0 ]] || { echo 'Usage: bash commands/prepare-traces/run.sh' >&2; exit 2; }
mkdir -p downloads traces
archive=downloads/rodinia_2.0-ft.tgz
if [[ ! -f "$archive" ]]; then
  curl -fL https://engineering.purdue.edu/tgrogers/accel-sim/traces/tesla-v100/latest/rodinia_2.0-ft.tgz \
    -o "$archive.part"
  gzip -t "$archive.part"
  mv "$archive.part" "$archive"
fi
gzip -t "$archive"
# Reuse existing traces without overwriting their files (including upstream run.sh).
tar --skip-old-files -xzf "$archive" -C traces
test -f "$(cat case.txt)"

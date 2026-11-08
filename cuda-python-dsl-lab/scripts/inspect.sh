#!/usr/bin/env bash
set -u
status=0
probe() {
    printf '\n$'; printf ' %q' "$@"; printf '\n'
    "$@"
    rc=$?
    printf 'exit_code=%d\n' "$rc"
    if (( rc != 0 )); then status=1; fi
}
probe uname -a
probe cat /etc/os-release
probe nvidia-smi
probe nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
probe bash -c 'command -v nvcc ptxas tileiras python3 uv'
probe nvcc --version
probe ptxas --version
probe tileiras --version
probe python3 --version
probe uv --version
probe cc --version
probe df -h .
exit "$status"

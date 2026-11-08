#!/usr/bin/env bash
# Low-level logging and artifact checks only; each command owns its CUDA workflow.
lab_tools=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

lab_init() {
    lab_variant=$1
    lab_group=${2:-main}
    command_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[1]}")" && pwd)
    action=${command_dir##*/}
    out="$root/build/$lab_variant"
    work="$command_dir/.steps-$lab_variant"
    mkdir -p "$root/build"
    exec 9>"$root/build/.lock"
    flock -n 9 || { echo "Another command is using $root/build" >&2; exit 2; }
    mkdir -p "$out/bin" "$work"
    log="$command_dir/$lab_variant.log"
    exec > >(tee "$log") 2>&1
    trap 'lab_finish $?' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    printf 'action=%s\nvariant=%s\nCUDA_ARCH=%s\n' "$action" "$lab_variant" "$arch"
    printf 'Build: %s\nLog: %s\n' "$out" "$log"
    printf 'CUDA_VISIBLE_DEVICES=%s\nCUDA_FORCE_PTX_JIT=%s\nCUDA_LAUNCH_BLOCKING=%s\n' \
        "${CUDA_VISIBLE_DEVICES:-unset}" "${CUDA_FORCE_PTX_JIT:-unset}" "${CUDA_LAUNCH_BLOCKING:-unset}"
    printf 'step\texit_code\n' > "$work/status.tsv"
    if [[ "$action" == compile ]]; then
        rm -f "$out/.$lab_group.compiled"
    fi
}

lab_finish() {
    local rc=$1
    trap - EXIT INT TERM
    if [[ "$action" == compile && "$rc" == 0 ]]; then
        touch "$out/.$lab_group.compiled"
    fi
    if [[ -f "$work/metadata.txt" ]]; then cat "$work/metadata.txt"; fi
    cat "$work/status.tsv"
    printf 'action_exit_code=%s\n' "$rc"
    rm -rf -- "$work"
    exit "$rc"
}

require_artifacts() {
    local file
    if [[ ! -f "$out/.$lab_group.compiled" ]]; then
        echo "Missing successful compilation for $lab_variant ($lab_group). Run commands/compile/run.sh first with the matching CUDA_ARCH, BUILD_MODE and TARGET." >&2
        return 2
    fi
    for file in "$@"; do
        if [[ ! -f "$file" ]]; then
            echo "Missing artifact: $file. Run commands/compile/run.sh first." >&2
            return 2
        fi
    done
}

step() {
    local label=$1
    shift
    python3 "$lab_tools/lab_state.py" "$work" "$label" "$@"
}

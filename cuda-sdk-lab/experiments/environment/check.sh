#!/usr/bin/env bash
# Run in the current directory; inspect tools without launching GPU workloads.
set -uo pipefail

mkdir -p log || exit 1
printf 'step\texit_code\n' > log/status.tsv
failures=0

run_step() {
  local label=$1 rc=0
  shift
  printf '%q ' "$@" > "log/$label.sh"
  printf '\n' >> "log/$label.sh"
  "$@" > "log/$label.log" 2>&1 || rc=$?
  printf '%s\t%d\n' "$label" "$rc" >> log/status.tsv
  printf '%-24s exit %d\n' "$label" "$rc"
  if ((rc != 0)); then failures=$((failures + 1)); fi
}

check_version() {
  local label=$1 tool=$2
  if command -v -- "$tool" >/dev/null 2>&1; then
    run_step "$label" "$tool" --version
  else
    rm -f -- "log/$label.sh"
    printf 'SKIPPED: %s unavailable\n' "$tool" > "log/$label.log"
    printf '%s\tSKIPPED\n' "$label" >> log/status.tsv
    printf '%-24s SKIPPED: %s unavailable\n' "$label" "$tool"
  fi
}

check_version nvcc_version "${NVCC:-nvcc}"
check_version ptxas_version "${PTXAS:-ptxas}"
check_version cuobjdump_version "${CUOBJDUMP:-cuobjdump}"
check_version nvdisasm_version "${NVDISASM:-nvdisasm}"
check_version nvlink_version "${NVLINK:-nvlink}"
check_version sanitizer_version "${COMPUTE_SANITIZER:-compute-sanitizer}"
check_version nsys_version "${NSYS:-nsys}"
check_version ncu_version "${NCU:-ncu}"
check_version smi_version "${NVIDIA_SMI:-nvidia-smi}"
if command -v -- "${NVIDIA_SMI:-nvidia-smi}" >/dev/null 2>&1; then
  run_step gpu_info "${NVIDIA_SMI:-nvidia-smi}" \
    --query-gpu=index,name,compute_cap,driver_version --format=csv,noheader
else
  rm -f -- log/gpu_info.sh
  printf 'SKIPPED: nvidia-smi unavailable\n' > log/gpu_info.log
  printf 'gpu_info\tSKIPPED\n' >> log/status.tsv
fi

printf 'Results: log/status.tsv\n'
((failures == 0))

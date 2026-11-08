#!/usr/bin/env bash
set -euo pipefail
source ../record.sh
[[ $# == 0 ]] || { echo 'Usage: bash run.sh' >&2; exit 2; }
report=../nsys/log/timeline.nsys-rep
require_file "$report"
# SQLite belongs to this statistics action, not to the capture action.
rm -f -- ./log/timeline.sqlite
record stats "${NSYS:-nsys}" stats --report cuda_gpu_kern_sum --report cuda_api_sum \
  --sqlite ./log/timeline.sqlite "$report"

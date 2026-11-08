#!/usr/bin/env bash
set -euo pipefail
source ../record.sh
[[ $# == 0 ]] || { echo 'Usage: bash run.sh' >&2; exit 2; }
report=../ncu/log/basic.ncu-repz
require_file "$report"
record details "${NCU:-ncu}" --import "$report" --page details

#!/usr/bin/env bash
# Source from an action directory. This file only records commands, never dispatches actions.
mkdir -p ./log
printf 'step\texit_code\n' > ./log/status.tsv

record() {
  local label=$1 rc=0
  shift
  printf '%q ' "$@" > "./log/$label.sh"
  printf '\n' >> "./log/$label.sh"
  "$@" > "./log/$label.log" 2>&1 || rc=$?
  printf '%s\t%d\n' "$label" "$rc" >> ./log/status.tsv
  cat "./log/$label.log"
  return "$rc"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    printf 'Missing %s; first run the explicit build described in README.md\n' "$1" > ./log/input.log
    printf 'input\t1\n' >> ./log/status.tsv
    cat ./log/input.log >&2
    return 1
  fi
}

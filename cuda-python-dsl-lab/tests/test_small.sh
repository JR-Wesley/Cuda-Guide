#!/usr/bin/env bash
# Run from project root. Existing environments only; no setup or benchmarks.
set -euo pipefail
exec python3 commands/learn/run.py "${1:-all}"

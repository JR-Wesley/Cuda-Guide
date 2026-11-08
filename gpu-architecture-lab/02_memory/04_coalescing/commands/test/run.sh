#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
# CPU 地址模型不选择 CUDA 架构；仅复用现有锁、日志和退出状态机制。
arch=not-applicable
source "$root/../../tools/lab-common.sh"
lab_init cpu

step python python3 --version
step unit python3 -B -m unittest discover -s "$root" -p test_coalescing_model.py -v

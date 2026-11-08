#!/usr/bin/env python3
"""Validate full trace completion; this does not check application values."""
import json
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
log = log_path.read_text(errors="replace")
trace_list = Path(sys.argv[2])
expected = sum(line.strip().startswith("kernel-") for line in trace_list.read_text().splitlines())
kernels = re.findall(r"^kernel_name = (.+)$", log, re.M)
assert expected > 0, "Trace has no kernels"
assert len(kernels) == expected, f"Expected {expected} completed kernels, got {len(kernels)}"
assert "GPGPU-Sim: *** simulation thread exiting ***" in log, "Missing completion marker"
assert "GPGPU-Sim: *** exit detected ***" in log, "Missing exit marker"
assert not re.search(r"deadlock detected|Assertion .*failed|maximum number of cycles|break due to reaching", log, re.I), "Simulation failure detected"
metrics = {}
for key in ("gpu_tot_sim_cycle", "gpu_tot_sim_insn"):
    values = re.findall(r"^" + key + r" = (\d+)\s*$", log, re.M)
    assert values and int(values[-1]) > 0, f"Missing/non-positive {key}"
    metrics[key] = int(values[-1])
print(json.dumps({"trace_replay": "PASS", "expected_kernels": expected,
                  "completed_kernels": len(kernels), **metrics,
                  "numerical_correctness": "not measured (trace-driven replay)"}, indent=2))

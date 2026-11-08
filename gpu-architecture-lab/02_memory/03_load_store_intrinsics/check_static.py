#!/usr/bin/env python3
"""Check PTX intent per named kernel; report (not generalize) target SASS."""
import re
import sys
from pathlib import Path

ptx = Path(sys.argv[1]).read_text()
sass = Path(sys.argv[2]).read_text()
entries = list(re.finditer(r"\.visible\s+\.entry\s+(\w+)\s*\(", ptx))
bodies = {m[1]: ptx[m.end():entries[i+1].start() if i+1 < len(entries) else len(ptx)]
          for i, m in enumerate(entries)}
functions = list(re.finditer(r"Function\s*:\s*(\w+)", sass))
machine = {m[1]: sass[m.end():functions[i+1].start() if i+1 < len(functions) else len(sass)]
           for i, m in enumerate(functions)}

checks = {"probe_load_plain": ("ld", "global", "u32"),
          "probe_load_restrict": ("ld", "global", "nc", "u32"),
          "probe_store_plain": ("st", "global", "u32"),
          "probe_float4_ca": ("ld", "global", "ca", "v4", "f32"),
          "probe_half2_ldg": ("ld", "global", "nc", "b32"),
          "probe_bfloat162_cs": ("st", "global", "cs", "b32")}
for op in ("ldg", "ca", "cg", "cs", "lu", "cv"):
    checks[f"probe_load_{op}"] = ("ld", "global", "nc" if op == "ldg" else op, "u32")
for op in ("wb", "cg", "cs", "wt"):
    checks[f"probe_store_{op}"] = ("st", "global", op, "u32")

failed = 0
for name, required in checks.items():
    instructions = re.findall(r"\b(?:ld|st)\.[\w:.]+\s+[^;]+;", bodies.get(name, ""))
    matching = [x.strip() for x in instructions
                if set(required) <= set(x.split()[0].split("."))]
    mnemonic = "LDG" if required[0] == "ld" else "STG"
    native = [line.strip() for line in machine.get(name, "").splitlines()
              if re.search(rf"\b{mnemonic}(?:\.|\s)", line)]
    ok = bool(matching and native)
    failed += not ok
    print(f"{'PASS' if ok else 'FAIL'} {name}")
    print("  PTX: " + (" | ".join(matching) or f"missing {required}; observed={instructions}"))
    print("  SASS: " + (" | ".join(native) or "missing native memory instruction"))
print(f"SUMMARY static={len(checks)} failed={failed}")
sys.exit(bool(failed))

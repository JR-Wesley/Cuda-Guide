#!/usr/bin/env python3
"""Verify PTX coverage and loads inside the measured SASS interval."""
import re
import sys
from pathlib import Path

if len(sys.argv) != 3:
    raise SystemExit("Usage: python3 inspect.py eviction_priority.sass eviction_priority.ptx")
sass, ptx = (Path(p).read_text() for p in sys.argv[1:])
hints = ["evict_normal", "evict_first", "evict_last", "evict_unchanged", "no_allocate"]
for op in ("ld", "st"):
    for space in ("", ".global"):
        for hint in hints:
            spelling = f"{op}{space}.L1::{hint}.u32"
            if spelling not in ptx:
                raise SystemExit(f"FAIL missing PTX {spelling}")
print("PASS PTX: all 20 load/store + global/generic hint spellings")
copies = probes = 0
for section in sass.split("Function : ")[1:]:
    name = section.splitlines()[0].strip()
    if "copy_hint" in name:
        copies += 1
        forms = sorted(set(re.findall(r"\b(?:LDG|STG|LD|ST)\.[A-Z0-9_.]+", section)))
        if not any(f.startswith(("LDG.", "LD.")) for f in forms) or not any(
            f.startswith(("STG.", "ST.")) for f in forms
        ):
            raise SystemExit(f"FAIL {name}: expected both load and store instructions")
        print(name, ", ".join(forms))
    elif "hot_probe" in name:
        probes += 1
        instructions = []
        for line in section.splitlines():
            match = re.search(r"/\*([0-9a-f]+)\*/\s+(.*?)\s*;", line)
            if match:
                instructions.append((int(match[1], 16), match[2]))
        clocks = [pc for pc, inst in instructions if "SR_CLOCK" in inst]
        if len(clocks) != 2:
            raise SystemExit(f"FAIL {name}: expected 2 clock reads, got {clocks}; inspect manually")
        between = [(pc, inst) for pc, inst in instructions if clocks[0] < pc < clocks[1]]
        loads = [(pc, inst) for pc, inst in between if "LDG." in inst]
        loops = []
        for pc, inst in between:
            branch = re.search(r"BRA\s+0x([0-9a-f]+)", inst)
            if branch:
                target = int(branch[1], 16)
                if any(target <= ld_pc < pc for ld_pc, _ in loads) and target < pc:
                    loops.append((target, pc))
        if not loops:
            raise SystemExit(f"FAIL {name}: no load-containing loop between clocks")
        print(f"{name}: clocks 0x{clocks[0]:x}..0x{clocks[1]:x}; measured LDG loop retained")
if copies != 10 or probes != 12:
    raise SystemExit(f"FAIL expected 10 copy and 12 probe kernels; got {copies}, {probes}")
print("PASS SASS: 10 copy kernels and 12 probes; all timed loops contain loads")

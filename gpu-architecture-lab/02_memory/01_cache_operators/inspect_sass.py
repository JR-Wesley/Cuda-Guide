#!/usr/bin/env python3
"""Inspect this experiment's cuobjdump SASS, including the actual reuse loop."""
import re
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 inspect_sass.py cache_operators.sass")
    text = Path(sys.argv[1]).read_text()
    load_names = ["default", "ca", "cg", "cs", "lu", "cv"]
    store_names = ["default", "wb", "cg", "cs", "wt"]
    seen = set()
    for section in text.split("Function : ")[1:]:
        name = section.splitlines()[0].strip()
        match = re.match(r"_Z10read_reuseIL4Load([0-5])E", name)
        store = re.match(r"_Z12write_streamIL5Store([0-4])E", name)
        if not match and not store:
            continue
        kind = "load" if match else "store"
        index = int((match or store)[1])
        op = (load_names if match else store_names)[index]
        seen.add((kind, index))
        instructions = []
        for line in section.splitlines():
            instruction = re.search(r"/\*([0-9a-f]+)\*/\s+(.*?)\s*;", line)
            if instruction:
                instructions.append((int(instruction[1], 16), instruction[2]))
        mnemonic = "LDG" if match else "STG"
        forms = sorted(set(re.findall(rf"\b{mnemonic}[A-Z0-9_.]*", section)))
        print(f"{kind}.{op}: {', '.join(forms)}")
        if match:
            loops = []
            for address, instruction in instructions:
                branch = re.search(r"BRA\s+0x([0-9a-f]+)", instruction)
                if branch and int(branch[1], 16) < address:
                    start = int(branch[1], 16)
                    loads = [(pc, inst) for pc, inst in instructions
                             if start <= pc < address and "LDG." in inst]
                    if loads:
                        loops.append((start, address, len(loads)))
            if not loops:
                raise SystemExit(f"FAIL {name}: no LDG inside a backward loop; inspect manually")
            for start, end, count in loops:
                print(f"  loop 0x{start:x}..0x{end:x}: {count} LDG instructions")
    if len(seen) != 11:
        raise SystemExit(f"FAIL expected 11 distinct variants, found {len(seen)}; inspect manually")
    print("PASS all 11 variants found; all 6 reuse kernels retain loads inside loops")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compile-only overload boundaries, with separate positive controls."""
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from lab_state import compile_command

nvcc, arch, source, output, log_directory = sys.argv[1:]
out = Path(output)
work = Path(log_directory)
failed = 0
rows = ["type\top\texpected\texit_code\tverdict"]
for typ in ("unsigned", "bool", "long double", "int1", "float3", "Pair"):
    for op, spelling in enumerate(("plain", "__ldg", "__ldca", "__stwb")):
        tag = f"type_{typ.replace(' ', '_')}_{op}"
        cmd = [nvcc, "-std=c++17", "-O3", f"-arch={arch}", "-c", source,
               f"-DPROBE_TYPE={typ}", f"-DPROBE_OP={op}", "-o", str(out / "bin" / f"{tag}.o")]
        print(shlex.join(cmd), flush=True)
        expect_success = typ == "unsigned" or op == 0
        # Both controls are fresh compilations, not cached compiler diagnostics.
        result = compile_command(cmd)
        print(result.stdout, end="", flush=True)
        # A generic compiler/tool failure is NOT an expected overload failure.
        overload_error = (spelling in result.stdout and
                          ("no instance of overloaded function" in result.stdout or
                           "no matching function" in result.stdout))
        ok = (result.returncode == 0 if expect_success else
              result.returncode != 0 and overload_error)
        failed += not ok
        expected = "compile" if expect_success else "overload rejection"
        verdict = "PASS" if ok else "FAIL"
        rows.append(f"{typ}\t{spelling}\t{expected}\t{result.returncode}\t{verdict}")
        print(f"{verdict} {typ:12} {spelling:8} expected={expected} rc={result.returncode}", flush=True)
        if not ok:
            print(result.stdout[:2000], flush=True)
print("\n".join(rows), flush=True)
(work / "types.tsv").write_text("\n".join(rows) + "\n")
print(f"SUMMARY compile_probes={len(rows)-1} failed={failed}")
sys.exit(bool(failed))

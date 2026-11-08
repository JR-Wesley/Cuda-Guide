"""PTX and SASS checks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import re
from common import execute, parser_for


def contains_loop_load(sass):
    instructions = [(int(m.group(1),16),m.group(2)) for m in re.finditer(r'/\*([0-9a-f]+)\*/([^\n]+)',sass)]
    for pc, text in instructions:
        branch = re.search(r'\bBRA\s+(0x[0-9a-f]+)', text)
        if branch:
            target = int(branch.group(1),16)
            if target < pc and any(target <= p < pc and re.search(r'\bLDG\.', op) for p,op in instructions):
                return True
    return False


def check(session, selected, directory):
    binaries = directory / "bin"
    for name in selected:
        if name == "host_transfer":
            (session.path / "host_transfer.static.txt").write_text("Not Applicable: measured work is CUDA Runtime memcpy, not an application kernel; unused kernels in the binary are not transfer evidence\n")
            continue
        for fmt in ("ptx", "sass", "resource-usage"):
            _, output = session.command(["cuobjdump", f"--dump-{fmt}", binaries / name])
            (session.path / f"{name}.{fmt}.txt").write_text(output)
            if fmt == "ptx" and (".entry" not in output or "ld.global" not in output or "st.global" not in output):
                raise RuntimeError(f"expected PTX memory operations missing in {name}")
            if fmt == "sass" and ("Function :" not in output or "LDG" not in output or "STG" not in output):
                raise RuntimeError(f"expected SASS memory operations missing in {name}; inspect parser/tool output")
            if name == "gemm" and fmt == "sass":
                if not all(op in output for op in ("FFMA", "LDS", "STS", "BAR.")):
                    raise RuntimeError("GEMM FP32/shared/barrier instructions missing")
                if re.search(r"\b(?:HMMA|IMMA|DMMA|MMA|WGMMA)(?:\.|\s)", output):
                    raise RuntimeError("Tensor Core instruction outside lab scope")
            if name == "global_copy" and fmt == "sass":
                for width in (64, 128):
                    for op in ("LDG", "STG"):
                        if not re.search(rf"\b{op}(?:\.[A-Z0-9_]+)*\.{width}(?:\.[A-Z0-9_]+)*\s", output):
                            raise RuntimeError(f"{op}.{width} absent; inspect vector lowering before interpreting results")
            if fmt == "sass" and name == "transpose" and not all(op in output for op in ("LDS", "STS", "BAR.")):
                raise RuntimeError("shared transpose operations/barrier absent; inspect SASS")
            if fmt == "sass" and name == "cache_working_set" and not contains_loop_load(output):
                raise RuntimeError("global load not found inside a SASS back edge; inspect before citing cache results")
            if name == "reduction" and fmt == "sass":
                if not all(op in output for op in ("SHFL", "BAR.", "LDS", "STS")):
                    raise RuntimeError("reduction communication operations missing; inspect SASS")
                chunks = output.split("Function :")[1:]
                masked = [c for c in chunks if "reduce_passILi3" in c.splitlines()[0]]
                if len(masked) != 1 or "MUFU.RCP" in masked[0]:
                    raise RuntimeError("masked reduction still has reciprocal or symbol parser failed; inspect before interpreting")
            if name == "integer_division" and fmt == "ptx" and not all(op in output for op in ("div.u32", "div.s32")):
                raise RuntimeError("runtime integer division absent from PTX; inspect compiler transformations")
            if name == "local_array" and fmt == "ptx" and not all(op in output for op in ("ld.local", "st.local")):
                raise RuntimeError("addressable local-array accesses absent; do not label this a local-memory comparison")

def main():
    args = parser_for(__doc__).parse_args()
    return execute("static", args.experiment, check, tool="cuobjdump")

if __name__ == "__main__":
    raise SystemExit(main())

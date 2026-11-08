"""Run GPU CTest and check CSV schemas and invalid inputs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from common import execute, parser_for
from config import SMOKE
from verification import (verify_async, verify_csv, verify_execution, verify_gemm,
                          verify_global, verify_hierarchy, verify_instructions)

def check(session, selected, directory):
    binaries = directory / "bin"
    host_filter = "^(host_tools|" + "|".join(name + "_help" for name in selected) + ")$"
    session.command(["ctest", "--test-dir", directory, "-L", "host", "-R", host_filter,
                     "--output-on-failure", "--no-tests=error"], show=True)
    label = "^" + selected[0] + "$" if len(selected) == 1 else "gpu"
    session.command(["ctest", "--test-dir", directory, "-L", label, "--output-on-failure", "--no-tests=error"], show=True)
    for name in selected:
        binary = binaries / name
        if name == "gemm":
            verify_gemm(session, binary)
            continue
        if name == "global_copy":
            verify_global(session, binary)
            continue
        if name in ("reduction", "register_pressure"):
            verify_execution(session, binary, name)
            continue
        if name in ("integer_division", "loop_unroll", "branch_flow"):
            verify_instructions(session, binary, name)
            continue
        if name in ("host_transfer", "stream_pipeline", "concurrent_kernels"):
            verify_async(session, binary, name)
            continue
        if name in SMOKE:
            verify_hierarchy(session, binary, name)
            continue
        for block in (1, 31, 128, 1024):
            dest = session.path / f"{name}_block{block}.csv"
            session.command([binary, "--n", "4099", "--block", block, "--warmup", "1", "--repeats", "2", "--iterations", "3", "--csv", dest])
            verify_csv(dest, name == "memory_copy", 2)
        bad_csv = session.path / f"{name}_invalid.csv"
        for invalid in (("--n", "0"), ("--n", "-1"), ("--n", "12x"), ("--block", "1025"),
                        ("--warmup", "0"), ("--iterations", "0"), ("--n", "1000000000"),
                        ("--variant", "unknown"), ("--n", "18446744073709551615")):
            session.command([binary, *invalid, "--csv", bad_csv], expected=1)
            assert not bad_csv.exists(), "failed validation published CSV"
        existing = session.path / f"{name}_block1.csv"
        before = existing.read_bytes()
        session.command([binary, "--csv", existing], expected=1)
        assert existing.read_bytes() == before, "overwrote evidence"
    (session.path / "verification.txt").write_text("PASS: GPU boundaries, launch sizes, CLI rejection, CSV schema/units, overwrite protection\n")

def main():
    args = parser_for(__doc__).parse_args()
    return execute("verify", args.experiment, check)

if __name__ == "__main__":
    raise SystemExit(main())

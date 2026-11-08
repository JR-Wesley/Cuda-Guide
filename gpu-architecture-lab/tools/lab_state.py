#!/usr/bin/env python3
"""Run a logged step; explicit compilation has no cache or source tracking."""
from pathlib import Path
import shlex
import subprocess
import sys


def execute(argv: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(argv, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
    except OSError as exc:
        return subprocess.CompletedProcess(argv, 127, str(exc) + "\n")


def compile_command(argv: list[str]) -> subprocess.CompletedProcess:
    """Publish only a successful output and discard stale/partial failed output."""
    output = Path(argv[argv.index("-o") + 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    result = execute(argv)
    if result.returncode == 0 and not output.is_file():
        result = subprocess.CompletedProcess(argv, 1,
                                             result.stdout + "Compiler produced no output file.\n")
    if result.returncode != 0:
        output.unlink(missing_ok=True)
    return result


def step(work: Path, label: str, argv: list[str]) -> int:
    print(f"\n[{label}] {shlex.join(argv)}", flush=True)
    # Only explicit compiler steps use -o. Profile tools use -o too, but their
    # output is a report prefix rather than a compiler output file.
    compiling = (label == "build" or label.startswith(("build_", "ptx", "cubin"))
                 or label.endswith("_ptx")) and "-o" in argv
    result = compile_command(argv) if compiling else execute(argv)
    (work / f"{label}.log").write_text(result.stdout)
    with (work / "status.tsv").open("a") as stream:
        stream.write(f"{label}\t{result.returncode}\n")
    print(result.stdout, end="", flush=True)
    print(f"[{label}] exit_code={result.returncode}", flush=True)
    return result.returncode


if __name__ == "__main__":
    sys.exit(step(Path(sys.argv[1]), sys.argv[2], sys.argv[3:]))

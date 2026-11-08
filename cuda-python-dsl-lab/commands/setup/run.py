"""Explicitly install locked environments; this action uses the network."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from command_support import NAMES, PYTHON, ROOT, Record, python_for, target_parser


def main():
    args = target_parser(__doc__).parse_args()
    record = Record("setup")
    if not record.call("python", ["uv", "python", "install", PYTHON, "--no-bin"]):
        return record.finish()
    for name in NAMES if args.target == "all" else (args.target,):
        py = python_for(name)
        lock = ROOT / "requirements" / f"{name}.lock"
        if not py.exists() and not record.call(f"{name}-venv", ["uv", "venv", "--seed", "--python", PYTHON, py.parent.parent]):
            continue
        if not lock.exists() and not record.call(f"{name}-resolve", ["uv", "pip", "compile", "--quiet", "--python", py,
                "--index-url", "https://pypi.org/simple", "--generate-hashes", ROOT / "requirements" / f"{name}.in", "-o", lock]):
            continue
        record.call(f"{name}-install", ["uv", "pip", "sync", "--python", py,
                    "--index-url", "https://pypi.org/simple", "--require-hashes", lock])
    return record.finish()


if __name__ == "__main__":
    raise SystemExit(main())

"""Check installed dependencies and imports; no installation or kernels."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from command_support import NAMES, ROOT, Record, python_for, target_parser


def main():
    args = target_parser(__doc__).parse_args()
    record = Record("check")
    for name in NAMES if args.target == "all" else (args.target,):
        py = python_for(name)
        record.call(f"{name}-dependencies", ["uv", "pip", "check", "--python", py])
        record.call(f"{name}-freeze", ["uv", "pip", "freeze", "--python", py])
        record.call(f"{name}-imports", [py, ROOT / "scripts/check.py", name])
    return record.finish()


if __name__ == "__main__":
    raise SystemExit(main())

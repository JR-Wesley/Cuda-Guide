"""Run Vector Add examples using existing environments and framework JIT."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from command_support import NAMES, ROOT, Record, python_for, target_parser


def main():
    args = target_parser(__doc__).parse_args()
    record = Record("run")
    for name in NAMES if args.target == "all" else (args.target,):
        py = python_for(name)
        if not record.call(f"{name}-gpu", [py, ROOT / "scripts/check.py", name, "--gpu"]):
            continue
        for sample in ("triton", "gluon") if name == "triton" else (name,):
            record.call(sample, [py, ROOT / "examples" / f"{sample}_vector_add.py"])
    return record.finish()


if __name__ == "__main__":
    raise SystemExit(main())

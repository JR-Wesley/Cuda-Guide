"""Run selected small functional cases; timing and scans are opt-in."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from command_support import ROOT, Record, python_for, target_parser


def main():
    parser = target_parser(__doc__, ("triton", "gluon", "cute", "cutile", "cpu"))
    parser.add_argument("--ids", help="comma-separated plan IDs or full case IDs")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--compare-configs", action="store_true")
    args = parser.parse_args()
    if args.ids and args.target in ("all", "cpu"):
        parser.error("--ids requires one DSL target")
    record = Record("learn")
    if args.target in ("all", "cpu"):
        record.call("cpu-tests", [python_for("triton"), "-m", "unittest", "discover", "-s", "tests", "-v"])
        record.call("G3.2-layouts", [python_for("triton"), "-m", "experiments.gluon_layouts", "--output", record.path / "G3.2-layouts"])
        record.call("C4.2-layouts", [python_for("cute"), "-m", "experiments.cute_layouts", "--output", record.path / "C4.2-layouts"])
    selected = ("triton", "gluon", "cute", "cutile") if args.target == "all" else (args.target,)
    for name in selected:
        if name == "cpu":
            continue
        py = python_for(name)
        if not record.call(f"{name}-gpu", [py, ROOT / "scripts/check.py", "triton" if name == "gluon" else name, "--gpu"]):
            continue
        if name == "gluon" and not args.ids:
            record.call("G3.2-gpu-layouts", [py, "-m", "experiments.gluon_layouts", "--output", record.path / "G3.2-gpu-layouts", "--verify-gpu"])
        if name == "cute" and not args.ids and args.target != "all":
            record.call("C4.2-layouts", [py, "-m", "experiments.cute_layouts", "--output", record.path / "C4.2-layouts"])
        cmd = [py, "-m", f"experiments.{name}_suite", "--output", record.path / name]
        if args.ids:
            cmd += ["--ids", args.ids]
        if args.bench:
            cmd.append("--bench")
        if args.compare_configs and name != "gluon":
            cmd.append({"triton": "--scan", "cute": "--tile-comparison", "cutile": "--scan-tiles"}[name])
        record.call(f"{name}-cases", cmd)
    record.call("summary", [sys.executable, ROOT / "scripts/learning_summary.py", record.path])
    return record.finish()


if __name__ == "__main__":
    raise SystemExit(main())

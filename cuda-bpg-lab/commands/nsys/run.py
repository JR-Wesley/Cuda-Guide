"""Capture an Nsight Systems timeline for one experiment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from common import benchmark_args, execute, parser_for
from profiling import benchmark


def main():
    args, extra = benchmark_args(parser_for(__doc__, required=True))

    def profile(session, selected, directory):
        session.command(["nsys", "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
                         "-o", session.path / "timeline", *benchmark(session, directory, args.experiment, extra)], show=True)

    return execute("nsys", args.experiment, profile, tool="nsys")


if __name__ == "__main__":
    raise SystemExit(main())

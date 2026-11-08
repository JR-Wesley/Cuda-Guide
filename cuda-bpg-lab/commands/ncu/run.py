"""Collect basic Nsight Compute counters for one experiment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from common import benchmark_args, execute, parser_for
from profiling import benchmark


def main():
    args, extra = benchmark_args(parser_for(__doc__, required=True))

    def profile(session, selected, directory):
        session.command(["ncu", "--list-sections"])
        session.command(["ncu", "--query-metrics"])
        session.command(["ncu", "--set", "basic", "--launch-count", "3", "-o", session.path / "compute",
                         *benchmark(session, directory, args.experiment, extra)], show=True)

    return execute("ncu", args.experiment, profile, tool="ncu")


if __name__ == "__main__":
    raise SystemExit(main())

"""Run one prebuilt experiment and replace this action's CSV/logs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from common import benchmark_args, execute, parser_for


def main():
    parser = parser_for(__doc__, required=True)
    args, extra = benchmark_args(parser)

    def run(session, selected, directory):
        session.command([directory / "bin" / args.experiment, *extra,
                         "--csv", session.path / "samples.csv"], show=True)

    return execute("run", args.experiment, run)


if __name__ == "__main__":
    raise SystemExit(main())

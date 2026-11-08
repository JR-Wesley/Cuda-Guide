"""Test existing CPU-only targets without building or using a GPU."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from common import execute, parser_for


def main():
    args = parser_for(__doc__).parse_args()
    def check(session, selected, directory):
        session.command(["ctest", "--test-dir", directory, "-L", "host",
                         "--output-on-failure", "--no-tests=error"], show=True)

    return execute("host-test", args.experiment, check, host_only=True)


if __name__ == "__main__":
    raise SystemExit(main())

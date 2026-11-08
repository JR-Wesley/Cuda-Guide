"""Build selected CUDA experiments; no tests or GPU execution."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from build_support import build
from common import execute, parser_for


def main():
    args = parser_for(__doc__).parse_args()
    return execute("build", args.experiment,
                   lambda session, selected, directory: build(session, selected=selected),
                   needs_build=False)


if __name__ == "__main__":
    raise SystemExit(main())

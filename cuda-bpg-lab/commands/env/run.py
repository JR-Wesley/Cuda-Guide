"""Record available GPU and toolchain information."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import argparse
from common import execute


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    return execute("env", work=lambda session, selected, directory: session.environment(),
                   needs_build=False)


if __name__ == "__main__":
    raise SystemExit(main())

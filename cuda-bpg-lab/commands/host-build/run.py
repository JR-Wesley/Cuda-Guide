"""Build CPU-only tests without running them."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build_support import build
from common import execute


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    return execute("host-build", work=lambda session, selected, directory: build(session, host_only=True),
                   needs_build=False)


if __name__ == "__main__":
    raise SystemExit(main())

"""Inspect host tools without installation or GPU kernels."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from command_support import ROOT, Record


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    record = Record("inspect")
    record.call("host", ["bash", ROOT / "scripts/inspect.sh"])
    record.call("tools", [sys.executable, ROOT / "scripts/learning_tools.py"])
    return record.finish()


if __name__ == "__main__":
    raise SystemExit(main())

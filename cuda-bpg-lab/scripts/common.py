"""Shared CLI rules and evidence lifecycle; actions own their work."""
import argparse
import shutil
import sys

from build_support import require_build
from config import EXPERIMENTS
from session import Session


def parser_for(description, required=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("experiment", choices=EXPERIMENTS, nargs=None if required else "?")
    return parser


def benchmark_args(parser):
    args, extra = parser.parse_known_args()
    if "--csv" in extra or any(arg.startswith("--csv=") for arg in extra):
        parser.error("this action writes CSV under its own log directory")
    return args, extra


def execute(action, experiment=None, work=None, *, tool=None, host_only=False, needs_build=True):
    session = Session(action)
    selected = [experiment] if experiment else list(EXPERIMENTS)
    try:

        if tool and not shutil.which(tool):
            (session.path / "status.txt").write_text(f"Not Tested: {tool} unavailable\n")
            return 77
        directory = require_build(selected, host_only) if needs_build else None
        if work:
            work(session, selected, directory)
        status = "ENVIRONMENT_RECORDED (individual tool probes may have failed)" if action == "env" else "PASS: " + action
        (session.path / "status.txt").write_text(status + "\n")
        print(status)
        return 0
    except Exception as error:
        (session.path / "status.txt").write_text("FAILED: " + str(error) + "\n")
        print(str(error), file=sys.stderr)
        return 1

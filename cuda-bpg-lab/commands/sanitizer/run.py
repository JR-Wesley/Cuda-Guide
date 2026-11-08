"""Run all four sanitizer tools on short cases."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from common import execute, parser_for
from config import SMOKE

def check(session, selected, directory):
    binaries = directory / "bin"
    for tool in ("memcheck", "initcheck", "racecheck", "synccheck"):
        for name in selected:
            session.command(["compute-sanitizer", "--tool", tool, "--error-exitcode", "99", binaries / name,
                             *SMOKE.get(name, ["--n", "4099"]), "--warmup", "1", "--repeats", "1", "--iterations", "2"], show=True)
            if name == "host_transfer":
                session.command(["compute-sanitizer", "--tool", tool, "--error-exitcode", "99", binaries / name,
                                 "--n", "4099", "--direction", "d2h", "--warmup", "1", "--repeats", "1", "--iterations", "2"], show=True)
            if name == "gemm":
                session.command(["compute-sanitizer", "--tool", tool, "--error-exitcode", "99", binaries / name,
                                 *SMOKE[name], "--tile", "32", "--block", "1024", "--reg-rows", "2", "--pattern", "cancellation",
                                 "--warmup", "1", "--repeats", "1", "--iterations", "2"], show=True)
            if name == "stream_pipeline":
                session.command(["compute-sanitizer", "--tool", tool, "--error-exitcode", "99", binaries / name,
                                 *SMOKE[name], "--order", "phased", "--warmup", "1", "--repeats", "1", "--iterations", "2"], show=True)
            if name in ("integer_division", "branch_flow"):
                opts = ["--n", "33", "--rounds", "3", "--divisor", "7", "--signed", "0"] if name == "integer_division" else ["--n", "65", "--rounds", "1", "--body", "short"]
                session.command(["compute-sanitizer", "--tool", tool, "--error-exitcode", "99", binaries / name,
                                 *opts, "--warmup", "1", "--repeats", "1", "--iterations", "2"], show=True)
            if name == "global_copy":
                for direction in ("gather", "scatter", "both"):
                    session.command(["compute-sanitizer", "--tool", tool, "--error-exitcode", "99", binaries / name,
                                     "--variant", "scalar", "--direction", direction, "--stride", "3", "--offset", "7",
                                     "--n", "4099", "--warmup", "1", "--repeats", "1", "--iterations", "2"], show=True)

def main():
    args = parser_for(__doc__).parse_args()
    return execute("sanitizer", args.experiment, check, tool="compute-sanitizer")

if __name__ == "__main__":
    raise SystemExit(main())

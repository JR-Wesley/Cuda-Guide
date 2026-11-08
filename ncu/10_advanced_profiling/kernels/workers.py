"""Launch the same prebuilt CUDA program; change WORKERS to 2 for the contrast."""
from pathlib import Path
import subprocess
import sys

WORKERS = 1


def main():
    executable = Path(__file__).resolve().parents[1] / "build" / "lesson10"
    children = []
    failed = False
    try:
        for worker in range(WORKERS):
            child = subprocess.Popen([str(executable)])
            children.append(child)
            print(f"worker={worker} pid={child.pid}", flush=True)
    except OSError as error:
        print(f"Cannot launch {executable}: {error}; compile first.", file=sys.stderr)
        failed = True
    for child in children:
        status = child.wait()
        print(f"pid={child.pid} exit_code={status}", flush=True)
        failed |= status != 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

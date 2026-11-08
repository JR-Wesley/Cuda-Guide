"""Environment variables and fixed logs shared by independent action scripts."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("triton", "cute", "cutile")
PYTHON = "3.12.15"


def target_parser(description, choices=NAMES):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("target", nargs="?", choices=(*choices, "all"), default="all")
    return parser


def python_for(name):
    return ROOT / ".venvs" / ("triton" if name == "gluon" else name) / "bin/python"


class Record:
    def __init__(self, action):
        self.path = ROOT / "commands" / action / "log"
        if self.path.exists():
            shutil.rmtree(self.path)
        self.path.mkdir(parents=True)
        self.commands = []
        self.env = os.environ.copy()
        self.env.update(UV_CACHE_DIR=str(ROOT / ".cache/uv"),
                        UV_PYTHON_INSTALL_DIR=str(ROOT / ".python"),
                        UV_PYTHON_BIN_DIR=str(ROOT / ".python/bin"),
                        UV_LINK_MODE="hardlink", PYTHONUNBUFFERED="1",
                        PYTHONDONTWRITEBYTECODE="1",
                        TRITON_CACHE_DIR=str(ROOT / "build/triton"),
                        XDG_CACHE_HOME=str(ROOT / ".cache"),
                        CUTE_DSL_CACHE_DIR=str(ROOT / "build/cute"),
                        CUDA_TILE_CACHE_DIR=str(ROOT / "build/cutile"),
                        CUPY_CACHE_DIR=str(ROOT / "build/cupy"),
                        CUDA_CACHE_PATH=str(ROOT / "build/cuda"),
                        TMPDIR=str(ROOT / ".cache/tmp"),
                        CUDA_HOME=self.env.get("CUDA_HOME", "/opt/cuda"),
                        CUDA_PATH=self.env.get("CUDA_PATH", "/opt/cuda"))

    def call(self, label, command):
        command = list(map(str, command))
        (ROOT / ".cache/tmp").mkdir(parents=True, exist_ok=True)
        print(shlex.join(command), flush=True)
        with (self.path / f"{label}.log").open("w") as log:
            log.write(shlex.join(command) + "\n")
            log.flush()
            try:
                proc = subprocess.run(command, cwd=ROOT, env=self.env,
                                      stdout=log, stderr=subprocess.STDOUT)
                rc = proc.returncode
            except OSError as exc:
                log.write(str(exc) + "\n")
                if str(ROOT / ".venvs") in command[0]:
                    message = "Environment missing/unusable; run python3 commands/setup/run.py for this target first."
                    log.write(message + "\n")
                    print(message, flush=True)
                rc = 127
            log.write(f"exit_code={rc}\n")
        self.commands.append(dict(label=label, command=command, cwd=str(ROOT), exit_code=rc))
        (self.path / "manifest.json").write_text(json.dumps(dict(commands=self.commands), indent=2) + "\n")
        print(f"{label}: exit {rc}; {self.path / (label + '.log')}", flush=True)
        return rc == 0

    def finish(self):
        failed = any(command["exit_code"] for command in self.commands)
        (self.path / "status.txt").write_text("FAIL\n" if failed else "PASS\n")
        return int(failed)

"""Fixed action-local logs; no source or binary snapshots."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from config import ROOT

class Session:
    def __init__(self, action):
        self.path = ROOT / "commands" / action / "log"
        # Only this action's replaceable output is cleared, never historical results.
        if self.path.exists():
            shutil.rmtree(self.path)
        self.path.mkdir(parents=True)
        self.counter = 0
        self.env = os.environ.copy()
        self.env["BPG_MEASUREMENT_CONTEXT"] = action if action in ("sanitizer", "nsys", "ncu") else "standalone"
        (self.path / "invocation.json").write_text(json.dumps({
            "argv": sys.argv, "cwd": str(Path.cwd()),
            "environment": {k: os.environ.get(k) for k in (
                "CUDA_ARCH", "CUDA_VISIBLE_DEVICES", "CUDA_FORCE_PTX_JIT", "CUDA_DISABLE_PTX_JIT",
                "CUDA_LAUNCH_BLOCKING", "BPG_CUDA_HOST_COMPILER")}}, indent=2) + "\n")

        print(f"Evidence: {self.path}", flush=True)

    def command(self, args, required=True, show=False, expected=0):
        self.counter += 1
        log = self.path / f"{self.counter:03d}_{Path(str(args[0])).name}.log"
        args = [str(x) for x in args]
        try:
            proc = subprocess.run(args, cwd=ROOT, env=self.env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True)
            code, output = proc.returncode, proc.stdout
        except OSError as error:
            code, output = 127, str(error) + "\n"
        log.write_text(output)
        with (self.path / "commands.jsonl").open("a") as stream:
            stream.write(json.dumps({"argv": args, "shell_display": shlex.join(args),
                                     "cwd": str(ROOT), "exit_code": code, "expected_exit_code": expected,
                                     "log": log.name}) + "\n")
        print(f"[{code}] {shlex.join(args)} -> {log.name}", flush=True)
        if show or (required and code != expected):
            print(output, end="", flush=True)
        if required and code != expected:
            raise RuntimeError(f"command failed with exit {code}; see {log}")
        return code, output

    def environment(self):
        for command in (
            ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total,pstate,temperature.gpu,power.draw", "--format=csv"],
            ["nvcc", "--version"], ["nvcc", "--list-gpu-code"], ["cmake", "--version"],
            ["c++", "--version"], ["nsys", "--version"], ["ncu", "--version"],
            ["compute-sanitizer", "--version"], ["cuobjdump", "--version"]):
            self.command(command, required=False)

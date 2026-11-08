"""E0.4 tool availability only; never changes permissions or runs a profiler."""
import json
from pathlib import Path
import shutil
import subprocess


def main():
    tools = {}
    for name in ('nvcc', 'ncu', 'nsys', 'cuobjdump', 'nvdisasm', 'compute-sanitizer'):
        path = shutil.which(name)
        if not path and (Path('/opt/cuda/bin') / name).is_file():
            path = str(Path('/opt/cuda/bin') / name)
        if not path:
            tools[name] = {'status': 'unavailable'}
            continue
        try:
            result = subprocess.run([path, '--version'], capture_output=True, text=True, timeout=15)
            tools[name] = dict(path=path, command=[path, '--version'], exit_code=result.returncode,
                               stdout=result.stdout, stderr=result.stderr)
        except (OSError, subprocess.TimeoutExpired) as error:
            tools[name] = dict(path=path, error=str(error))
    print(json.dumps(dict(tools=tools,
        profile='not measured: functional-first run; version check is not kernel profiling',
        sass='not measured by this command; use retained cubin and commands/static/run.py'), indent=2))


if __name__ == '__main__':
    main()

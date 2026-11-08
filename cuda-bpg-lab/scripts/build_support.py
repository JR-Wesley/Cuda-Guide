"""Explicit CMake build and checks for reusable build outputs."""
import os
import re
from config import EXPERIMENTS, ROOT


def build_directory(host_only=False):
    if host_only:
        return ROOT / "build" / "host"
    arch = os.environ.get("CUDA_ARCH", "")
    if not re.fullmatch(r"sm_[78][0-9]", arch):
        raise RuntimeError("set CUDA_ARCH=sm_XX explicitly within SM70-SM89")
    return ROOT / "build" / arch


def require_build(selected, host_only=False):
    directory = build_directory(host_only)
    targets = [directory / "host_tests"] if host_only else [directory / "bin" / name for name in selected]
    for target in targets:
        if not target.is_file():
            action = "host-build" if host_only else "build"
            raise RuntimeError(f"Missing {target}; run python3 commands/{action}/run.py first")
    return directory


def build(session, host_only=False, selected=None):
    selected = selected or list(EXPERIMENTS)
    directory = build_directory(host_only)
    # A failed explicit build must not leave an older selected executable runnable.
    outputs = [directory / "host_tests"] if host_only else [directory / "bin" / name for name in selected]
    for output in outputs:
        output.unlink(missing_ok=True)
    if host_only:
        config = ["-DBPG_ENABLE_CUDA=OFF"]
    else:
        arch = directory.name
        _, codes = session.command(["nvcc", "--list-gpu-code"])
        if arch not in codes.split():
            raise RuntimeError(f"{arch} unsupported by this Toolkit")
        labs = sorted({EXPERIMENTS[name] for name in selected})
        config = [f"-DCMAKE_CUDA_ARCHITECTURES={arch[3:]}", "-DBPG_ENABLE_CUDA=ON",
                  "-DBPG_LABS=" + ";".join(labs)]
        if os.environ.get("BPG_CUDA_HOST_COMPILER"):
            config.append("-DCMAKE_CUDA_HOST_COMPILER=" + os.environ["BPG_CUDA_HOST_COMPILER"])
    session.command(["cmake", "-S", ROOT, "-B", directory, "-DCMAKE_BUILD_TYPE=Release", *config])
    targets = ["host_tests"] if host_only else ["host_tests", *selected]
    try:
        session.command(["cmake", "--build", directory, "--parallel", "2", "--target", *targets])
    except Exception:
        for output in outputs:
            output.unlink(missing_ok=True)
        raise
    return directory

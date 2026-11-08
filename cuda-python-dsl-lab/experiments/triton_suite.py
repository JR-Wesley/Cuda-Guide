"""Small Triton learning-plan cases for the shared benchmarks.runner contract.

No device is initialized while constructing the catalog. --scan expands only the
configuration catalog; execution/timing remains the shared runner's responsibility.
"""

import argparse
from functools import partial
from importlib import import_module
import json
import sys

import numpy as np


_BACKENDS = {
    "copy": "vector", "scale": "vector", "add": "vector", "axpy": "vector",
    "fusion": "vector", "transpose": "transpose", "sum": "reduction",
    "softmax": "reduction", "gemm": "gemm",
}


def _validate_inputs(op, inputs):
    expected_count = 2 if op in {"add", "axpy", "fusion", "gemm"} else 1
    if len(inputs) != expected_count:
        raise ValueError(f"{op} expects {expected_count} input arrays")
    dtype = np.dtype("float16" if op == "gemm" else "float32")
    ndim = 1 if _BACKENDS[op] == "vector" else 2
    for array in inputs:
        if not isinstance(array, np.ndarray) or array.dtype != dtype:
            raise ValueError(f"{op} requires NumPy {dtype} arrays")
        if array.ndim != ndim or not array.flags.c_contiguous or any(n <= 0 for n in array.shape):
            raise ValueError(f"{op} requires nonempty contiguous {ndim}D arrays")
    if op == "gemm":
        if inputs[0].shape[1] != inputs[1].shape[0]:
            raise ValueError("GEMM expects A[M,K] and B[K,N]")
    elif len(inputs) == 2 and inputs[0].shape != inputs[1].shape:
        raise ValueError(f"{op} requires matching input shapes")
    if op == "softmax" and not np.isfinite(inputs[0]).all():
        raise ValueError("Softmax supports only finite inputs")


def _artifacts(compiled):
    if not compiled:
        raise RuntimeError("Launch the workload before requesting compiler artifacts")
    artifacts = {}
    for label, kernel in compiled.items():
        # These are the actual compiler exports, not disassembly reconstructed from text.
        for extension in ("ptx", "cubin"):
            value = kernel.asm.get(extension)
            expected = str if extension == "ptx" else bytes
            if not isinstance(value, expected) or not value:
                raise RuntimeError(f"Compiled kernel {label} has no exported {extension}")
            artifacts[f"{label}.{extension}"] = value
        for extension in ("ttir", "ttgir", "llir"):
            value = kernel.asm.get(extension)
            if isinstance(value, str):
                artifacts[f"{label}.{extension}"] = value
        metadata = kernel.metadata
        artifacts[f"{label}.metadata.json"] = json.dumps({
            "name": kernel.name,
            "target": str(getattr(metadata, "target", None)),
            "num_warps": getattr(metadata, "num_warps", None),
            "num_stages": getattr(metadata, "num_stages", None),
            "shared_bytes": getattr(metadata, "shared", None),
            "registers_per_thread": getattr(kernel, "n_regs", None),
            "spills": getattr(kernel, "n_spills", None),
        }, indent=2) + "\n"
    return artifacts


def _prepare(op, inputs, config):
    _validate_inputs(op, inputs)
    import torch
    from benchmarks.runner import Workload

    backend = import_module(f"experiments.{_BACKENDS[op]}.triton_ops")
    tensors = tuple(torch.from_numpy(array).to(device="cuda") for array in inputs)
    if op == "gemm":
        output_shape = (inputs[0].shape[0], inputs[1].shape[1])
    elif op == "transpose":
        output_shape = inputs[0].shape[::-1]
    elif op == "sum":
        output_shape = (inputs[0].shape[0],)
    else:
        output_shape = inputs[0].shape
    output = torch.empty(output_shape, dtype=tensors[0].dtype, device=tensors[0].device)
    compiled = {}
    launch, metadata = backend.make_launch(tensors, output, {**config, "op": op}, compiled)
    metadata.update({
        "backend": "triton",
        "input_shapes": [tuple(array.shape) for array in inputs],
        "output_shape": output_shape,
        "output_contiguous": True,
        "stream": "current PyTorch CUDA stream; no private stream or host synchronization",
        "timing_scope": "entire launch closure; two kernels for unfused fusion, otherwise one",
        "enable_fp_fusion": False,
    })
    return Workload(launch=launch, output=output, framework="torch",
                    artifacts=partial(_artifacts, compiled), metadata=metadata)


def build_experiments(scan=False):
    """Return ordered learning IDs; repeated IDs denote distinct shape/config cases."""
    from benchmarks.runner import Experiment

    cases = []
    vector = dict(block=256, num_warps=4)
    gemm = dict(block_m=16, block_n=32, block_k=16, num_warps=4, num_stages=2)

    def add(identifier, op, shape, **config):
        cases.append(Experiment(id=identifier, op=op, shape=shape, config=config,
                                prepare=partial(_prepare, op)))

    # E cases exercise the adapter/artifact path, not environment or profiler collection.
    add("E0.1", "add", (1,), **vector)
    add("E0.2", "add", (255,), **vector)
    add("E0.3", "add", (4099,), **vector)
    add("E0.4", "copy", (256,), **vector)
    for op in ("copy", "add", "axpy"):
        for n in (1, 255, 256, 4099):
            add("T1.1", op, (n,), **vector)

    # One-factor-at-a-time catalog, not an autotuner or a performance claim.
    vector_configs = [vector]
    if scan:
        vector_configs += [dict(block=b, num_warps=4) for b in (128, 512, 1024)]
        vector_configs += [dict(block=256, num_warps=8)]
    for config in vector_configs:
        add("T1.2", "add", (4099,), **config)

    for op in ("sum", "softmax"):
        for shape in ((1, 127), (3, 129), (2, 4097)):
            add("T1.3", op, shape, num_warps=4)
    for shape in ((32, 32, 32), (37, 53, 29)):
        add("T1.4", "gemm", shape, **gemm)

    # Minimal row-length and M-tile comparison with all other variables fixed.
    for cols in (127, 129):
        add("T1.5", "softmax", (1, cols), num_warps=4)
    for bm in (16, 32):
        add("T1.5", "gemm", (37, 53, 29), **{**gemm, "block_m": bm})
    if scan:
        add("T1.5", "softmax", (1, 4097), num_warps=4)

    for shape in ((32, 32), (37, 53)):
        add("T2.1", "transpose", shape, variant="naive", **vector)
        add("T2.1", "transpose", shape, variant="tiled", tile_m=32, tile_n=32, num_warps=4)

    gemm_configs = [gemm]
    if scan:
        gemm_configs += [{**gemm, "num_warps": 8}]
        gemm_configs += [{**gemm, "num_stages": stages} for stages in (1, 3)]
    for config in gemm_configs:
        add("T2.2", "gemm", (37, 53, 29), **config)

    for variant in ("unfused", "fused"):
        add("T2.3", "fusion", (4099,), variant=variant, **vector)

    # Reproducible evidence pairs; no hardware explanation is inferred by this catalog.
    add("T2.4", "transpose", (37, 53), variant="naive", **vector)
    add("T2.4", "transpose", (37, 53), variant="tiled", tile_m=32, tile_n=32, num_warps=4)
    for variant in ("unfused", "fused"):
        add("T2.4", "fusion", (4099,), variant=variant, **vector)
    return cases


def main():
    from benchmarks.runner import run_cli

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--scan", action="store_true", help="include small opt-in configuration comparisons")
    options, remaining = parser.parse_known_args()
    original = sys.argv[:]
    try:
        sys.argv[:] = [sys.argv[0], *remaining]
        run_cli(build_experiments(scan=options.scan))
    finally:
        sys.argv[:] = original


if __name__ == "__main__":
    main()

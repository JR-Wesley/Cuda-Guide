"""Small cuTile U5.1-U5.4 adapters for benchmarks.CONTRACT; no private references.

Run from the project root with .venvs/cutile/bin/python -m experiments.cutile_suite.
Imports, --help and --list-ids do not allocate CUDA arrays or launch kernels.
"""

import argparse
from functools import partial
from importlib.metadata import version
import io
import json
import platform
import sys


GEMM_TILES = ((32, 32, 16), (32, 64, 16))


def experiment_specs(scan_tiles=False):
    """Return (id, op, shape, config) in plan order, without GPU/runner imports."""
    specs = []
    for op in ("copy", "add"):
        for n in (1, 255, 256, 4099):
            specs.append((f"U5.1.cutile.{op}.n{n}.b256", op, (n,), {"block": 256}))
    for rows, cols in ((32, 32), (37, 53)):
        specs.append((f"U5.2.cutile.transpose.{rows}x{cols}.t16x32", "transpose",
                      (rows, cols), {"tile": (16, 32)}))
    for rows, cols in ((1, 127), (3, 129), (2, 4097)):
        width = 1 << (cols - 1).bit_length()
        specs.append((f"U5.2.cutile.sum.{rows}x{cols}.b{width}", "sum",
                      (rows, cols), {"block": width}))
    tiles = GEMM_TILES if scan_tiles else GEMM_TILES[:1]
    for plan_step, (bm, bn, bk) in enumerate(tiles, start=3):
        for m, n, k in ((32, 32, 32), (37, 53, 29)):
            specs.append((f"U5.{plan_step}.cutile.gemm.{m}x{n}x{k}.t{bm}x{bn}x{bk}", "gemm",
                          (m, n, k), {"tile": (bm, bn, bk)}))
    return specs


def api_provenance():
    """Record installed versions without reading or fingerprinting package sources."""
    return {"python": platform.python_version(),
            "packages": {name: version(name)
                         for name in ("cuda-tile", "cupy-cuda13x", "numpy")}}


def _artifact_export(kernel, args):
    """Export a separate, architecture-independent specialization, not JIT SASS."""
    from cuda.tile.compilation import CallingConvention, KernelSignature, export_kernel

    report = {
        "api": "cuda.tile.compilation.export_kernel",
        "output_format": "tileir_bytecode",
        "bytecode_version": "13.3",
        "calling_convention": "cutile_python_v1",
        "scope": "separate frontend export from the same kernel and launch arguments",
        "jit_binary_identity": "not verified",
        "ptx_cubin_sass_registers_shared_memory": "not measured",
        "toolkit_tileiras_driver_device": "not queried by this export",
        "provenance": api_provenance(),
    }
    artifacts = {}
    try:
        signature = KernelSignature.from_kernel_args(
            kernel, args, CallingConvention.cutile_python_v1())
        buffer = io.BytesIO()
        export_kernel(kernel, [signature], buffer, output_format="tileir_bytecode",
                      bytecode_version="13.3")
        bytecode = buffer.getvalue()
        if not bytecode:
            raise RuntimeError("export_kernel returned empty TileIR bytecode")
        artifacts["kernel.tileirbc"] = bytecode
        report.update(status="exported", signature=repr(signature))
    except Exception as error:
        # Artifact support must not hide a launch/correctness failure: this catches
        # only the optional export, and preserves its failure in the run evidence.
        report.update(status="not measured", error=f"{type(error).__name__}: {error}")
    artifacts["compiler-artifacts.json"] = json.dumps(report, indent=2) + "\n"
    return artifacts


def prepare(op, inputs, config):
    """Upload only the common runner's arrays; allocate outside the launch closure."""
    import cupy as cp
    import cuda.tile as ct
    import numpy as np
    from benchmarks.runner import Workload
    from experiments.vector.cutile_ops import add_kernel, copy_kernel
    from experiments.transpose.cutile_ops import transpose_kernel
    from experiments.reduction.cutile_ops import row_sum_kernel
    from experiments.gemm.cutile_ops import gemm_kernel

    expected_count = 2 if op in ("add", "gemm") else 1
    dtype = np.dtype("float16" if op == "gemm" else "float32")
    ndim = 1 if op in ("copy", "add") else 2
    if len(inputs) != expected_count:
        raise ValueError(f"{op} requires {expected_count} shared input arrays")
    if any(x.dtype != dtype or x.ndim != ndim or not x.flags.c_contiguous
           or any(d <= 0 for d in x.shape) for x in inputs):
        raise ValueError(f"{op} requires nonempty, contiguous {ndim}D {dtype} inputs")
    if op == "add" and inputs[0].shape != inputs[1].shape:
        raise ValueError("add input shapes must match")
    if op == "gemm" and inputs[0].shape[1] != inputs[1].shape[0]:
        raise ValueError("gemm requires A[M,K] and B[K,N]")
    arrays = tuple(cp.asarray(x) for x in inputs)
    a = arrays[0]
    metadata = {
        "backend": "cutile", "provenance": api_provenance(),
        "mapping": "compiler-selected; tile extents are not CUDA thread counts",
        "stages": "compiler-managed; no explicit async pipeline",
        "bounds": "ZERO loads and bounds-checked stores; positive contiguous shapes only",
        "input_dtype": str(dtype), "output_dtype": str(dtype),
        "artifacts": "post-launch separate TileIR 13.3 export; see compiler-artifacts.json",
        "resources": "not measured",
    }
    if op in ("copy", "add"):
        block = config["block"]
        out = cp.empty(a.shape, dtype=dtype, order="C")
        kernel = copy_kernel if op == "copy" else add_kernel
        args = (*arrays, out, block)
        grid = ((a.size + block - 1) // block, 1, 1)
        metadata["logical_tile"] = (block,)
    elif op == "transpose":
        rows, cols = config["tile"]
        out = cp.empty((a.shape[1], a.shape[0]), dtype=dtype, order="C")
        kernel, args = transpose_kernel, (a, out, rows, cols)
        grid = ((a.shape[0] + rows - 1) // rows,
                (a.shape[1] + cols - 1) // cols, 1)
        metadata.update(logical_tile=(rows, cols), output_layout="materialized C-contiguous")
    elif op == "sum":
        block = config["block"]
        if block < a.shape[1] or block & (block - 1):
            raise ValueError("row sum block must be a covering power of two")
        out = cp.empty((a.shape[0],), dtype=dtype, order="C")
        kernel, args = row_sum_kernel, (a, out, block)
        grid = (a.shape[0], 1, 1)
        metadata.update(logical_tile=(1, block), reduction_axis=1,
                        accumulator_dtype="float32")
    elif op == "gemm":
        bm, bn, bk = config["tile"]
        if (bm, bn, bk) not in GEMM_TILES:
            raise ValueError(f"supported GEMM tiles: {GEMM_TILES}")
        b = arrays[1]
        out = cp.empty((a.shape[0], b.shape[1]), dtype=dtype, order="C")
        kernel, args = gemm_kernel, (a, b, out, bm, bn, bk)
        grid = ((a.shape[0] + bm - 1) // bm, (b.shape[1] + bn - 1) // bn, 1)
        metadata.update(logical_tile=(bm, bn, bk), accumulator_dtype="float32",
                        primitive="ct.mma; final ct.astype(float16)",
                        k_tiles=(a.shape[1] + bk - 1) // bk,
                        reuse="A across BN columns, B across BM rows within each K tile")
    else:
        raise ValueError(f"unsupported cuTile operation: {op}")
    metadata["grid"] = grid
    launched = False

    def launch():
        nonlocal launched
        ct.launch(cp.cuda.get_current_stream(), grid, kernel, args)
        launched = True

    def artifacts():
        if not launched:
            return {"compiler-artifacts.json": json.dumps(
                {"status": "not measured", "reason": "kernel has not been launched"}) + "\n"}
        return _artifact_export(kernel, args)

    return Workload(launch=launch, output=out, framework="cupy",
                    artifacts=artifacts, metadata=metadata)


def build_experiments(scan_tiles=False):
    from benchmarks.runner import Experiment

    return [Experiment(id=id_, op=op, shape=shape, config=config,
                       prepare=partial(prepare, op))
            for id_, op, shape, config in experiment_specs(scan_tiles)]


def main():
    parser = argparse.ArgumentParser(
        description="cuTile U5.1-U5.4; remaining arguments go to benchmarks.runner",
        epilog="Runner: --output DIR [--ids ID,ID] [--bench --warmup 10 --repeat 50 --batches 5]",
        allow_abbrev=False)
    parser.add_argument("--scan-tiles", action="store_true",
                        help="include the second small GEMM tile; no autotuning or large cases")
    parser.add_argument("--list-ids", action="store_true", help="list selected IDs without CUDA")
    options, remaining = parser.parse_known_args()
    if options.list_ids:
        print("\n".join(spec[0] for spec in experiment_specs(options.scan_tiles)))
        return
    from benchmarks.runner import run_cli

    sys.argv[1:] = remaining
    run_cli(build_experiments(options.scan_tiles))


if __name__ == "__main__":
    main()

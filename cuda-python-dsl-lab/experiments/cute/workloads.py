"""Common-runner adapters. All tensor allocation and compilation is in prepare."""

import importlib.metadata
import json
import tempfile


def prepare(inputs, config):
    # Lazy imports keep suite discovery/--help free of CUDA context creation.
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    import torch
    from cutlass.cute.runtime import from_dlpack
    from benchmarks.runner import Workload
    from experiments.cute.kernels import SharedCopy, Transpose, Vector, WarpGemm

    version = importlib.metadata.version("nvidia-cutlass-dsl")
    if version != "4.8.1":
        raise RuntimeError(f"This adapter targets installed CuTe 4.8.1, found {version}")
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    if capability != (8, 9):
        raise RuntimeError(f"This suite explicitly targets SM89, found SM{capability}")
    kind = config["kind"]
    expected_inputs = 2 if kind in ("add", "gemm") else 1
    if len(inputs) != expected_inputs:
        raise ValueError(f"{kind} needs {expected_inputs} CPU inputs")
    expected_dtype = "float16" if kind == "gemm" else "float32"
    for array in inputs:
        if str(array.dtype) != expected_dtype or not array.flags.c_contiguous:
            raise ValueError(f"Expected contiguous {expected_dtype} arrays")
        if any(d <= 0 for d in array.shape):
            raise ValueError("Empty tensors are outside this small-suite contract")
    owners = tuple(torch.from_numpy(array).to(device=f"cuda:{device}") for array in inputs)
    a = owners[0]
    b = owners[1] if len(owners) == 2 else a
    if kind == "gemm":
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise ValueError("GEMM expects A[M,K], B[K,N]")
        shape = (a.shape[0], b.shape[1])
        kernel = WarpGemm(tuple(config["tile"]), config.get("stages", 1))
    elif kind == "transpose":
        if a.ndim != 2:
            raise ValueError("Transpose expects a matrix")
        shape = (a.shape[1], a.shape[0])
        kernel = Transpose(shared=config["shared"])
    elif kind == "shared_copy":
        if a.ndim != 1:
            raise ValueError("Shared copy expects a vector")
        shape = a.shape
        kernel = SharedCopy(config["values"])
    elif kind in ("copy", "add"):
        if a.ndim != 1 or a.shape != b.shape:
            raise ValueError("Vector operations expect equal-length vectors")
        shape = a.shape
        kernel = Vector(add=kind == "add")
    else:
        raise ValueError(f"Unknown CuTe kernel kind: {kind}")
    output = torch.empty(shape, device=a.device, dtype=a.dtype)
    # Fresh torch allocations are aligned; row strides are NOT assumed aligned.
    views = tuple(from_dlpack(t, assumed_align=16) for t in (a, b, output))
    stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    dump = tempfile.TemporaryDirectory(prefix="cute-c4-compiler-")
    try:
        compiled = cute.compile(
            kernel, *views, stream,
            options=f"--gpu-arch=sm_89 --keep-ptx --keep-cubin --dump-dir={dump.name}",
        )
    except BaseException:
        dump.cleanup()
        raise

    def launch(_owners=owners, _output=output):
        # No tensor allocation, DLPack conversion, reference work, or compilation.
        # Use the runner's current torch stream, never implicit CUstream(0).
        current = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        compiled(*views, current)

    def artifacts(_dump=dump):
        exported = {}
        status = {"compiler": version, "target": "sm_89"}
        ptx = compiled.__ptx__
        cubin = compiled.__cubin__
        mlir = compiled.__mlir__
        # 4.8.1 can leave a missing artifact's path in a string field. Do not
        # mislabel that filename (or a hand-written placeholder) as compiler IR.
        if isinstance(ptx, str) and ".version" in ptx and ".target" in ptx:
            exported["kernel.ptx"] = ptx
            status["ptx"] = "exported"
            if kind == "gemm":
                status["mma_sync_in_ptx"] = "mma.sync.aligned" in ptx
        else:
            status["ptx"] = "unavailable from compiled.__ptx__"
        if isinstance(cubin, bytes) and cubin.startswith(b"\x7fELF"):
            exported["kernel.cubin"] = cubin
            status["cubin"] = "exported"
        else:
            status["cubin"] = "unavailable from compiled.__cubin__"
        if isinstance(mlir, str) and "module" in mlir and "{" in mlir:
            exported["kernel.mlir"] = mlir
            status["mlir"] = "exported"
        else:
            status["mlir"] = "not retained (PTX/cubin requested)"
        exported["compiler-artifacts.json"] = json.dumps(status, indent=2) + "\n"
        return exported

    metadata = {
        "backend": "cute", "cutlass_dsl": version, "target": "sm_89",
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(capability), "torch_cuda": torch.version.cuda,
        "stream": "torch.cuda.current_stream(device) passed explicitly at every launch",
        "timing": "Runner events must use that same stream; prepare excludes allocation/compile",
        "precision": "FP16 inputs/output, FP32 MMA accumulator" if kind == "gemm" else "FP32",
        "mapping": dict(config),
        "pipeline": "synchronous, one stage; no cp.async/TMA/WGMMA",
        "validation": "runtime validation belongs to results.json",
        "resource_measurements": "unmeasured; inspect exported cubin separately",
    }
    return Workload(launch=launch, output=output, framework="torch",
                    artifacts=artifacts, metadata=metadata)

"""FP32 vector kernels, including a two-rounding fused/unfused chain."""

import torch
import triton
import triton.language as tl


@triton.jit
def vector_kernel(A, B, Out, N: tl.constexpr, OP: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    a = tl.load(A + offsets, mask=mask, other=0)
    if OP == "copy":
        value = a
    elif OP == "scale":
        value = a * 0.5
    elif OP == "add":
        value = a + tl.load(B + offsets, mask=mask, other=0)
    else:
        # enable_fp_fusion=False at launch preserves the FP32 rounding boundary.
        scaled = a * 0.5
        value = scaled + tl.load(B + offsets, mask=mask, other=0)
    tl.store(Out + offsets, value, mask=mask)


def make_launch(inputs, output, config, compiled):
    """Allocate scratch once; capture the actual compiled kernels on each launch."""
    op = config["op"]
    variant = config.get("variant", "fused")
    if op not in {"copy", "scale", "add", "axpy", "fusion"}:
        raise ValueError(f"Unsupported vector op: {op}")
    if variant not in {"fused", "unfused"} or (variant == "unfused" and op != "fusion"):
        raise ValueError(f"Unsupported vector variant: {op}/{variant}")
    block = config.get("block", 256)
    warps = config.get("num_warps", 4)
    if block not in {128, 256, 512, 1024} or warps not in {4, 8}:
        raise ValueError("Vector block must be 128/256/512/1024; num_warps must be 4/8")
    a = inputs[0]
    b = inputs[1] if len(inputs) == 2 else a
    n = a.numel()
    grid = (triton.cdiv(n, block),)
    options = dict(BLOCK=block, num_warps=warps, num_stages=3, enable_fp_fusion=False)
    if variant == "unfused":
        scratch = torch.empty_like(output)

        def launch():
            compiled["scale"] = vector_kernel[grid](a, a, scratch, n, OP="scale", **options)
            compiled["add"] = vector_kernel[grid](scratch, b, output, n, OP="add", **options)

    else:

        def launch():
            compiled["vector"] = vector_kernel[grid](a, b, output, n, OP=op, **options)

    return launch, {
        "mapping": "one program per contiguous logical block; masked tail",
        "grid": grid,
        "block": block,
        "num_warps": warps,
        "threads_per_program": 32 * warps,
        "num_stages": 3,
        "precision": "FP32; multiply and add round separately; enable_fp_fusion=False",
        "launches_per_call": 2 if variant == "unfused" else 1,
        "scratch_elements": n if variant == "unfused" else 0,
    }

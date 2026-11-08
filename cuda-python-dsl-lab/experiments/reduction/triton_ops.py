"""One-program-per-row FP32 reductions with neutral masked padding."""

import triton
import triton.language as tl


@triton.jit
def row_sum(A, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    values = tl.load(A + row * N + cols, mask=cols < N, other=0)
    total = tl.sum(values, axis=0)
    tl.store(Out + row, total)


@triton.jit
def row_softmax(A, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    values = tl.load(A + row * N + cols, mask=cols < N, other=-float("inf"))
    shifted = values - tl.max(values, axis=0)
    numerator = tl.exp(shifted)
    probabilities = numerator / tl.sum(numerator, axis=0)
    tl.store(Out + row * N + cols, probabilities, mask=cols < N)


def make_launch(inputs, output, config, compiled):
    a = inputs[0]
    rows, cols = a.shape
    op = config["op"]
    if op not in {"sum", "softmax"}:
        raise ValueError(f"Unsupported reduction op: {op}")
    block = triton.next_power_of_2(cols)
    if block > 8192:
        raise ValueError("This learning kernel supports row lengths up to 8192")
    warps = config.get("num_warps", 4)
    if warps not in {4, 8}:
        raise ValueError("Reduction num_warps must be 4/8")
    kernel = row_sum if op == "sum" else row_softmax
    grid = (rows,)

    def launch():
        compiled[op] = kernel[grid](a, output, cols, BLOCK=block, num_warps=warps,
                                   num_stages=3, enable_fp_fusion=False)

    return launch, {
        "mapping": "one program per row; power-of-two padded logical vector; compiler-selected reduction communication",
        "grid": grid,
        "block": block,
        "num_warps": warps,
        "threads_per_program": 32 * warps,
        "num_stages": 3,
        "precision": "FP32 input/reduction/output; softmax uses approximate tl.exp",
        "padding": "0" if op == "sum" else "-inf before max; exp(-inf)=0",
        "launches_per_call": 1,
    }

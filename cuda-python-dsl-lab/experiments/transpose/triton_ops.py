"""Materializing naive and tiled transposes of contiguous FP32 matrices."""

import triton
import triton.language as tl


@triton.jit
def naive_transpose(A, Out, M: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offsets // N
    col = offsets % N
    values = tl.load(A + offsets, mask=offsets < M * N, other=0)
    tl.store(Out + col * M + row, values, mask=offsets < M * N)


@triton.jit
def tiled_transpose(A, Out, M: tl.constexpr, N: tl.constexpr,
                    TILE_M: tl.constexpr, TILE_N: tl.constexpr):
    rows = tl.program_id(0) * TILE_M + tl.arange(0, TILE_M)
    cols = tl.program_id(1) * TILE_N + tl.arange(0, TILE_N)
    values = tl.load(A + rows[:, None] * N + cols[None, :],
                     mask=(rows[:, None] < M) & (cols[None, :] < N), other=0)
    tl.store(Out + cols[:, None] * M + rows[None, :], tl.trans(values),
             mask=(cols[:, None] < N) & (rows[None, :] < M))


def make_launch(inputs, output, config, compiled):
    a = inputs[0]
    m, n = a.shape
    variant = config.get("variant", "naive")
    warps = config.get("num_warps", 4)
    if warps not in {4, 8}:
        raise ValueError("Transpose num_warps must be 4/8")
    options = dict(num_warps=warps, num_stages=3, enable_fp_fusion=False)
    metadata = {"num_warps": warps, "num_stages": 3, "precision": "FP32 bit-preserving movement",
                "threads_per_program": 32 * warps, "launches_per_call": 1}
    if variant == "naive":
        block = config.get("block", 256)
        if block not in {128, 256, 512, 1024}:
            raise ValueError("Transpose block must be 128/256/512/1024")
        grid = (triton.cdiv(m * n, block),)

        def launch():
            compiled["naive_transpose"] = naive_transpose[grid](a, output, m, n, BLOCK=block, **options)

        metadata.update(mapping="flattened input traversal; strided output stores", block=block)
    elif variant == "tiled":
        tile_m = config.get("tile_m", 32)
        tile_n = config.get("tile_n", 32)
        if tile_m not in {16, 32} or tile_n not in {16, 32}:
            raise ValueError("Transpose tile dimensions must be 16/32")
        grid = (triton.cdiv(m, tile_m), triton.cdiv(n, tile_n))

        def launch():
            compiled["tiled_transpose"] = tiled_transpose[grid](
                a, output, m, n, TILE_M=tile_m, TILE_N=tile_n, **options)

        metadata.update(mapping="one logical 2D tile per program; tl.trans before output store; compiler-chosen layout",
                        tile_m=tile_m, tile_n=tile_n)
    else:
        raise ValueError(f"Unsupported transpose variant: {variant}")
    metadata["grid"] = grid
    return launch, metadata

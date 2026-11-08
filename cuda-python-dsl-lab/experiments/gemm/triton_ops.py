"""Masked FP16 x FP16 tiled GEMM with FP32 accumulation and FP16 output."""

import triton
import triton.language as tl


@triton.jit
def gemm_kernel(A, B, Out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    accumulator = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)
    for k_tile in range(tl.cdiv(K, BLOCK_K)):
        ks = k_tile * BLOCK_K + k_offsets
        a = tl.load(A + rows[:, None] * K + ks[None, :],
                    mask=(rows[:, None] < M) & (ks[None, :] < K), other=0)
        b = tl.load(B + ks[:, None] * N + cols[None, :],
                    mask=(ks[:, None] < K) & (cols[None, :] < N), other=0)
        accumulator = tl.dot(a, b, accumulator)
    tl.store(Out + rows[:, None] * N + cols[None, :], accumulator.to(tl.float16),
             mask=(rows[:, None] < M) & (cols[None, :] < N))


def make_launch(inputs, output, config, compiled):
    a, b = inputs
    m, k = a.shape
    _, n = b.shape
    bm = config.get("block_m", 16)
    bn = config.get("block_n", 32)
    bk = config.get("block_k", 16)
    warps = config.get("num_warps", 4)
    stages = config.get("num_stages", 2)
    if (bm, bn, bk) not in {(16, 32, 16), (32, 32, 16)}:
        raise ValueError("Supported GEMM tiles: (16,32,16), (32,32,16)")
    if warps not in {4, 8} or stages not in {1, 2, 3}:
        raise ValueError("GEMM num_warps must be 4/8; num_stages must be 1/2/3")
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))

    def launch():
        compiled["gemm"] = gemm_kernel[grid](
            a, b, output, m, n, k, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
            num_warps=warps, num_stages=stages, enable_fp_fusion=False)

    return launch, {
        "mapping": "one program per output M/N tile; sequential K tiles with masked zero padding; tl.dot",
        "grid": grid,
        "block_m": bm,
        "block_n": bn,
        "block_k": bk,
        "num_warps": warps,
        "threads_per_program": 32 * warps,
        "num_stages": stages,
        "precision": "FP16 inputs; tl.dot with FP32 accumulator; one final FP16 cast; no TF32 inputs",
        "launches_per_call": 1,
        "pipeline": "compiler-managed num_stages; no explicit async/TMA/warp-specialized pipeline",
    }

"""Transpose with explicit ownership conversion, plus 2-D mapping probes."""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout


@g.jit
def transpose_kernel(A, Y, M: gl.constexpr, N: gl.constexpr,
                     TILE: gl.constexpr, CONVERT: gl.constexpr):
    src: gl.constexpr = BlockedLayout([1, 1], [4, 8], [4, 1], [1, 0])
    dst: gl.constexpr = BlockedLayout([1, 1], [8, 4], [1, 4], [0, 1])
    rows = gl.program_id(0) * TILE + gl.arange(0, TILE, layout=SliceLayout(1, src))
    cols = gl.program_id(1) * TILE + gl.arange(0, TILE, layout=SliceLayout(0, src))
    value = gl.load(A + rows[:, None] * N + cols[None, :],
                    (rows[:, None] < M) & (cols[None, :] < N), other=0)
    if CONVERT:
        value = gl.convert_layout(value, dst)
        out_rows = gl.program_id(0) * TILE + gl.arange(0, TILE, layout=SliceLayout(1, dst))
        out_cols = gl.program_id(1) * TILE + gl.arange(0, TILE, layout=SliceLayout(0, dst))
        gl.store(Y + out_cols[None, :] * M + out_rows[:, None], value,
                 (out_rows[:, None] < M) & (out_cols[None, :] < N))
    else:
        gl.store(Y + cols[None, :] * M + rows[:, None], value,
                 (rows[:, None] < M) & (cols[None, :] < N))


@g.jit
def mapping_2d_kernel(Owners, ROWS: gl.constexpr, COLS: gl.constexpr,
                      SPT: gl.constexpr, TPW: gl.constexpr,
                      WPC: gl.constexpr, ORDER: gl.constexpr):
    layout: gl.constexpr = BlockedLayout(SPT, TPW, WPC, ORDER)
    rows = gl.arange(0, ROWS, layout=SliceLayout(1, layout))
    cols = gl.arange(0, COLS, layout=SliceLayout(0, layout))
    offset = rows[:, None] * COLS + cols[None, :]
    tid = gl.inline_asm_elementwise("mov.u32 $0, %tid.x;", "=r,r", [offset],
                                   dtype=gl.int32, is_pure=True, pack=1)
    gl.store(Owners + offset, tid)

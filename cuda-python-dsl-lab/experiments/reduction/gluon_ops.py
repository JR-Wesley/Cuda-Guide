"""Two explicit row-reduction ownership layouts with the same algorithm."""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout


@g.jit
def row_sum_kernel(A, Y, M: gl.constexpr, N: gl.constexpr,
                   BLOCK: gl.constexpr, WARPS_PER_ROW: gl.constexpr):
    # Both variants process four rows per CTA. One assigns a warp to each row;
    # the other assigns four warps to a row and holds four rows in registers.
    layout: gl.constexpr = BlockedLayout(
        [1, 1], [1, 32], [4 // WARPS_PER_ROW, WARPS_PER_ROW], [1, 0])
    rows = gl.program_id(0) * 4 + gl.arange(0, 4, layout=SliceLayout(1, layout))
    cols = gl.arange(0, BLOCK, layout=SliceLayout(0, layout))
    value = gl.load(A + rows[:, None] * N + cols[None, :],
                    (rows[:, None] < M) & (cols[None, :] < N), other=0)
    result = gl.sum(value, 1)
    gl.store(Y + rows, result, rows < M)

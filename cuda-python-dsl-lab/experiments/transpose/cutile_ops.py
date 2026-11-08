"""U5.2: materialize a transpose into a distinct C-contiguous array."""

import cuda.tile as ct


@ct.kernel
def transpose_kernel(a, out, ROWS: ct.Constant[int], COLS: ct.Constant[int]):
    row, col = ct.bid(0), ct.bid(1)
    tile = ct.load(a, (row, col), (ROWS, COLS), padding_mode=ct.PaddingMode.ZERO)
    ct.store(out, (col, row), ct.transpose(tile))

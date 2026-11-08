"""U5.2: one row per tile block, with zero padding to a power of two."""

import cuda.tile as ct


@ct.kernel
def row_sum_kernel(a, out, COLS: ct.Constant[int]):
    row = ct.bid(0)
    tile = ct.load(a, (row, 0), (1, COLS), padding_mode=ct.PaddingMode.ZERO)
    total = ct.sum(tile, axis=1)
    ct.store(out, (row,), total)

"""U5.1: array partitions become compiler-mapped tiles, not thread blocks."""

import cuda.tile as ct


@ct.kernel
def copy_kernel(a, out, BLOCK: ct.Constant[int]):
    block = ct.bid(0)
    tile = ct.load(a, (block,), (BLOCK,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(out, (block,), tile)


@ct.kernel
def add_kernel(a, b, out, BLOCK: ct.Constant[int]):
    block = ct.bid(0)
    x = ct.load(a, (block,), (BLOCK,), padding_mode=ct.PaddingMode.ZERO)
    y = ct.load(b, (block,), (BLOCK,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(out, (block,), x + y)

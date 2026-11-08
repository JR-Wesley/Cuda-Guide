"""U5.3/U5.4: tiled FP16 MMA with FP32 accumulation and M/N/K tails."""

import cuda.tile as ct


@ct.kernel
def gemm_kernel(a, b, out, BM: ct.Constant[int], BN: ct.Constant[int],
                BK: ct.Constant[int]):
    row, col = ct.bid(0), ct.bid(1)
    acc = ct.full((BM, BN), 0.0, dtype=ct.float32)
    for k in range(ct.cdiv(a.shape[1], BK)):
        x = ct.load(a, (row, k), (BM, BK), padding_mode=ct.PaddingMode.ZERO)
        y = ct.load(b, (k, col), (BK, BN), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(x, y, acc)
    ct.store(out, (row, col), ct.astype(acc, ct.float16))

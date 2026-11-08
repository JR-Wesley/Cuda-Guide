import numpy as np
import cupy as cp
import cuda.tile as ct


@ct.kernel
def vector_add(a, b, c, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(BLOCK,), padding_mode=ct.PaddingMode.ZERO)
    y = ct.load(b, index=(pid,), shape=(BLOCK,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(c, index=(pid,), tile=x + y)


if __name__ == "__main__":
    rng = np.random.default_rng(20261104)
    print("CUDA driver API:", cp.cuda.runtime.driverGetVersion())
    print("CUDA runtime:", cp.cuda.runtime.runtimeGetVersion())
    for n in (1, 255, 256, 4099, 65536):
        a_cpu = rng.standard_normal(n, dtype=np.float32)
        b_cpu = rng.standard_normal(n, dtype=np.float32)
        a, b = cp.asarray(a_cpu), cp.asarray(b_cpu)
        c = cp.empty_like(a)
        for repeat in range(2):
            c.fill(cp.nan)
            ct.launch(cp.cuda.get_current_stream(), (ct.cdiv(n, 256), 1, 1), vector_add, (a, b, c, 256))
            cp.cuda.runtime.deviceSynchronize()
            np.testing.assert_allclose(cp.asnumpy(c), a_cpu + b_cpu, rtol=1e-6, atol=1e-6)
        print(f"PASS cuTile: n={n}, dtype=float32, repeats=2, CPU reference")

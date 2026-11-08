"""Same CPU-generated inputs and correctness criteria for Torch-backed DSLs."""
import torch

SIZES = (1, 255, 256, 4099, 65536)


def validate(name, launch):
    generator = torch.Generator().manual_seed(20261104)
    for n in SIZES:
        a_cpu = torch.randn(n, generator=generator, dtype=torch.float32)
        b_cpu = torch.randn(n, generator=generator, dtype=torch.float32)
        expected = a_cpu + b_cpu
        a, b = a_cpu.cuda(), b_cpu.cuda()
        c = torch.full_like(a, float("nan"))
        torch.cuda.synchronize()
        for repeat in range(2):
            c.fill_(float("nan"))
            torch.cuda.synchronize()
            launch(a, b, c, n)
            torch.cuda.synchronize()
            torch.testing.assert_close(c.cpu(), expected, rtol=1e-6, atol=1e-6)
        print(f"PASS {name}: n={n}, dtype=float32, repeats=2, CPU reference")


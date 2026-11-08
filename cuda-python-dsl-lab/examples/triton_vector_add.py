import triton
import triton.language as tl
from torch_inputs import validate


@triton.jit
def vector_add(a, b, c, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(a + offsets, mask=mask, other=0)
    y = tl.load(b + offsets, mask=mask, other=0)
    tl.store(c + offsets, x + y, mask=mask)


if __name__ == "__main__":
    validate("Triton", lambda a, b, c, n: vector_add[(triton.cdiv(n, 256),)](a, b, c, n, BLOCK=256))

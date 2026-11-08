import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout
from torch_inputs import validate


@g.jit
def vector_add(a, b, c, n, BLOCK: gl.constexpr):
    offsets = gl.program_id(0) * BLOCK + gl.arange(0, BLOCK, layout=BlockedLayout([1], [32], [4], [0]))
    mask = offsets < n
    x = gl.load(a + offsets, mask=mask, other=0)
    y = gl.load(b + offsets, mask=mask, other=0)
    gl.store(c + offsets, x + y, mask=mask)


if __name__ == "__main__":
    validate("Gluon", lambda a, b, c, n: vector_add[(triton.cdiv(n, 256),)](a, b, c, n, BLOCK=256))

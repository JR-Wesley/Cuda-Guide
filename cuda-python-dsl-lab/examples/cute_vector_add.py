import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from torch_inputs import validate


@cute.kernel
def vector_add(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, n: cutlass.Int32):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    index = bid * 256 + tid
    if index < n:
        c[index] = a[index] + b[index]


@cute.jit
def launch(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, n: cutlass.Int32):
    vector_add(a, b, c, n).launch(grid=((n + 255) // 256, 1, 1), block=(256, 1, 1))


if __name__ == "__main__":
    validate("CuTe DSL", lambda a, b, c, n: launch(from_dlpack(a), from_dlpack(b), from_dlpack(c), n))

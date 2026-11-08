"""Small SM89 teaching kernels using the installed CuTe DSL 4.8.1 API.

GPU compilation/execution is deliberately left to the serialized common runner.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import CopyUniversalOp
from cutlass.cute.nvgpu.warp import MmaF16BF16Op
from cutlass.utils import SmemAllocator


class Vector:
    def __init__(self, add=False):
        self.add = add

    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, out: cute.Tensor,
                 stream: cuda.CUstream):
        self.kernel(a, b, out).launch(
            grid=(cute.ceil_div(cute.size(out), 256), 1, 1),
            block=(256, 1, 1), stream=stream,
        )

    @cute.kernel
    def kernel(self, a: cute.Tensor, b: cute.Tensor, out: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        index = bid * 256 + tid
        if index < cute.size(out):
            if cutlass.const_expr(self.add):
                out[index] = a[index] + b[index]
            else:
                out[index] = a[index]


class Transpose:
    def __init__(self, shared=False):
        self.shared = shared

    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, out: cute.Tensor,
                 stream: cuda.CUstream):
        if cutlass.const_expr(self.shared):
            self.shared_kernel(a, out).launch(
                grid=(cute.ceil_div(a.shape[0], 32),
                      cute.ceil_div(a.shape[1], 32), 1),
                block=(256, 1, 1), stream=stream,
            )
        else:
            self.naive_kernel(a, out).launch(
                grid=(cute.ceil_div(cute.size(a), 256), 1, 1),
                block=(256, 1, 1), stream=stream,
            )

    @cute.kernel
    def naive_kernel(self, a: cute.Tensor, out: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        index = bid * 256 + tid
        if index < cute.size(a):
            row = index // a.shape[1]
            col = index % a.shape[1]
            out[col, row] = a[row, col]

    @cute.kernel
    def shared_kernel(self, a: cute.Tensor, out: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bm, bn, _ = cute.arch.block_idx()
        smem = SmemAllocator()
        tile = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((32, 32), stride=(33, 1)),
            byte_alignment=16,
        )
        for j in cutlass.range_constexpr(4):
            index = tid + 256 * j
            row, col = index // 32, index % 32
            m, n = bm * 32 + row, bn * 32 + col
            tile[row, col] = cutlass.Float32(0)
            if m < a.shape[0] and n < a.shape[1]:
                tile[row, col] = a[m, n]
        # The consumer uses transposed thread ownership, not just a new address.
        cute.arch.sync_threads()
        for j in cutlass.range_constexpr(4):
            index = tid + 256 * j
            row, col = index % 32, index // 32
            m, n = bm * 32 + row, bn * 32 + col
            if m < a.shape[0] and n < a.shape[1]:
                out[n, m] = tile[row, col]


class SharedCopy:
    """Fixed 512-element tile; change value layout/atom width, not tile size."""

    def __init__(self, values=1):
        if values not in (1, 4):
            raise ValueError("SharedCopy supports values=1 or 4")
        self.values = values

    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, out: cute.Tensor,
                 stream: cuda.CUstream):
        self.kernel(a, out).launch(
            grid=(cute.ceil_div(cute.size(a), 512), 1, 1),
            block=(128, 1, 1), stream=stream,
        )

    @cute.kernel
    def kernel(self, a: cute.Tensor, out: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        atom = cute.make_copy_atom(
            CopyUniversalOp(), cutlass.Float32,
            num_bits_per_copy=32 * self.values,
        )
        tiled = cute.make_tiled_copy_tv(
            atom, cute.make_layout((1, 128), stride=(128, 1)),
            cute.make_layout((1, self.values)),
        )
        smem = SmemAllocator()
        shared = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((1, 512), stride=(512, 1)),
            byte_alignment=16,
        )
        # Reinterpret the contiguous vector as a one-row tensor; no data copy.
        ga = cute.make_tensor(a.iterator, cute.make_layout(
            (1, cute.size(a)), stride=(cute.size(a), 1)))
        go = cute.make_tensor(out.iterator, ga.layout)
        src = cute.local_tile(ga, (1, 512), (0, bid))
        dst = cute.local_tile(go, (1, 512), (0, bid))
        thread = tiled.get_slice(tid)
        ts = thread.partition_S(src)
        td = thread.partition_D(shared)
        ss = thread.partition_S(shared)
        gd = thread.partition_D(dst)
        coords = thread.partition_S(cute.make_identity_tensor((1, 512)))
        if bid * 512 + 512 <= cute.size(a):
            cute.copy(tiled, ts, td)
        else:
            # Never issue a partially valid 128-bit transaction on a tail.
            for j in cutlass.range_constexpr(cute.size(ts)):
                index = bid * 512 + coords[j][1]
                td[j] = cutlass.Float32(0)
                if index < cute.size(a):
                    td[j] = ts[j]
        # Conservative CTA visibility barrier; producer/consumer ownership agrees.
        cute.arch.sync_threads()
        if bid * 512 + 512 <= cute.size(a):
            cute.copy(tiled, ss, gd)
        else:
            for j in cutlass.range_constexpr(cute.size(gd)):
                index = bid * 512 + coords[j][1]
                if index < cute.size(a):
                    gd[j] = ss[j]


class WarpGemm:
    """One warp/CTA, synchronous shared staging, genuine m16n8k16 MMA.

    B arrives in the common runner's (K,N) row-major format. Cooperative
    loads stage it as (N,K) K-major for the warp MMA's B operand.
    """

    def __init__(self, tile=(32, 32, 16), stages=1):
        if tuple(tile) not in ((16, 8, 16), (16, 16, 16), (32, 32, 16)):
            raise ValueError("Unsupported teaching tile")
        if stages != 1:
            raise NotImplementedError("Only synchronous single-stage GEMM is implemented")
        self.bm, self.bn, self.bk = tile

    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, out: cute.Tensor,
                 stream: cuda.CUstream):
        self.kernel(a, b, out).launch(
            grid=(cute.ceil_div(a.shape[0], self.bm),
                  cute.ceil_div(b.shape[1], self.bn), 1),
            block=(32, 1, 1), stream=stream,
        )

    @cute.kernel
    def kernel(self, a: cute.Tensor, b: cute.Tensor, out: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        block_m, block_n, _ = cute.arch.block_idx()
        smem = SmemAllocator()
        sa = smem.allocate_tensor(
            cutlass.Float16,
            cute.make_layout((self.bm, self.bk), stride=(self.bk, 1)),
            byte_alignment=16,
        )
        sb = smem.allocate_tensor(
            cutlass.Float16,
            cute.make_layout((self.bn, self.bk), stride=(self.bk, 1)),
            byte_alignment=16,
        )
        tiled_mma = cute.make_tiled_mma(
            MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (16, 8, 16)),
            (1, 1, 1),
        )
        thread = tiled_mma.get_slice(tid)
        ta = thread.partition_A(sa)
        tb = thread.partition_B(sb)
        coords = thread.partition_C(cute.make_identity_tensor((self.bm, self.bn)))
        ra = tiled_mma.make_fragment_A(ta)
        rb = tiled_mma.make_fragment_B(tb)
        acc = tiled_mma.make_fragment_C(coords.shape)
        acc.fill(0.0)
        for kt in range(cute.ceil_div(a.shape[1], self.bk)):
            for j in cutlass.range_constexpr(self.bm * self.bk // 32):
                index = tid + 32 * j
                row, k = index // self.bk, index % self.bk
                gm, gk = block_m * self.bm + row, kt * self.bk + k
                sa[row, k] = cutlass.Float16(0)
                if gm < a.shape[0] and gk < a.shape[1]:
                    sa[row, k] = a[gm, gk]
            for j in cutlass.range_constexpr(self.bn * self.bk // 32):
                index = tid + 32 * j
                col, k = index // self.bk, index % self.bk
                gn, gk = block_n * self.bn + col, kt * self.bk + k
                sb[col, k] = cutlass.Float16(0)
                if gn < b.shape[1] and gk < b.shape[0]:
                    sb[col, k] = b[gk, gn]
            cute.arch.sync_threads()
            # Scalar *loads* into instruction-defined fragments, not scalar GEMM.
            for j in cutlass.range_constexpr(cute.size(ra)):
                ra[j] = ta[j]
            for j in cutlass.range_constexpr(cute.size(rb)):
                rb[j] = tb[j]
            cute.gemm(tiled_mma, acc, ra, rb, acc)
            # Protect shared reads before the next iteration overwrites the tile.
            cute.arch.sync_threads()
        for j in cutlass.range_constexpr(cute.size(acc)):
            coord = coords[j]
            m, n = block_m * self.bm + coord[0], block_n * self.bn + coord[1]
            if m < out.shape[0] and n < out.shape[1]:
                out[m, n] = cutlass.Float16(acc[j])

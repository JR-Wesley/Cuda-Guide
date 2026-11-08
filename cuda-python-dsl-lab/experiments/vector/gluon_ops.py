"""Gluon vector kernels; importing this module does not initialize CUDA."""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SwizzledSharedLayout
from triton.experimental.gluon.language.nvidia.ampere import async_copy


@g.jit
def add_kernel(A, B, Y, N: gl.constexpr, BLOCK: gl.constexpr, SPT: gl.constexpr):
    layout: gl.constexpr = BlockedLayout([SPT], [32], [4], [0])
    x = gl.program_id(0) * BLOCK + gl.arange(0, BLOCK, layout=layout)
    a = gl.load(A + x, x < N, other=0)
    b = gl.load(B + x, x < N, other=0)
    gl.store(Y + x, a + b, x < N)


@g.jit
def copy_kernel(A, Y, N: gl.constexpr, BLOCK: gl.constexpr,
                TILES: gl.constexpr, MODE: gl.constexpr):
    """MODE: registers, shared_sync, async_single, or async_double.

    Each CTA copies TILES consecutive blocks, so single/double buffering have
    identical work assignment, masks, and number of global memory operations.
    """
    layout: gl.constexpr = BlockedLayout([1], [32], [4], [0])
    x = gl.arange(0, BLOCK, layout=layout)
    base = gl.program_id(0) * (TILES * BLOCK)
    if MODE == "registers":
        for tile in gl.static_range(TILES):
            offset = base + tile * BLOCK + x
            value = gl.load(A + offset, offset < N, other=0)
            gl.store(Y + offset, value, offset < N)
    elif MODE == "shared_sync":
        smem = gl.allocate_shared_memory(gl.float32, [BLOCK],
                                        SwizzledSharedLayout(1, 1, 1, [0]))
        for tile in gl.static_range(TILES):
            offset = base + tile * BLOCK + x
            value = gl.load(A + offset, offset < N, other=0)
            smem.store(value)
            gl.barrier()
            value = smem.load(layout)
            gl.barrier()
            gl.store(Y + offset, value, offset < N)
    elif MODE == "async_single":
        smem = gl.allocate_shared_memory(gl.float32, [BLOCK],
                                        SwizzledSharedLayout(1, 1, 1, [0]))
        for tile in gl.static_range(TILES):
            offset = base + tile * BLOCK + x
            async_copy.async_load(smem, A + offset, offset < N)
            async_copy.commit_group()
            async_copy.wait_group(0)
            gl.barrier()
            value = smem.load(layout)
            # All readers finish before the next iteration overwrites smem.
            gl.barrier()
            gl.store(Y + offset, value, offset < N)
    elif MODE == "async_double":
        # The extra leading dimension selects buffers, not layout dimensions.
        smem = gl.allocate_shared_memory(gl.float32, [2, BLOCK],
                                        SwizzledSharedLayout(1, 1, 1, [0]))
        async_copy.async_load(smem.index(0), A + base + x, base + x < N)
        async_copy.commit_group()
        for tile in gl.static_range(TILES):
            if tile + 1 < TILES:
                next_offset = base + (tile + 1) * BLOCK + x
                async_copy.async_load(smem.index((tile + 1) % 2),
                                      A + next_offset, next_offset < N)
                async_copy.commit_group()
                # The oldest tile is ready; the next group may remain in flight.
                async_copy.wait_group(1)
            else:
                async_copy.wait_group(0)
            gl.barrier()
            value = smem.index(tile % 2).load(layout)
            # Release this slot only after every thread has consumed it.
            gl.barrier()
            offset = base + tile * BLOCK + x
            gl.store(Y + offset, value, offset < N)
    else:
        gl.static_assert(False, "unknown copy mode")


@g.jit
def mapping_1d_kernel(Owners, BLOCK: gl.constexpr, SPT: gl.constexpr):
    x = gl.arange(0, BLOCK, layout=BlockedLayout([SPT], [32], [4], [0]))
    tid = gl.inline_asm_elementwise("mov.u32 $0, %tid.x;", "=r,r", [x],
                                   dtype=gl.int32, is_pure=True, pack=1)
    gl.store(Owners + x, tid)

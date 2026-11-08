#pragma once

#include "common.cuh"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>
#include <vector>

static constexpr int transpose_tile = 32;
static constexpr int transpose_block_rows = 8;

__global__ void transpose_naive(const float* in, float* out, int n) {
    const int x = blockIdx.x * transpose_tile + threadIdx.x;
    const int y = blockIdx.y * transpose_tile + threadIdx.y;
    for (int j = 0; j < transpose_tile; j += transpose_block_rows) {
        if (x < n && y + j < n) {
            out[std::size_t(x) * n + y + j] = in[std::size_t(y + j) * n + x];
        }
    }
}

template<int Pad>
__global__ void transpose_tiled(const float* in, float* out, int n) {
    __shared__ float tile[transpose_tile][transpose_tile + Pad];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * transpose_tile + tx;
    const int y = blockIdx.y * transpose_tile + ty;
    for (int j = 0; j < transpose_tile; j += transpose_block_rows) {
        if (x < n && y + j < n) {
            tile[ty + j][tx] = in[std::size_t(y + j) * n + x];
        }
    }
    __syncthreads();

    // Swap both the tile coordinates and the coordinates within the tile.
    const int out_x = blockIdx.y * transpose_tile + tx;
    const int out_y = blockIdx.x * transpose_tile + ty;
    for (int j = 0; j < transpose_tile; j += transpose_block_rows) {
        if (out_x < n && out_y + j < n) {
            out[std::size_t(out_y + j) * n + out_x] = tile[tx][ty + j];
        }
    }
}

inline int run_transpose(int n, int variant) {
    // grid.y is limited to 65535; size_t keeps matrix offsets out of int arithmetic.
    if (n <= 0 || n > 65535 * transpose_tile || variant < 0 || variant > 2) {
        std::fprintf(stderr, "transpose: require 1 <= N <= %d and variant in {0,1,2}\n",
                     65535 * transpose_tile);
        return 1;
    }
    const std::size_t count = std::size_t(n) * n;
    const std::size_t bytes = count * sizeof(float);
    std::vector<float> input(count), output(count);
    for (std::size_t i = 0; i < count; ++i) {
        input[i] = (int(i % 251) - 125) * 0.125f;
    }
    DeviceBuffer<float> device_in(count), device_out(count);
    CUDA_CHECK(cudaMemcpy(device_in.data(), input.data(), bytes, cudaMemcpyHostToDevice));
    // A finite expected value can never equal this NaN sentinel (all bits set).
    CUDA_CHECK(cudaMemset(device_out.data(), 0xff, bytes));
    const dim3 block(transpose_tile, transpose_block_rows);
    const dim3 grid((n + transpose_tile - 1) / transpose_tile,
                    (n + transpose_tile - 1) / transpose_tile);
    const float ms = time_kernel([&] {
        if (variant == 0) {
            transpose_naive<<<grid, block>>>(device_in.data(), device_out.data(), n);
        } else if (variant == 1) {
            transpose_tiled<0><<<grid, block>>>(device_in.data(), device_out.data(), n);
        } else {
            transpose_tiled<1><<<grid, block>>>(device_in.data(), device_out.data(), n);
        }
    });
    CUDA_CHECK(cudaMemcpy(output.data(), device_out.data(), bytes, cudaMemcpyDeviceToHost));
    std::size_t errors = 0;
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            const float expected = input[std::size_t(y) * n + x];
            const float actual = output[std::size_t(x) * n + y];
            if (actual != expected) {
                if (errors == 0) {
                    std::fprintf(stderr, "transpose mismatch: input(y=%d,x=%d), got=%g expected=%g\n",
                                 y, x, actual, expected);
                }
                ++errors;
            }
        }
    }

    const char* names[] = {"naive", "tiled", "padded"};
    std::printf("transpose variant=%d(%s) N=%d block=(32,8) logical_bytes=%zu "
                "kernel_ms=%.6f check=%s errors=%zu\n",
                variant, names[variant], n, 2 * bytes, ms,
                errors == 0 ? "PASS" : "FAIL", errors);
    return errors == 0 ? 0 : 1;
}

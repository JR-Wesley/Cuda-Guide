#pragma once

#include "common.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

static constexpr int reduction_block_size = 256;

__global__ void reduction_shared(const float* input, float* partial_sums, int n) {
    __shared__ float values[reduction_block_size];
    const int tid = threadIdx.x;
    const std::size_t i = std::size_t(blockIdx.x) * reduction_block_size + tid;
    values[tid] = i < std::size_t(n) ? input[i] : 0.0f;
    __syncthreads();
    for (int offset = reduction_block_size / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            values[tid] += values[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial_sums[blockIdx.x] = values[0];
    }
}

__global__ void reduction_shuffle(const float* input, float* partial_sums, int n) {
    __shared__ float warp_sums[8];
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const std::size_t i = std::size_t(blockIdx.x) * reduction_block_size + tid;
    float value = i < std::size_t(n) ? input[i] : 0.0f;
    // No early return: even lanes beyond n participate with a zero input.
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        value = lane < 8 ? warp_sums[lane] : 0.0f;
        // All 32 lanes of warp 0 participate, including the 24 zero lanes.
        for (int offset = 16; offset > 0; offset /= 2) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (lane == 0) {
            partial_sums[blockIdx.x] = value;
        }
    }
}

inline int run_reduction(int n, bool use_shuffle) {
    if (n <= 0) {
        std::fprintf(stderr, "reduction: require N > 0\n");
        return 1;
    }
    const int blocks = (n - 1) / reduction_block_size + 1;
    const std::size_t bytes = std::size_t(n) * sizeof(float);
    const std::size_t partial_bytes = std::size_t(blocks) * sizeof(float);
    std::vector<float> input(n), partial_sums(blocks);
    for (int i = 0; i < n; ++i) {
        input[i] = (i % 7 - 3) * 0.125f;
    }
    DeviceBuffer<float> device_in(n), device_partial(blocks);
    CUDA_CHECK(cudaMemcpy(device_in.data(), input.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device_partial.data(), 0xff, partial_bytes));
    const float ms = time_kernel([&] {
        if (use_shuffle) {
            reduction_shuffle<<<blocks, reduction_block_size>>>(device_in.data(), device_partial.data(), n);
        } else {
            reduction_shared<<<blocks, reduction_block_size>>>(device_in.data(), device_partial.data(), n);
        }
    });
    CUDA_CHECK(cudaMemcpy(partial_sums.data(), device_partial.data(), partial_bytes,
                          cudaMemcpyDeviceToHost));
    double reference_total = 0.0;
    double actual_total = 0.0;
    int errors = 0;
    for (int b = 0; b < blocks; ++b) {
        double reference = 0.0;
        for (int t = 0; t < reduction_block_size; ++t) {
            const std::size_t i = std::size_t(b) * reduction_block_size + t;
            if (i < std::size_t(n)) {
                reference += double(input[i]);
            }
        }
        const double tolerance = 1e-6 * (1.0 + std::abs(reference));
        if (!(std::abs(double(partial_sums[b]) - reference) <= tolerance)) {
            if (errors == 0) {
                std::fprintf(stderr, "reduction mismatch: block=%d got=%g expected=%.17g\n",
                             b, partial_sums[b], reference);
            }
            ++errors;
        }
        reference_total += reference;
        actual_total += double(partial_sums[b]);
    }
    const bool total_ok = std::abs(actual_total - reference_total)
                         <= 1e-6 * (1.0 + std::abs(reference_total));

    const bool ok = errors == 0 && total_ok;
    std::printf("reduction variant=%s N=%d block=256 partials=%d "
                "partial_gpu_ms=%.6f cpu_sum=%.17g reference=%.17g "
                "check=%s block_errors=%d\n",
                use_shuffle ? "shuffle" : "shared", n, blocks, ms,
                actual_total, reference_total, ok ? "PASS" : "FAIL", errors);
    return ok ? 0 : 1;
}

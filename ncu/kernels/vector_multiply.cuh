#pragma once

#include "common.cuh"

#include <cstddef>
#include <cstdio>
#include <limits>
#include <vector>

template <int Stride>
__global__ void vector_mul(const float* a, const float* b, float* c, int n) {
    static_assert(Stride > 0, "Stride must be positive");
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < static_cast<std::size_t>(n)) {
        const std::size_t input = i * static_cast<std::size_t>(Stride);
        c[i] = a[input] * b[input];
    }
}

template <int Stride>
inline int run_vector(int n, int block) {
    if (n <= 0 || block <= 0 || block > 1024) {
        std::fprintf(stderr, "Require n > 0, 1 <= block <= 1024\n");
        return 1;
    }
    const std::size_t max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (static_cast<std::size_t>(n - 1) >
        (max_elements - 1) / static_cast<std::size_t>(Stride)) {
        std::fprintf(stderr, "Input storage size overflows size_t\n");
        return 1;
    }
    const std::size_t storage = static_cast<std::size_t>(n - 1) * Stride + 1;
    const std::size_t input_bytes = storage * sizeof(float);
    const std::size_t output_bytes = static_cast<std::size_t>(n) * sizeof(float);
    std::vector<float> a(storage, 0.0f), b(storage, 0.0f), c(n);
    for (int i = 0; i < n; ++i) {
        const std::size_t input = static_cast<std::size_t>(i) * Stride;
        // 只改变地址间隔，不改变各线程参与计算的值。
        a[input] = static_cast<float>(i % 1024) * 0.125f;
        b[input] = 2.0f;
    }

    DeviceBuffer<float> d_a(storage), d_b(storage), d_c(n);
    CUDA_CHECK(cudaMemcpy(d_a.data(), a.data(), input_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b.data(), b.data(), input_bytes, cudaMemcpyHostToDevice));

    const int grid = (n - 1) / block + 1;
    const float average_ms = time_kernel([&] {
        vector_mul<Stride><<<grid, block>>>(d_a.data(), d_b.data(), d_c.data(), n);
    });
    CUDA_CHECK(cudaMemcpy(c.data(), d_c.data(), output_bytes, cudaMemcpyDeviceToHost));
    int mismatches = 0;
    for (int i = 0; i < n; ++i) {
        const float expected = static_cast<float>(i % 1024) * 0.25f;
        if (c[i] != expected) {
            if (mismatches == 0) {
                std::fprintf(stderr, "FAIL: i=%d, actual=%g, expected=%g\n", i, c[i], expected);
            }
            ++mismatches;
        }
    }

    std::printf("%s: N=%d, grid=%d, block=%d, stride=%d, checked=%d, mismatches=%d\n",
                mismatches == 0 ? "PASS" : "FAIL", n, grid, block, Stride, n, mismatches);
    std::printf("storage_elements_per_input=%zu, allocated_bytes=%.0f, logical_bytes=%.0f\n",
                storage, 2.0 * input_bytes + output_bytes, 12.0 * n);
    std::printf("launches=30 (warmup=10, timed=20), event_average_ms=%.6f\n", average_ms);
    std::printf("Event batch average excludes allocation/copies/checking; may include submission gaps.\n");
    std::printf("Event timing under NCU is NOT an independent benchmark.\n");
    if (n <= 16) {
        for (int i = 0; i < n; ++i) {
            const std::size_t input = static_cast<std::size_t>(i) * Stride;
            std::printf("c[%d] = %g * %g = %g\n", i, a[input], b[input], c[i]);
        }
    }

    return mismatches == 0 ? 0 : 1;
}

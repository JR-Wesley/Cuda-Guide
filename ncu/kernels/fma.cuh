#pragma once

#include "common.cuh"

#include <cmath>
#include <cstdio>
#include <vector>

__host__ __device__ inline float fma_initial(int i, int j) {
    return 0.5f + static_cast<float>(i) * 0x1p-16f
                + static_cast<float>(j + 1) * 0x1p-8f;
}

template<int Chains>
__global__ void fma_chain(float* out, int n, int total_fmas_per_thread,
                          float multiplier, float addend) {
    static_assert(Chains >= 1 && Chains <= 64, "Chains must be in [1, 64]");
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float acc[Chains];
#pragma unroll
    for (int j = 0; j < Chains; ++j) {
        acc[j] = fma_initial(i, j);
    }

    const int rounds = total_fmas_per_thread / Chains;
#pragma unroll 1
    for (int r = 0; r < rounds; ++r) {
#pragma unroll
        for (int j = 0; j < Chains; ++j) {
            acc[j] = fmaf(acc[j], multiplier, addend);
        }
    }

    float sum = acc[0];
#pragma unroll
    for (int j = 1; j < Chains; ++j) {
        sum += acc[j];
    }
    out[i] = sum;
}

template<int Chains>
inline int run_fma(int n, int block, int total_fmas_per_thread) {
    static_assert(Chains >= 1 && Chains <= 64, "Chains must be in [1, 64]");
    if (n <= 0 || block <= 0 || block > 1024 ||
        total_fmas_per_thread <= 0 || total_fmas_per_thread % Chains != 0) {
        std::fprintf(stderr,
                     "Require n > 0, block in [1, 1024], total > 0 and total %% Chains == 0\n");
        return 1;
    }

    // These are kernel arguments, not constants embedded in the device loop.
    const float multiplier = 1.0001f;
    const float addend = 0.0001f;
    const int rounds = total_fmas_per_thread / Chains;
    const int grid = (n - 1) / block + 1;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    std::vector<float> output(n);
    DeviceBuffer<float> device_output(n);

    const float ms = time_kernel([&] {
        fma_chain<Chains><<<grid, block>>>(device_output.data(), n, total_fmas_per_thread,
                                         multiplier, addend);
    });
    CUDA_CHECK(cudaMemcpy(output.data(), device_output.data(), bytes, cudaMemcpyDeviceToHost));

    // CPU recurrence and all output checks are outside the GPU timing interval.
    const float atol = 1e-5f;
    const float rtol = 1e-4f;
    int mismatches = 0;
    float max_abs_error = 0.0f;
    for (int i = 0; i < n; ++i) {
        float acc[Chains];
        for (int j = 0; j < Chains; ++j) acc[j] = fma_initial(i, j);
        for (int r = 0; r < rounds; ++r) {
            for (int j = 0; j < Chains; ++j) {
                acc[j] = fmaf(acc[j], multiplier, addend);
            }
        }
        float expected = acc[0];
        for (int j = 1; j < Chains; ++j) expected += acc[j];

        const float error = std::fabs(output[i] - expected);
        const bool finite = std::isfinite(output[i]) && std::isfinite(expected);
        if (!finite || error > atol + rtol * std::fabs(expected)) {
            if (mismatches == 0) {
                std::fprintf(stderr, "First mismatch: i=%d GPU=%.9g CPU=%.9g\n",
                             i, output[i], expected);
            }
            ++mismatches;
        }
        if (finite && error > max_abs_error) max_abs_error = error;
    }


    const unsigned long long logical_fma_flops =
        2ULL * static_cast<unsigned long long>(n) * total_fmas_per_thread;
    std::printf("FMA: N=%d block=%d Chains=%d total/thread=%d rounds=%d\n",
                n, block, Chains, total_fmas_per_thread, rounds);
    std::printf("multiplier=%.9g addend=%.9g average_kernel_ms=%.6f\n",
                multiplier, addend, ms);
    std::printf("logical_FMA_FLOPs=%llu (excludes final %d additions/thread)\n",
                logical_fma_flops, Chains - 1);
    if (ms > 0.0f) {
        std::printf("logical_FMA_GFLOP/s=%.3f\n", logical_fma_flops / (ms * 1e6));
    }
    std::printf("Check: %s, checked=%d mismatches=%d max_finite_abs_error=%.9g "
                "atol=%g rtol=%g\n",
                mismatches == 0 ? "PASS" : "FAIL", n, mismatches,
                max_abs_error, atol, rtol);
    return mismatches == 0 ? 0 : 1;
}

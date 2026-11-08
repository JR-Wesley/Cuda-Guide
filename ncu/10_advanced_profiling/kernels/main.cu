#include "../../kernels/common.cuh"
#include "../../kernels/vector_multiply.cuh"
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include <cstdio>
#include <vector>

int main() {
    constexpr int n = 16;
    constexpr int block = 256;
    constexpr int warmup = 10;
    constexpr int repeats = 20;
    constexpr int stride = 1;
    const size_t bytes = size_t(n) * sizeof(float);
    std::vector<float> a(n), b(n), result(n);
    for (int i = 0; i < n; ++i) {
        a[i] = float(i % 17);
        b[i] = float(i % 7 + 1);
    }

    CUDA_CHECK(cudaSetDevice(0));
    DeviceBuffer<float> d_a(n), d_b(n), d_c(n);
    CUDA_CHECK(cudaMemcpy(d_a.data(), a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b.data(), b.data(), bytes, cudaMemcpyHostToDevice));
    const int grid = (n + block - 1) / block;
    for (int i = 0; i < warmup; ++i) {
        vector_mul<stride><<<grid, block>>>(d_a.data(), d_b.data(), d_c.data(), n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaProfilerStart());
    nvtxRangePushA("measure");
    for (int i = 0; i < repeats; ++i) {
        vector_mul<stride><<<grid, block>>>(d_a.data(), d_b.data(), d_c.data(), n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();
    CUDA_CHECK(cudaProfilerStop());

    CUDA_CHECK(cudaMemcpy(result.data(), d_c.data(), bytes, cudaMemcpyDeviceToHost));
    int errors = 0;
    for (int i = 0; i < n; ++i) {
        const float expected = a[i] * b[i];
        if (result[i] != expected) {
            if (errors == 0)
                std::fprintf(stderr, "Mismatch i=%d: got=%g expected=%g\n",
                             i, result[i], expected);
            ++errors;
        }
    }

    std::printf("%s n=%d block=%d warmup=%d measure=%d checked=%d errors=%d\n",
                errors ? "FAIL" : "PASS", n, block, warmup, repeats, n, errors);
    return errors ? 1 : 0;
}

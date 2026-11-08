#pragma once

#include "../../kernels/vector_multiply.cuh"

// 连续与跨步读取复用 vector_mul<Stride>；其余 kernel 各自只有一种地址模式。
enum class VectorAccess { contiguous, offset, strided_read, strided_write, broadcast };

template <int Offset>
__global__ void vector_mul_offset(const float* a, const float* b, float* c, int n) {
    static_assert(Offset >= 0, "Offset must be nonnegative");
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < std::size_t(n)) {
        c[i] = a[i + Offset] * b[i + Offset];
    }
}

template <int Stride>
__global__ void vector_mul_strided_write(const float* a, const float* b, float* c, int n) {
    static_assert(Stride > 0, "Stride must be positive");
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < std::size_t(n)) {
        c[i * Stride] = a[i] * b[i];
    }
}

__global__ void vector_mul_broadcast(const float* a, const float* b, float* c, int n) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < std::size_t(n)) {
        c[i] = a[i] * b[(i / 32) * 32];
    }
}

template <VectorAccess Access, int Stride, int Offset>
inline int run_vector_access(int n, int block) {
    static_assert(Stride > 0 && Offset >= 0, "Require Stride > 0 and Offset >= 0");
    if (n <= 0 || block <= 0 || block > 1024) {
        std::fprintf(stderr, "Require n > 0, 1 <= block <= 1024\n");
        return 1;
    }
    constexpr int input_stride = Access == VectorAccess::strided_read ? Stride : 1;
    constexpr int input_offset = Access == VectorAccess::offset ? Offset : 0;
    constexpr int output_stride = Access == VectorAccess::strided_write ? Stride : 1;
    const std::size_t input_count = std::size_t(n - 1) * input_stride + input_offset + 1;
    const std::size_t output_count = std::size_t(n - 1) * output_stride + 1;
    const std::size_t input_bytes = input_count * sizeof(float);
    const std::size_t output_bytes = output_count * sizeof(float);

    std::vector<float> a(input_count, 0.0f), b(input_count, 0.0f), c(output_count);
    for (int i = 0; i < n; ++i) {
        const std::size_t input = std::size_t(i) * input_stride + input_offset;
        a[input] = float(i % 1024) * 0.125f;
        b[input] = 2.0f;
    }
    // b 的有效值都为 2，广播版仍得到与其他版本相同的 N 个结果。
    DeviceBuffer<float> d_a(input_count), d_b(input_count), d_c(output_count);
    CUDA_CHECK(cudaMemcpy(d_a.data(), a.data(), input_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b.data(), b.data(), input_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c.data(), 0xff, output_bytes));

    const int grid = (n - 1) / block + 1;
    const float average_ms = time_kernel([&] {
        if constexpr (Access == VectorAccess::contiguous) {
            vector_mul<1><<<grid, block>>>(d_a.data(), d_b.data(), d_c.data(), n);
        } else if constexpr (Access == VectorAccess::offset) {
            vector_mul_offset<Offset><<<grid, block>>>(d_a.data(), d_b.data(), d_c.data(), n);
        } else if constexpr (Access == VectorAccess::strided_read) {
            vector_mul<Stride><<<grid, block>>>(d_a.data(), d_b.data(), d_c.data(), n);
        } else if constexpr (Access == VectorAccess::strided_write) {
            vector_mul_strided_write<Stride><<<grid, block>>>(d_a.data(), d_b.data(), d_c.data(), n);
        } else if constexpr (Access == VectorAccess::broadcast) {
            vector_mul_broadcast<<<grid, block>>>(d_a.data(), d_b.data(), d_c.data(), n);
        }
    });
    CUDA_CHECK(cudaMemcpy(c.data(), d_c.data(), output_bytes, cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (int i = 0; i < n; ++i) {
        const std::size_t output = std::size_t(i) * output_stride;
        const float expected = float(i % 1024) * 0.25f;
        if (c[output] != expected) {
            if (mismatches == 0) {
                std::fprintf(stderr, "FAIL: i=%d output=%zu actual=%g expected=%g\n",
                             i, output, c[output], expected);
            }
            ++mismatches;
        }
    }

    constexpr const char* names[] = {
        "contiguous", "offset", "strided_read", "strided_write", "broadcast"
    };
    std::printf("%s: access=%s N=%d grid=%d block=%d checked=%d mismatches=%d\n",
                mismatches == 0 ? "PASS" : "FAIL", names[static_cast<int>(Access)],
                n, grid, block, n, mismatches);
    std::printf("input_stride=%d input_offset=%d output_stride=%d\n",
                input_stride, input_offset, output_stride);
    std::printf("input_elements=%zu output_elements=%zu allocated_bytes=%zu logical_bytes=%.0f\n",
                input_count, output_count, 2 * input_bytes + output_bytes, 12.0 * n);
    std::printf("launches=30 (warmup=10, timed=20), event_average_ms=%.6f\n", average_ms);
    std::printf("Event batch average excludes allocation/copies/checking; may include submission gaps.\n");
    std::printf("Event timing under NCU is NOT an independent benchmark.\n");
    if (n <= 16) {
        for (int i = 0; i < n; ++i) {
            std::printf("logical c[%d], storage[%zu] = %g\n",
                        i, std::size_t(i) * output_stride, c[std::size_t(i) * output_stride]);
        }
    }
    return mismatches == 0 ? 0 : 1;
}

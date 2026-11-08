#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t status = (call); \
    if (status != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(status)); \
        return EXIT_FAILURE; \
    } \
} while (0)

__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    constexpr int n = 257;
    constexpr int threads = 128;
    constexpr int blocks = (n + threads - 1) / threads;
    float a[n], b[n], c[n];
    for (int i = 0; i < n; ++i) { a[i] = float(i); b[i] = float(2 * i + 1); }
    float *da = nullptr, *db = nullptr, *dc = nullptr;
    CUDA_CHECK(cudaMalloc(&da, sizeof(a)));
    CUDA_CHECK(cudaMalloc(&db, sizeof(b)));
    CUDA_CHECK(cudaMalloc(&dc, sizeof(c)));
    CUDA_CHECK(cudaMemcpy(da, a, sizeof(a), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b, sizeof(b), cudaMemcpyHostToDevice));
    vector_add<<<blocks, threads>>>(da, db, dc, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(c, dc, sizeof(c), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        if (c[i] != a[i] + b[i]) {
            std::fprintf(stderr, "FAIL index=%d got=%g expected=%g\n", i, c[i], a[i] + b[i]);
            return EXIT_FAILURE;
        }
    }
    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));
    std::printf("PASS vector_add n=%d blocks=%d threads=%d checked=%d first=%g last=%g\n", n, blocks, threads, n, c[0], c[n - 1]);
    return EXIT_SUCCESS;
}

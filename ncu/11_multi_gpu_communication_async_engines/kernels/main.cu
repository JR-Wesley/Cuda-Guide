#include "../../kernels/common.cuh"
#include <cstdio>
#include <vector>

__global__ void copy_kernel(const float* input, float* output, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = input[i];
}

int main() {
    constexpr bool use_peer = true;
    constexpr int n = 16;
    constexpr int block = 256;
    constexpr int warmup = 10;
    constexpr int repeats = 20;
    int devices = 0;
    const cudaError_t status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || (status == cudaSuccess && devices < 2)) {
        std::printf("SKIPPED: need GPU0 and GPU1; found %d device(s)\n", devices);
        return 77;
    }
    CUDA_CHECK(status);
    if (use_peer) {
        int can_access = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, 1, 0));
        if (!can_access) {
            std::puts("SKIPPED: GPU1 cannot access GPU0 memory");
            return 77;
        }
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaDeviceEnablePeerAccess(0, 0));
    }

    const size_t bytes = size_t(n) * sizeof(float);
    std::vector<float> input(n), result(n);
    for (int i = 0; i < n; ++i) input[i] = float(i % 251);
    float *source0, *local1, *output1;
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc(&source0, bytes));
    CUDA_CHECK(cudaMemcpy(source0, input.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaMalloc(&local1, bytes));
    CUDA_CHECK(cudaMalloc(&output1, bytes));
    CUDA_CHECK(cudaMemcpy(local1, input.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    const float* source = use_peer ? source0 : local1;
    const int grid = (n + block - 1) / block;
    // GPU1 owns the events, executes every launch, and writes its local output.
    const float ms = time_kernel([&] {
        copy_kernel<<<grid, block>>>(source, output1, n);
        CUDA_CHECK(cudaGetLastError());
    }, warmup, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(result.data(), output1, bytes, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < n; ++i) {
        if (result[i] != input[i]) {
            if (errors == 0)
                std::fprintf(stderr, "Mismatch i=%d: got=%g expected=%g\n",
                             i, result[i], input[i]);
            ++errors;
        }
    }
    CUDA_CHECK(cudaFree(output1));
    CUDA_CHECK(cudaFree(local1));
    if (use_peer) CUDA_CHECK(cudaDeviceDisablePeerAccess(0));
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(source0));
    std::printf("%s mode=%s n=%d block=%d checked=%d errors=%d avg_ms=%.6f\n",
                errors ? "FAIL" : "PASS", use_peer ? "peer" : "local",
                n, block, n, errors, ms);
    return errors ? 1 : 0;
}

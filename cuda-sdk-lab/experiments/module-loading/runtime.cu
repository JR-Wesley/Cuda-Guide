#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "kernel.cuh"

static void check(cudaError_t result, const char* step) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "%s: %s (%d)\n", step, cudaGetErrorString(result), result);
        std::exit(1);
    }
}

int main() {
    constexpr int n = 64;
    int input[n], output[n];
    for (int i = 0; i < n; ++i) input[i] = i;

    int *device_input = nullptr, *device_output = nullptr;
    check(cudaMalloc(&device_input, sizeof(input)), "cudaMalloc input");
    check(cudaMalloc(&device_output, sizeof(output)), "cudaMalloc output");
    check(cudaMemcpy(device_input, input, sizeof(input), cudaMemcpyHostToDevice), "HtoD");

    void* args[] = {&device_input, &device_output};
    check(cudaLaunchKernel(reinterpret_cast<const void*>(add_one), dim3(1), dim3(n),
                           args, 0, nullptr), "cudaLaunchKernel");
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    check(cudaMemcpy(output, device_output, sizeof(output), cudaMemcpyDeviceToHost), "DtoH");
    for (int i = 0; i < n; ++i) {
        if (output[i] != input[i] + 1) {
            std::fprintf(stderr, "mismatch at %d: got %d, expected %d\n",
                         i, output[i], input[i] + 1);
            return 1;
        }
    }
    check(cudaFree(device_input), "cudaFree input");
    check(cudaFree(device_output), "cudaFree output");
    std::puts("api=Runtime correct=yes n=64");
}

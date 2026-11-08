#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <vector>

static void check(CUresult result, const char* step) {
    if (result != CUDA_SUCCESS) {
        const char *name = nullptr, *message = nullptr;
        cuGetErrorName(result, &name);
        cuGetErrorString(result, &message);
        std::fprintf(stderr, "%s: %s (%d): %s\n", step,
                     name ? name : "unknown", result, message ? message : "unknown");
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <image.cubin|image.ptx>\n", argv[0]);
        return 2;
    }
    std::ifstream file(argv[1], std::ios::binary);
    if (!file) {
        std::perror(argv[1]);
        return 1;
    }
    std::vector<char> image(std::istreambuf_iterator<char>{file}, {});
    if (image.empty() || file.bad()) {
        std::fprintf(stderr, "empty or unreadable image: %s\n", argv[1]);
        return 1;
    }
    image.push_back('\0'); // cuModuleLoadDataEx requires NUL-terminated PTX.

    check(cuInit(0), "cuInit");
    CUdevice device;
    check(cuDeviceGet(&device, 0), "cuDeviceGet");
    CUcontext context;
    check(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    check(cuCtxSetCurrent(context), "cuCtxSetCurrent");

    char info[8192] = {}, error[8192] = {};

    CUjit_option options[] = {CU_JIT_INFO_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                              CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
    // Buffer-size options take integer values cast to void*, not addresses of sizes.
    void* values[] = {info, reinterpret_cast<void*>(sizeof(info)),
                      error, reinterpret_cast<void*>(sizeof(error))};
    CUmodule module;
    CUresult load = cuModuleLoadDataEx(&module, image.data(), 4, options, values);
    std::printf("image=%s\nJIT info: %s\nJIT error: %s\n", argv[1], info, error);
    std::fflush(stdout);
    check(load, "cuModuleLoadDataEx");

    CUfunction function;
    check(cuModuleGetFunction(&function, module, "add_one"), "cuModuleGetFunction");
    constexpr int n = 64;
    int input[n], output[n];
    for (int i = 0; i < n; ++i) input[i] = i;
    CUdeviceptr device_input, device_output;
    check(cuMemAlloc(&device_input, sizeof(input)), "cuMemAlloc input");
    check(cuMemAlloc(&device_output, sizeof(output)), "cuMemAlloc output");
    check(cuMemcpyHtoD(device_input, input, sizeof(input)), "cuMemcpyHtoD");
    void* args[] = {&device_input, &device_output};
    check(cuLaunchKernel(function, 1, 1, 1, n, 1, 1, 0, nullptr, args, nullptr),
          "cuLaunchKernel");
    check(cuCtxSynchronize(), "cuCtxSynchronize");
    check(cuMemcpyDtoH(output, device_output, sizeof(output)), "cuMemcpyDtoH");
    for (int i = 0; i < n; ++i) {
        if (output[i] != input[i] + 1) {
            std::fprintf(stderr, "mismatch at %d: got %d, expected %d\n",
                         i, output[i], input[i] + 1);
            return 1;
        }
    }
    check(cuMemFree(device_input), "cuMemFree input");
    check(cuMemFree(device_output), "cuMemFree output");
    check(cuModuleUnload(module), "cuModuleUnload");
    check(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease");
    std::puts("api=Driver correct=yes n=64");
}

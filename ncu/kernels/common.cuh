#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t error = (call);                                            \
        if (error != cudaSuccess) {                                           \
            std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call,  \
                         cudaGetErrorString(error));                           \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// 单设备实验的显存随作用域释放；拷贝和 kernel launch 仍显式书写。
template <class T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        CUDA_CHECK(cudaMalloc(&data_, count * sizeof(T)));
    }
    ~DeviceBuffer() { CUDA_CHECK(cudaFree(data_)); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() { return data_; }
    const T* data() const { return data_; }

private:
    T* data_ = nullptr;
};

class CudaEvent {
public:
    CudaEvent() { CUDA_CHECK(cudaEventCreate(&event_)); }
    ~CudaEvent() { CUDA_CHECK(cudaEventDestroy(event_)); }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_{};
};

// launch contains just one kernel launch. Transfers and CPU checks stay outside.
template <class Launch>
inline float time_kernel(Launch launch, int warmup = 10, int repeats = 20) {
    if (warmup < 0 || repeats <= 0) {
        std::fprintf(stderr, "Require warmup >= 0 and repeats > 0\n");
        std::exit(EXIT_FAILURE);
    }
    for (int k = 0; k < warmup; ++k) {
        launch();
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaEvent start, stop;
    CUDA_CHECK(cudaEventRecord(start.get()));
    for (int k = 0; k < repeats; ++k) {
        launch();
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop.get()));
    CUDA_CHECK(cudaEventSynchronize(stop.get()));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()));
    return milliseconds / repeats;
}

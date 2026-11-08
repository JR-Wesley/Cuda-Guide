#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

inline void check(cudaError_t e, const char* call, int line) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(call) + ":" +
        std::to_string(line) + ": " + cudaGetErrorName(e) + " " + cudaGetErrorString(e));
}
#define CUDA_CHECK(x) check((x), #x, __LINE__)
inline void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
using Clock = std::chrono::steady_clock;
inline double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b-a).count();
}
inline int number(const char* s, int upper) {
    char* end = nullptr;
    long n = std::strtol(s, &end, 10);
    require(*s && !*end && n > 0 && n <= upper, "invalid integer argument");
    return static_cast<int>(n);
}
inline bool initialize() {
    int count = 0;
    auto e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess || !count) {
        std::printf("SKIP GPU unavailable: %s\n", cudaGetErrorString(e)); return false;
    }
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    std::printf("GPU=%s cc=%d.%d asyncEngineCount=%d concurrentKernels=%d\n",
                p.name, p.major, p.minor, p.asyncEngineCount, p.concurrentKernels);
    const char* blocking = std::getenv("CUDA_LAUNCH_BLOCKING");
    std::printf("CUDA_LAUNCH_BLOCKING=%s\n", blocking ? blocking : "unset");
    return true;
}

template<class T> struct Device {
    T* p = nullptr;
    explicit Device(size_t n) { CUDA_CHECK(cudaMalloc(&p, n*sizeof(T))); }
    ~Device() { if (p) cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};
template<class T> struct Host {
    T* p = nullptr; bool pinned;
    Host(size_t n, bool pin=true) : pinned(pin) {
        if (pinned) CUDA_CHECK(cudaMallocHost(&p, n*sizeof(T)));
        else p = new T[n];
    }
    ~Host() { if (pinned) cudaFreeHost(p); else delete[] p; }
    Host(const Host&) = delete;
    Host& operator=(const Host&) = delete;
};
struct Stream {
    cudaStream_t s{};
    explicit Stream(unsigned flags=cudaStreamNonBlocking) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&s, flags));
    }
    // Unlike cudaStreamDestroy alone, this RAII guard waits before cleanup.
    ~Stream() { cudaStreamSynchronize(s); cudaStreamDestroy(s); }
    Stream(const Stream&) = delete;
    operator cudaStream_t() const { return s; }
};
struct Event {
    cudaEvent_t e{};
    explicit Event(unsigned flags=cudaEventDefault) { CUDA_CHECK(cudaEventCreateWithFlags(&e, flags)); }
    ~Event() { cudaEventDestroy(e); }
    Event(const Event&) = delete;
    operator cudaEvent_t() const { return e; }
};
inline float elapsed(cudaEvent_t a, cudaEvent_t b) {
    float result; CUDA_CHECK(cudaEventElapsedTime(&result, a, b)); return result;
}
__host__ __device__ inline uint32_t transform(uint32_t x, int steps) {
    for (int i=0; i<steps; ++i) x = (x ^ (x >> 13)) * 1664525u + 1013904223u;
    return x;
}
__global__ void compute(uint32_t* data, size_t n, int steps) {
    for (size_t i=blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=gridDim.x*blockDim.x)
        data[i]=transform(data[i],steps);
}
__device__ inline unsigned long long timer_ns() {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
__global__ void delay_kernel(unsigned long long* timestamps, unsigned long long duration) {
    const auto start=timer_ns();
    auto end=start;
    while (end-start<duration) end=timer_ns();
    timestamps[0]=start; timestamps[1]=end;
}
inline void validate(const uint32_t* output, const std::vector<uint32_t>& expected) {
    for (size_t i=0; i<expected.size(); ++i)
        if(output[i]!=expected[i]) throw std::runtime_error("mismatch at element "+std::to_string(i));
}

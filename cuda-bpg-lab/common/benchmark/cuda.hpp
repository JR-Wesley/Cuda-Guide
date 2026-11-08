#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace bpg {
inline void check(cudaError_t status, const char* call, const char* file, int line) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + " " + call +
                             " -> " + cudaGetErrorName(status) + ": " + cudaGetErrorString(status));
}
#define CUDA_CHECK(call) ::bpg::check((call), #call, __FILE__, __LINE__)
inline void cleanup(cudaError_t status, const char* call) noexcept {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "cleanup failed: %s -> %s\n", call, cudaGetErrorString(status));
    std::abort(); // Never silently convert a teardown error into success.
  }
}
struct Stream {
  cudaStream_t value{};
  Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cleanup(cudaStreamDestroy(value), "cudaStreamDestroy"); }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
};
struct Event {
  cudaEvent_t value{};
  Event() { CUDA_CHECK(cudaEventCreate(&value)); }
  ~Event() { cleanup(cudaEventDestroy(value), "cudaEventDestroy"); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
};
struct DeviceBuffer {
  float* data{};
  explicit DeviceBuffer(std::size_t bytes) { if (bytes) CUDA_CHECK(cudaMalloc(&data, bytes)); }
  ~DeviceBuffer() { if (data) cleanup(cudaFree(data), "cudaFree"); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
}

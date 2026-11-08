#pragma once
#include "common/benchmark/cuda.hpp"
#include <iostream>

namespace bpg {
struct Device {
  cudaDeviceProp props{};
  int driver = 0, runtime = 0;
  std::size_t free_bytes = 0, total_bytes = 0;
  char pci[32]{};
  explicit Device(int ordinal) {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (ordinal >= count) throw std::runtime_error("requested GPU does not exist");
    CUDA_CHECK(cudaSetDevice(ordinal));
    CUDA_CHECK(cudaGetDeviceProperties(&props, ordinal));
    if (props.major < 7 || props.major > 8 || (props.major == 8 && props.minor > 9))
      throw std::runtime_error("Not Tested: only CC 7.x/8.x through SM89 is in scope");
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    CUDA_CHECK(cudaDeviceGetPCIBusId(pci, sizeof(pci), ordinal));
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    std::cout << "GPU=" << props.name << " CC=" << props.major << '.' << props.minor
              << " SMs=" << props.multiProcessorCount << " L2_bytes=" << props.l2CacheSize
              << " driver_api=" << driver << " runtime=" << runtime << " PCI=" << pci << '\n';
  }
};
}

#pragma once
#include "common/benchmark/cuda.hpp"
#include <chrono>

namespace bpg {
using Clock = std::chrono::steady_clock;
inline double elapsed_us(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::micro>(end - start).count();
}
struct Sample { double event_us, host_batch_us, host_e2e_us; };
template<class Launch>
Sample measure_batch(cudaStream_t stream, Event& start, Event& stop, int iterations, Launch launch) {
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const auto host_start = Clock::now();
  CUDA_CHECK(cudaEventRecord(start.value, stream));
  for (int i = 0; i < iterations; ++i) launch();
  CUDA_CHECK(cudaEventRecord(stop.value, stream));
  CUDA_CHECK(cudaEventSynchronize(stop.value));
  const auto host_stop = Clock::now();
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start.value, stop.value));
  if (!(ms > 0)) throw std::runtime_error("zero Event duration; increase --iterations");
  return {ms * 1000.0 / iterations, elapsed_us(host_start, host_stop) / iterations, 0};
}
}

#pragma once
#include "common/benchmark/experiment.hpp"
#include <cstdint>
#include <limits>

namespace instructions {
using Word = std::uint32_t;
static_assert(sizeof(Word) == sizeof(float));
__host__ __device__ inline std::int32_t signed_word(Word x) {
  return x <= 0x7fffffffu ? static_cast<std::int32_t>(x) : -1 - static_cast<std::int32_t>(0xffffffffu - x);
}
__host__ __device__ inline Word mix(Word x) { return (x * 1664525u + 1013904223u) ^ (x >> 13); }
inline Word seed(std::size_t i) {
  constexpr Word edges[] = {0, 1, 7, 8, 0x7fffffffu, 0x80000000u, 0xffffffffu, 0xfffffff9u, 0xfffffff8u};
  return i < 9 ? edges[i] : mix(static_cast<Word>(i) ^ 0x6a09e667u);
}
inline void cap_work(const bpg::Options& o, std::size_t work) {
  if (o.n > (1ULL << 20) || work > (1ULL << 28)) throw std::invalid_argument("n <= 2^20 and logical updates <= 2^28 per op");
}
// Allocations and CPU reference are outside every reported timer.
template<class Kernel, class Launch>
bpg::CaseResult run(const bpg::Options& o, const bpg::Device& d, const std::vector<Word>& input,
                    const std::vector<Word>& expected, bpg::CaseResult result, Kernel kernel, Launch launch) {
  const auto bytes = (input.size() + expected.size()) * sizeof(Word);
  bpg::check_allocation(o, bytes);
  if (bytes > d.free_bytes / 2) throw std::invalid_argument("exceeds 50% of free GPU memory");
  bpg::DeviceBuffer in(input.size()*4), out(expected.size()*4);
  auto* input_device = reinterpret_cast<Word*>(in.data);
  auto* output_device = reinterpret_cast<Word*>(out.data);
  bpg::Stream stream;
  std::vector<Word> actual(expected.size());
  auto prepare = [&] { CUDA_CHECK(cudaMemsetAsync(out.data, 0xff, expected.size()*4, stream.value)); };
  auto upload = [&] { CUDA_CHECK(cudaMemcpyAsync(in.data, input.data(), input.size()*4, cudaMemcpyHostToDevice, stream.value)); };
  auto download = [&] {
    CUDA_CHECK(cudaMemcpyAsync(actual.data(), out.data, expected.size()*4, cudaMemcpyDeviceToHost, stream.value));
    CUDA_CHECK(cudaStreamSynchronize(stream.value));
  };
  auto check = [&] {
    for (std::size_t i=0; i<actual.size(); ++i) if (actual[i]!=expected[i])
      throw std::runtime_error("integer mismatch at " + std::to_string(i) + " actual=" + std::to_string(actual[i]) + " expected=" + std::to_string(expected[i]));
  };
  auto invoke = [&] { launch(input_device, output_device, stream.value); CUDA_CHECK(cudaGetLastError()); };
  result.requested_bytes = bytes; result.flops = 0;
  for (const auto& kv : bpg::kernel_fields(kernel, o.block)) result.extra[kv.first]=kv.second;
  result.extra["allocation_bytes"] = std::to_string(bytes); result.extra["working_set_bytes"] = std::to_string(bytes);
  result.extra["validation_mode"] = "all uint32 output words exact";
  result.extra["e2e_scope"] = "input H2D + kernel + all output D2H + sync; allocation/reference/reset excluded";
  result.samples = bpg::collect(o, stream.value, prepare, upload, invoke, download, check);
  return result;
}
}

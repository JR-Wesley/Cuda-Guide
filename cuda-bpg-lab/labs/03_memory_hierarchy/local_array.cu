#include "common/benchmark/experiment.hpp"
#include "common/validation/compare.hpp"

namespace {
constexpr int slots = 32;
template<bool Dynamic>
__device__ __forceinline__ float update_values(float input, int rounds, int shift) {
  float x[slots];
  #pragma unroll
  for (int j = 0; j < slots; ++j) x[j] = input + j * 0.125f;
  #pragma unroll 1
  for (int r = 0; r < rounds; ++r) {
    #pragma unroll
    for (int j = 0; j < slots; ++j) {
      const int index = Dynamic ? (j + r + shift) & (slots - 1) : j;
      x[index] = fmaf(x[index], 1.0001f, 0.01f);
    }
  }
  float sum = 0;
  #pragma unroll
  for (int j = 0; j < slots; ++j) sum += x[j];
  return sum;
}
template<bool Dynamic>
__global__ void array_update(const float* __restrict__ input, float* __restrict__ output, std::size_t n, int rounds, int shift) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) output[i] = update_values<Dynamic>(input[i], rounds, shift);
}
#if BPG_HAS_MAXNREG
__global__ void __maxnreg__(32) array_limited(const float* __restrict__ input, float* __restrict__ output, std::size_t n, int rounds, int shift) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) output[i] = update_values<false>(input[i], rounds, shift);
}
#endif
std::vector<bpg::CaseResult> execute(const bpg::Options& o, const bpg::Device& d, int rounds, int shift,
                                    const std::vector<std::string>& variants) {
  const auto bytes = o.n * 4;
  if (bytes * 2 > d.free_bytes / 2) throw std::invalid_argument("exceeds 50% of free GPU memory");
  std::vector<float> input(o.n), expected(o.n), actual(o.n);
  for (std::size_t i = 0; i < o.n; ++i) {
    input[i] = (static_cast<int>(i % 257) - 128) / 32.0f;
    float x[slots];
    for (int j = 0; j < slots; ++j) x[j] = input[i] + j * 0.125f;
    for (int r = 0; r < rounds; ++r) for (auto& value : x) value = std::fma(value, 1.0001f, 0.01f);
    for (float value : x) expected[i] += value;
  }
  bpg::DeviceBuffer in(bytes), out(bytes);
  bpg::Stream stream;
  const auto grid = bpg::grid_for(o.n, o.block, d);
  auto prepare = [&] { CUDA_CHECK(cudaMemsetAsync(out.data, 0xff, bytes, stream.value)); };
  auto upload = [&] { CUDA_CHECK(cudaMemcpyAsync(in.data, input.data(), bytes, cudaMemcpyHostToDevice, stream.value)); };
  auto download = [&] {
    CUDA_CHECK(cudaMemcpyAsync(actual.data(), out.data, bytes, cudaMemcpyDeviceToHost, stream.value));
    CUDA_CHECK(cudaStreamSynchronize(stream.value));
  };
  auto check = [&] { bpg::validate(actual, expected, false, 1e-5, 1e-5); };
  std::vector<bpg::CaseResult> results;
  for (const auto& v : variants) {
    bpg::CaseResult r;
    r.variant = v; r.requested_bytes = bytes * 2; r.flops = 2.0 * o.n * slots * rounds;
    r.extra = v == "static_index" ? bpg::kernel_fields(array_update<false>, o.block) : bpg::kernel_fields(array_update<true>, o.block);
#if BPG_HAS_MAXNREG
    if (v == "register_limited") r.extra = bpg::kernel_fields(array_limited, o.block);
#endif
    r.extra["register_limit"] = v == "register_limited" ? "32" : "";
    r.extra["grid"] = std::to_string(grid); r.extra["rounds"] = std::to_string(rounds); r.extra["shift"] = std::to_string(shift);
    r.extra["slots_per_thread"] = std::to_string(slots); r.extra["working_set_bytes"] = std::to_string(bytes * 2);
    r.extra["allocation_bytes"] = std::to_string(bytes * 2);
    r.extra["working_set_scope"] = "global input/output only; addressable private backing size is not measured";
    r.extra["modeled_array_access_bytes"] = std::to_string(o.n * slots * sizeof(float) * (2ULL * rounds + 2));
    r.extra["flops_scope"] = "update FMA only (2 per value per round); excludes initialization and final reduction";
    r.extra["e2e_scope"] = "input H2D + update kernel + output D2H + sync; allocation and reset excluded";
    auto launch = [&] {
      if (v == "static_index") array_update<false><<<grid, o.block, 0, stream.value>>>(in.data, out.data, o.n, rounds, shift);
      else if (v == "dynamic_index") array_update<true><<<grid, o.block, 0, stream.value>>>(in.data, out.data, o.n, rounds, shift);
#if BPG_HAS_MAXNREG
      else array_limited<<<grid, o.block, 0, stream.value>>>(in.data, out.data, o.n, rounds, shift);
#endif
      CUDA_CHECK(cudaGetLastError());
    };
    r.samples = bpg::collect(o, stream.value, prepare, upload, launch, download, check);
    results.push_back(std::move(r));
  }
  return results;
}
}
int main(int argc, char** argv) {
  try {
    bpg::Arguments args(argc, argv, {{"n", "4096"}, {"rounds", "32"}, {"shift", "7"}});
    if (args.help) { args.usage("all|static_index|dynamic_index|register_limited (requires __maxnreg__ compiler support); 32 independent values/thread"); return 0; }
    const auto o = args.common();
    if (o.n > 65536) throw std::invalid_argument("local-array lab limits n to 65536 to bound private-state/CPU work");
    const int rounds = static_cast<int>(args.integer("rounds", 1, 256)), shift = static_cast<int>(args.integer("shift", 0, 31));
    bpg::check_allocation(o, bpg::checked_product(o.n, 8));
    std::vector<std::string> supported{"static_index", "dynamic_index"};
#if BPG_HAS_MAXNREG
    supported.push_back("register_limited");
#else
    std::cerr << "Not Tested: register_limited unavailable because the __maxnreg__ compiler probe failed\n";
#endif
    const auto variants = bpg::select_variants(o.variant, supported);
    const bpg::Device d(o.device);
    const auto results = execute(o, d, rounds, shift, variants);
    bpg::publish("local_array", o, d, results);
    return 0;
  } catch (const std::exception& e) { std::cerr << "FAILED (no valid performance CSV): " << e.what() << '\n'; return 1; }
}

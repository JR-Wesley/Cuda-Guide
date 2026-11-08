#include "common/benchmark/experiment.hpp"
#include "common/validation/compare.hpp"

namespace {
constexpr int slots = 64;
__device__ __forceinline__ void update(const float* input, float* output, std::size_t n, int rounds) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x[slots];
  #pragma unroll
  for (int j = 0; j < slots; ++j) x[j] = input[i] + j * 0.125f;
  #pragma unroll 1
  for (int r = 0; r < rounds; ++r) {
    #pragma unroll
    for (int j = 0; j < slots; ++j) x[j] = fmaf(x[j], 1.0001f, 0.01f);
  }
  float sum = 0;
  #pragma unroll
  for (int j = 0; j < slots; ++j) sum += x[j];
  output[i] = sum;
}
__global__ void budget_default(const float* input, float* output, std::size_t n, int rounds) { update(input, output, n, rounds); }
#if BPG_HAS_MAXNREG
#define BUDGET_KERNEL(B) __global__ void __maxnreg__(B) budget_##B(const float* input, float* output, std::size_t n, int rounds) { update(input, output, n, rounds); }
BUDGET_KERNEL(32)
BUDGET_KERNEL(48)
BUDGET_KERNEL(64)
#undef BUDGET_KERNEL
#endif
std::vector<bpg::CaseResult> execute(const bpg::Options& o, const bpg::Device& d, int rounds,
                                    const std::vector<std::string>& variants) {
  const auto bytes = o.n * 4;
  bpg::check_allocation(o, bytes * 2);
  if (bytes * 2 > d.free_bytes / 2) throw std::invalid_argument("exceeds 50% of free GPU memory");
  const auto grid = bpg::grid_for(o.n, o.block, d);
  std::vector<float> input(o.n), expected(o.n), actual(o.n);
  for (std::size_t i = 0; i < o.n; ++i) {
    input[i] = (static_cast<int>(i % 257) - 128) / 32.0f;
    float x[slots];
    for (int j = 0; j < slots; ++j) x[j] = input[i] + j * .125f;
    for (int r = 0; r < rounds; ++r) for (auto& value : x) value = std::fma(value, 1.0001f, .01f);
    for (float value : x) expected[i] += value;
  }
  bpg::DeviceBuffer in(bytes), out(bytes);
  bpg::Stream stream;
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
    auto kernel = budget_default;
#if BPG_HAS_MAXNREG
    if (v == "r32") kernel = budget_32;
    if (v == "r48") kernel = budget_48;
    if (v == "r64") kernel = budget_64;
#endif
    r.extra = bpg::kernel_fields(kernel, o.block);
    const int resident = std::stoi(r.extra.at("theoretical_active_blocks_per_sm"));
    const int regs = std::stoi(r.extra.at("registers"));
    int max_blocks = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_blocks, cudaDevAttrMaxBlocksPerMultiprocessor, o.device));
    const int rough = std::min({max_blocks, d.props.maxThreadsPerMultiProcessor / o.block,
                               d.props.regsPerMultiprocessor / (regs * o.block)});
    r.extra["rough_resident_blocks_bound"] = std::to_string(rough);
    r.extra["max_blocks_per_sm"] = std::to_string(max_blocks);
    r.extra["max_threads_per_sm"] = std::to_string(d.props.maxThreadsPerMultiProcessor);
    r.extra["registers_per_sm"] = std::to_string(d.props.regsPerMultiprocessor);
    r.extra["theoretical_occupancy"] = bpg::decimal(double(resident * ((o.block + 31) / 32)) / (d.props.maxThreadsPerMultiProcessor / 32));
    r.extra["grid"] = std::to_string(grid); r.extra["rounds"] = std::to_string(rounds);
    r.extra["slots_per_thread"] = std::to_string(slots);
    r.extra["register_limit"] = v == "unlimited" ? "" : v.substr(1);
    r.extra["allocation_bytes"] = std::to_string(bytes * 2);
    r.extra["working_set_bytes"] = std::to_string(bytes * 2);
    r.extra["working_set_scope"] = "global input/output; private spill backing not measured";
    r.extra["flops_scope"] = "update FMA only; excludes initialization and final sum";
    r.extra["e2e_scope"] = "input H2D + update kernel + output D2H + sync; allocation/reset excluded";
    auto launch = [&] {
      kernel<<<grid, o.block, 0, stream.value>>>(in.data, out.data, o.n, rounds);
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
    bpg::Arguments args(argc, argv, {{"n", "65536"}, {"rounds", "32"}});
    if (args.help) { args.usage("all|unlimited|r32|r48|r64; fixed 64 accumulators/thread; budget variants require __maxnreg__"); return 0; }
    const auto o = args.common();
    if (o.n > 65536) throw std::invalid_argument("n <= 65536 to bound private-state/CPU work");
    const int rounds = static_cast<int>(args.integer("rounds", 1, 256));
    std::vector<std::string> supported{"unlimited"};
#if BPG_HAS_MAXNREG
    supported.insert(supported.end(), {"r32", "r48", "r64"});
#else
    std::cerr << "Not Tested: register budget variants unavailable (compiler probe failed)\n";
#endif
    const auto variants = bpg::select_variants(o.variant, supported);
    const bpg::Device d(o.device);
    const auto results = execute(o, d, rounds, variants);
    bpg::publish("register_pressure", o, d, results);
    return 0;
  } catch (const std::exception& e) { std::cerr << "FAILED (no valid performance CSV): " << e.what() << '\n'; return 1; }
}

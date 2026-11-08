#include "common/benchmark/experiment.hpp"
#include "common/validation/compare.hpp"

namespace {
__global__ void sum_working_set(const float* __restrict__ input, float* __restrict__ out,
                               std::size_t threads, std::size_t mask, int steps) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= threads) return;
  float sum = 0;
  #pragma unroll 1
  for (int step = 0; step < steps; ++step) sum += input[(i + static_cast<std::size_t>(step) * threads) & mask];
  out[i] = sum;
}
std::vector<bpg::CaseResult> execute(const bpg::Options& o, const bpg::Device& d, std::size_t working_set, int steps) {
  const auto input_bytes = working_set * 4, output_bytes = o.n * 4;
  if (input_bytes + output_bytes > d.free_bytes / 2) throw std::invalid_argument("exceeds 50% of free GPU memory");
  std::vector<float> input(working_set), expected(o.n), actual(o.n);
  std::uint32_t seed = 20261023;
  for (auto& value : input) { seed = seed * 1664525u + 1013904223u; value = static_cast<int>((seed >> 8) % 17) - 8; }
  for (std::size_t i = 0; i < o.n; ++i)
    for (int step = 0; step < steps; ++step) expected[i] += input[(i + static_cast<std::size_t>(step) * o.n) & (working_set - 1)];
  // Integer-valued floats and steps<=65536 keep all partial sums exactly representable.
  bpg::DeviceBuffer in(input_bytes), out(output_bytes);
  bpg::Stream stream;
  const auto grid = bpg::grid_for(o.n, o.block, d);
  auto prepare = [&] { CUDA_CHECK(cudaMemsetAsync(out.data, 0xff, output_bytes, stream.value)); };
  auto upload = [&] { CUDA_CHECK(cudaMemcpyAsync(in.data, input.data(), input_bytes, cudaMemcpyHostToDevice, stream.value)); };
  auto launch = [&] {
    sum_working_set<<<grid, o.block, 0, stream.value>>>(in.data, out.data, o.n, working_set - 1, steps);
    CUDA_CHECK(cudaGetLastError());
  };
  auto download = [&] {
    CUDA_CHECK(cudaMemcpyAsync(actual.data(), out.data, output_bytes, cudaMemcpyDeviceToHost, stream.value));
    CUDA_CHECK(cudaStreamSynchronize(stream.value));
  };
  auto check = [&] { bpg::validate(actual, expected, true); };
  bpg::CaseResult r;
  r.variant = "scalar"; r.requested_bytes = o.n * (steps + 1ULL) * 4; r.flops = static_cast<double>(o.n) * steps;
  r.extra = bpg::kernel_fields(sum_working_set, o.block);
  r.extra["grid"] = std::to_string(grid); r.extra["steps"] = std::to_string(steps);
  r.extra["input_working_set_elements"] = std::to_string(working_set);
  r.extra["working_set_bytes"] = std::to_string(input_bytes + output_bytes);
  r.extra["allocation_bytes"] = std::to_string(input_bytes + output_bytes);
  r.extra["input_loads_per_op"] = std::to_string(o.n * steps);
  r.extra["e2e_scope"] = "whole input H2D + sum kernel + output D2H + sync; transfer bytes change with working set";
  r.samples = bpg::collect(o, stream.value, prepare, upload, launch, download, check);
  return {std::move(r)};
}
}
int main(int argc, char** argv) {
  try {
    bpg::Arguments args(argc, argv, {{"n", "65536"}, {"working-set", "1048576"}, {"steps", "256"}});
    if (args.help) { args.usage("scalar|all; --n is output threads; --working-set is a power-of-two float count"); return 0; }
    const auto o = args.common();
    const auto w = args.integer("working-set", 32, 1ULL << 28);
    const int steps = static_cast<int>(args.integer("steps", 1, 65536));
    bpg::select_variants(o.variant, {"scalar"});
    const auto reads = bpg::checked_product(o.n, steps);
    if ((w & (w - 1)) || w > reads) throw std::invalid_argument("working-set must be a power of two and fully covered by n*steps reads");
    if (reads > (1ULL << 28)) throw std::invalid_argument("at most 2^28 reads/op to bound default host validation work");
    bpg::check_allocation(o, bpg::checked_product(bpg::checked_sum(w, o.n), 4));
    const bpg::Device d(o.device);
    const auto results = execute(o, d, w, steps);
    bpg::publish("cache_working_set", o, d, results);
    return 0;
  } catch (const std::exception& e) { std::cerr << "FAILED (no valid performance CSV): " << e.what() << '\n'; return 1; }
}

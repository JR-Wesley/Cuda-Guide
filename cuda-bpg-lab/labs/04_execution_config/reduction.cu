#include "common/benchmark/experiment.hpp"
#include <cmath>
#include <limits>

namespace {
// Every block executes the barriers; tail threads contribute zero.
template<int Method>
__global__ void reduce_pass(const float* __restrict__ input, float* __restrict__ output,
                            std::size_t n, int items) {
  extern __shared__ float shared[];
  const unsigned t = threadIdx.x;
  const auto base = static_cast<std::size_t>(blockIdx.x) * blockDim.x * items + t;
  float x = 0;
  for (int j = 0; j < items; ++j) {
    const auto i = base + static_cast<std::size_t>(j) * blockDim.x;
    if (i < n) x += input[i];
  }
  if constexpr (Method == 2) {
    // block is a power of two >=32, so every physical warp is full.
    for (int offset = 16; offset > 0; offset /= 2) x += __shfl_down_sync(0xffffffffu, x, offset);
    if ((t & 31) == 0) shared[t / 32] = x;
    __syncthreads();
    if (t < 32) {
      x = t < blockDim.x / 32 ? shared[t] : 0.0f;
      for (int offset = 16; offset > 0; offset /= 2) x += __shfl_down_sync(0xffffffffu, x, offset);
      if (t == 0) output[blockIdx.x] = x;
    }
  } else {
    shared[t] = x;
    __syncthreads();
    if constexpr (Method == 0 || Method == 3) {
      for (unsigned stride = 1; stride < blockDim.x; stride *= 2) {
        const bool active = Method == 3 ? (t & (2 * stride - 1)) == 0 : t % (2 * stride) == 0;
        if (active) shared[t] += shared[t + stride];
        __syncthreads();
      }
    } else {
      for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (t < stride) shared[t] += shared[t + stride];
        __syncthreads();
      }
    }
    if (t == 0) output[blockIdx.x] = shared[0];
  }
}
struct Plan { std::vector<std::size_t> sizes; std::size_t bytes = 0; };
Plan plan_for(std::size_t n, int block, int items) {
  Plan p;
  do {
    const auto next = (n + block * items - 1) / (block * items);
    p.sizes.push_back(n); p.bytes += (n + next) * sizeof(float); n = next;
  } while (n > 1);
  return p;
}
std::vector<bpg::CaseResult> execute(const bpg::Options& o, const bpg::Device& d, int items,
                                    const std::string& pattern, const std::vector<std::string>& variants) {
  const auto plan = plan_for(o.n, o.block, items);
  const auto partials = (o.n + o.block * items - 1) / (o.block * items);
  const auto allocation = (o.n + 2 * partials) * sizeof(float);
  bpg::check_allocation(o, allocation);
  if (allocation > d.free_bytes / 2) throw std::invalid_argument("exceeds 50% of free GPU memory");
  bpg::grid_for(o.n, o.block, d); // device block/grid limit before launches
  std::vector<float> input(o.n);
  double reference = 0, abs_sum = 0;
  for (std::size_t i = 0; i < o.n; ++i) {
    const float magnitude = 1.0f + static_cast<float>((i * 17) % 257) / 257.0f;
    input[i] = pattern == "ones" ? 1.0f : (pattern == "cancellation" && (i & 1) ? -magnitude : magnitude);
    reference += input[i]; abs_sum += std::abs(static_cast<double>(input[i]));
  }
  // Conservative depth bound includes serial per-thread sums and both shuffle trees.
  const int log_block = static_cast<int>(std::log2(o.block));
  const int depth = static_cast<int>(plan.sizes.size()) * (items + log_block + 5);
  const double u = std::numeric_limits<float>::epsilon() / 2;
  const double gamma = depth * u / (1 - depth * u);
  const double tolerance = pattern == "ones" ? 0.0 : std::max(1e-6, gamma * abs_sum);
  bpg::DeviceBuffer in(o.n * 4), a(partials * 4), b(partials * 4);
  bpg::Stream stream;
  float actual = 0;
  float* final_output = plan.sizes.size() % 2 ? a.data : b.data;
  auto prepare = [&] {
    CUDA_CHECK(cudaMemsetAsync(a.data, 0xff, partials * 4, stream.value));
    CUDA_CHECK(cudaMemsetAsync(b.data, 0xff, partials * 4, stream.value));
  };
  auto upload = [&] { CUDA_CHECK(cudaMemcpyAsync(in.data, input.data(), o.n * 4, cudaMemcpyHostToDevice, stream.value)); };
  auto download = [&] {
    CUDA_CHECK(cudaMemcpyAsync(&actual, final_output, 4, cudaMemcpyDeviceToHost, stream.value));
    CUDA_CHECK(cudaStreamSynchronize(stream.value));
  };
  double max_error = 0;
  auto check = [&] {
    const double error = std::abs(actual - reference);
    if (!std::isfinite(actual) || error > tolerance)
      throw std::runtime_error("reduction mismatch: actual=" + bpg::decimal(actual) + " double_reference=" + bpg::decimal(reference) + " abs_error=" + bpg::decimal(error) + " tolerance=" + bpg::decimal(tolerance));
    max_error = std::max(max_error, error);
  };
  std::vector<bpg::CaseResult> results;
  for (const auto& v : variants) {
    const auto shared = (v == "warp" ? o.block / 32 : o.block) * sizeof(float);
    bpg::CaseResult r;
    r.variant = v; r.requested_bytes = plan.bytes; r.flops = o.n - 1;
    r.extra = v == "interleaved_mask" ? bpg::kernel_fields(reduce_pass<3>, o.block, shared) : v == "interleaved" ? bpg::kernel_fields(reduce_pass<0>, o.block, shared) :
              v == "sequential" ? bpg::kernel_fields(reduce_pass<1>, o.block, shared) : bpg::kernel_fields(reduce_pass<2>, o.block, shared);
    const int blocks = std::stoi(r.extra.at("theoretical_active_blocks_per_sm"));
    r.extra["theoretical_occupancy"] = bpg::decimal(double(blocks * o.block) / d.props.maxThreadsPerMultiProcessor);
    r.extra["max_threads_per_sm"] = std::to_string(d.props.maxThreadsPerMultiProcessor);
    r.extra["registers_per_sm"] = std::to_string(d.props.regsPerMultiprocessor);
    r.extra["items_per_thread"] = std::to_string(items); r.extra["pattern"] = pattern;
    r.extra["passes"] = std::to_string(plan.sizes.size()); r.extra["first_grid"] = std::to_string(partials);
    r.extra["allocation_bytes"] = std::to_string(allocation);
    r.extra["working_set_bytes"] = std::to_string(allocation);
    r.extra["flops_scope"] = "mathematical n-1 adds; excludes zero padding and redundant shuffle-lane operations";
    r.extra["event_scope"] = "complete multi-pass device reduction including all launches and gaps";
    r.extra["e2e_scope"] = "input H2D + all reduction passes + one scalar D2H + sync; allocation/reset excluded";
    r.extra["cpu_reference"] = bpg::decimal(reference); r.extra["sum_abs_input"] = bpg::decimal(abs_sum);
    r.extra["error_depth_bound"] = std::to_string(depth); r.extra["absolute_tolerance"] = bpg::decimal(tolerance);
    auto launch = [&] {
      const float* src = in.data;
      for (std::size_t pass = 0; pass < plan.sizes.size(); ++pass) {
        const auto n = plan.sizes[pass];
        const auto grid = static_cast<unsigned>((n + o.block * items - 1) / (o.block * items));
        float* dst = pass % 2 ? b.data : a.data;
        if (v == "interleaved_mask") reduce_pass<3><<<grid, o.block, shared, stream.value>>>(src, dst, n, items);
        else if (v == "interleaved") reduce_pass<0><<<grid, o.block, shared, stream.value>>>(src, dst, n, items);
        else if (v == "sequential") reduce_pass<1><<<grid, o.block, shared, stream.value>>>(src, dst, n, items);
        else reduce_pass<2><<<grid, o.block, shared, stream.value>>>(src, dst, n, items);
        CUDA_CHECK(cudaGetLastError()); src = dst;
      }
    };
    max_error = 0;
    r.samples = bpg::collect(o, stream.value, prepare, upload, launch, download, check);
    r.extra["max_absolute_error"] = bpg::decimal(max_error);
    results.push_back(std::move(r));
  }
  return results;
}
}
int main(int argc, char** argv) {
  try {
    bpg::Arguments args(argc, argv, {{"items", "4"}, {"pattern", "mixed"}});
    if (args.help) { args.usage("all|interleaved|interleaved_mask|sequential|warp; block power of two 32..1024; pattern mixed|ones|cancellation"); return 0; }
    const auto o = args.common();
    if (o.block < 32 || (o.block & (o.block - 1))) throw std::invalid_argument("block must be a power of two in 32..1024");
    if (o.n > (1ULL << 24)) throw std::invalid_argument("n <= 2^24 for bounded CPU work and exact ones validation");
    const int items = static_cast<int>(args.integer("items", 1, 32));
    const auto pattern = args.text("pattern");
    if (pattern != "mixed" && pattern != "ones" && pattern != "cancellation") throw std::invalid_argument("invalid pattern");
    const auto variants = bpg::select_variants(o.variant, {"interleaved", "interleaved_mask", "sequential", "warp"});
    const bpg::Device d(o.device);
    const auto results = execute(o, d, items, pattern, variants);
    bpg::publish("reduction", o, d, results);
    return 0;
  } catch (const std::exception& e) { std::cerr << "FAILED (no valid performance CSV): " << e.what() << '\n'; return 1; }
}

#include "common/benchmark/experiment.hpp"
#include "common/validation/compare.hpp"
#include "labs/02_global_memory/model.hpp"

namespace {
__global__ void scalar_copy(const float* __restrict__ in, float* __restrict__ out, std::size_t n,
                            std::size_t read_stride, std::size_t write_stride) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i * write_stride] = in[i * read_stride];
}
template<int Width> struct Vector;
template<> struct Vector<2> { using type = float2; };
template<> struct Vector<4> { using type = float4; };
template<int Width>
__global__ void vector_copy(const float* __restrict__ in, float* __restrict__ out, std::size_t n) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  using V = typename Vector<Width>::type;
  if (i < n / Width) reinterpret_cast<V*>(out)[i] = reinterpret_cast<const V*>(in)[i];
  // Separate scalar tail. Even N<Width launches one block and copies only the legal elements.
  if (i < n % Width) out[n / Width * Width + i] = in[n / Width * Width + i];
}
struct Layout {
  std::size_t read_stride, write_stride, read_offset, write_offset, input_count, output_count;
};
Layout layout(const bpg::Options& o, std::size_t stride, std::size_t offset, const std::string& direction) {
  const bool read = direction != "scatter", write = direction != "gather";
  Layout l{read ? stride : 1, write ? stride : 1, read ? offset : 0, write ? offset : 0, 0, 0};
  l.input_count = bpg::checked_sum(bpg::checked_sum(bpg::checked_product(o.n - 1, l.read_stride), l.read_offset), 1);
  l.output_count = bpg::checked_sum(bpg::checked_sum(bpg::checked_product(o.n - 1, l.write_stride), l.write_offset), 1);
  return l;
}
std::vector<bpg::CaseResult> execute(const bpg::Options& o, const bpg::Device& d, const Layout& l,
                                    std::size_t stride, std::size_t offset, const std::string& direction,
                                    const std::vector<std::string>& variants) {
  constexpr float guard = -777.25f; // Finite sentinel also validates every untouched output gap/prefix.
  const auto input_bytes = l.input_count * sizeof(float), output_bytes = l.output_count * sizeof(float);
  if (input_bytes + output_bytes > d.free_bytes / 2) throw std::invalid_argument("exceeds 50% of free GPU memory");
  std::vector<float> input(l.input_count, 123.25f), initial(l.output_count, guard), reference = initial, actual(l.output_count);
  std::uint32_t state = 20261023;
  for (std::size_t i = 0; i < o.n; ++i) {
    state = state * 1664525u + 1013904223u;
    const float value = (static_cast<int>((state >> 8) % 20001) - 10000) / 1000.0f;
    input[l.read_offset + i * l.read_stride] = value;
    reference[l.write_offset + i * l.write_stride] = value;
  }
  bpg::DeviceBuffer in(input_bytes), out(output_bytes);
  bpg::Stream stream;
  auto upload = [&] { CUDA_CHECK(cudaMemcpyAsync(in.data, input.data(), input_bytes, cudaMemcpyHostToDevice, stream.value)); };
  auto prepare = [&] { CUDA_CHECK(cudaMemcpyAsync(out.data, initial.data(), output_bytes, cudaMemcpyHostToDevice, stream.value)); };
  auto download = [&] {
    CUDA_CHECK(cudaMemcpyAsync(actual.data(), out.data, output_bytes, cudaMemcpyDeviceToHost, stream.value));
    CUDA_CHECK(cudaStreamSynchronize(stream.value));
  };
  auto check = [&] { bpg::validate(actual, reference, true); };
  std::vector<bpg::CaseResult> results;
  for (const auto& name : variants) {
    const int width = name == "float4" ? 4 : name == "float2" ? 2 : 1;
    const auto grid = bpg::grid_for(std::max<std::size_t>({1, o.n / width, o.n % width}), o.block, d);
    // Include tail threads even with block=1 and fewer than four complete vectors.
    bpg::CaseResult result;
    result.variant = name; result.requested_bytes = o.n * 8;
    result.extra = name == "scalar" ? bpg::kernel_fields(scalar_copy, o.block)
                 : name == "float2" ? bpg::kernel_fields(vector_copy<2>, o.block)
                                     : bpg::kernel_fields(vector_copy<4>, o.block);
    auto& f = result.extra;
    f["grid"] = std::to_string(grid); f["stride"] = std::to_string(stride); f["offset_elements"] = std::to_string(offset);
    f["direction"] = direction; f["vector_width"] = std::to_string(width); f["tail_elements"] = std::to_string(o.n % width);
    f["working_set_bytes"] = std::to_string(o.n * 8); // Unique bytes requested, gaps excluded.
    f["allocation_bytes"] = std::to_string(input_bytes + output_bytes);
    f["input_address_span_bytes"] = std::to_string(input_bytes - l.read_offset * 4);
    f["output_address_span_bytes"] = std::to_string(output_bytes - l.write_offset * 4);
    f["model_read_sectors_first_instruction"] = std::to_string(bpg::first_warp_sectors(o.n, l.read_stride, l.read_offset, width));
    f["model_write_sectors_first_instruction"] = std::to_string(bpg::first_warp_sectors(o.n, l.write_stride, l.write_offset, width));
    f["e2e_transfer_bytes"] = std::to_string(input_bytes + output_bytes);
    f["e2e_scope"] = "full allocated input H2D + copy kernel + full allocated output D2H + sync; output reset excluded";
    auto launch = [&] {
      if (name == "scalar") scalar_copy<<<grid, o.block, 0, stream.value>>>(in.data + l.read_offset, out.data + l.write_offset, o.n, l.read_stride, l.write_stride);
      else if (name == "float2") vector_copy<2><<<grid, o.block, 0, stream.value>>>(in.data + l.read_offset, out.data + l.write_offset, o.n);
      else vector_copy<4><<<grid, o.block, 0, stream.value>>>(in.data + l.read_offset, out.data + l.write_offset, o.n);
      CUDA_CHECK(cudaGetLastError());
    };
    result.samples = bpg::collect(o, stream.value, prepare, upload, launch, download, check);
    results.push_back(std::move(result));
  }
  return results;
}
}
int main(int argc, char** argv) {
  try {
    bpg::Arguments args(argc, argv, {{"stride", "1"}, {"offset", "0"}, {"direction", "gather"}});
    if (args.help) { args.usage("all|scalar|float2|float4; direction=gather|scatter|both; offset in float elements"); return 0; }
    const auto o = args.common();
    const auto stride = args.integer("stride", 1, 1024), offset = args.integer("offset", 0, 1024);
    const auto direction = args.text("direction");
    if (direction != "gather" && direction != "scatter" && direction != "both") throw std::invalid_argument("invalid direction");
    const auto variants = bpg::select_variants(o.variant, {"scalar", "float2", "float4"});
    for (const auto& v : variants) {
      const unsigned width = v == "float4" ? 4 : v == "float2" ? 2 : 1;
      if (width > 1 && (stride != 1 || offset % width))
        throw std::invalid_argument("vector copy requires stride=1 and offset divisible by vector width; use --variant scalar for arbitrary layout");
    }
    const auto l = layout(o, stride, offset, direction);
    bpg::check_allocation(o, bpg::checked_product(bpg::checked_sum(l.input_count, l.output_count), sizeof(float)));
    const bpg::Device device(o.device);
    const auto results = execute(o, device, l, stride, offset, direction, variants);
    bpg::publish("global_copy", o, device, results);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAILED (no valid performance CSV): " << e.what() << '\n'; return 1;
  }
}

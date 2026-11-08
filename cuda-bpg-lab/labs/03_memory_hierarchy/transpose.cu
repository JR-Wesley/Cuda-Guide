#include "common/benchmark/experiment.hpp"
#include "common/validation/compare.hpp"

namespace {
__global__ void transpose_naive(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < static_cast<std::size_t>(rows) * cols) out[(i % cols) * rows + i / cols] = in[i];
}
template<int Padding>
__global__ void transpose_tiled(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
  __shared__ float tile[32][32 + Padding];
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;
  #pragma unroll
  for (int j = 0; j < 32; j += 8)
    if (x < cols && y + j < rows) tile[threadIdx.y + j][threadIdx.x] = in[static_cast<std::size_t>(y + j) * cols + x];
  // Every thread participates, including those outside a partial edge tile.
  __syncthreads();
  x = blockIdx.y * 32 + threadIdx.x;
  y = blockIdx.x * 32 + threadIdx.y;
  #pragma unroll
  for (int j = 0; j < 32; j += 8)
    if (x < rows && y + j < cols) out[static_cast<std::size_t>(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
}
std::vector<bpg::CaseResult> execute(const bpg::Options& o, const bpg::Device& d, int rows, int cols,
                                    const std::vector<std::string>& variants) {
  const auto bytes = o.n * sizeof(float);
  if (bytes * 2 > d.free_bytes / 2) throw std::invalid_argument("exceeds 50% of free GPU memory");
  std::vector<float> input(o.n), expected(o.n), actual(o.n);
  std::uint32_t seed = 20261023;
  for (std::size_t i = 0; i < o.n; ++i) {
    seed = seed * 1664525u + 1013904223u;
    input[i] = static_cast<int>((seed >> 8) % 20001) / 128.0f;
    expected[(i % cols) * rows + i / cols] = input[i];
  }
  bpg::DeviceBuffer in(bytes), out(bytes);
  bpg::Stream stream;
  const auto naive_grid = bpg::grid_for(o.n, o.block, d);
  const dim3 grid((cols + 31) / 32, (rows + 31) / 32), block(32, 8);
  if (grid.y > static_cast<unsigned>(d.props.maxGridSize[1])) throw std::invalid_argument("tile grid exceeds device limit");
  auto prepare = [&] { CUDA_CHECK(cudaMemsetAsync(out.data, 0xff, bytes, stream.value)); };
  auto upload = [&] { CUDA_CHECK(cudaMemcpyAsync(in.data, input.data(), bytes, cudaMemcpyHostToDevice, stream.value)); };
  auto download = [&] {
    CUDA_CHECK(cudaMemcpyAsync(actual.data(), out.data, bytes, cudaMemcpyDeviceToHost, stream.value));
    CUDA_CHECK(cudaStreamSynchronize(stream.value));
  };
  auto check = [&] { bpg::validate(actual, expected, true); };
  std::vector<bpg::CaseResult> results;
  for (const auto& v : variants) {
    bpg::CaseResult r;
    r.variant = v; r.requested_bytes = bytes * 2;
    r.extra = v == "naive" ? bpg::kernel_fields(transpose_naive, o.block)
            : v == "tiled" ? bpg::kernel_fields(transpose_tiled<0>, 256)
                           : bpg::kernel_fields(transpose_tiled<1>, 256);
    r.extra["rows"] = std::to_string(rows); r.extra["cols"] = std::to_string(cols);
    r.extra["working_set_bytes"] = std::to_string(bytes * 2); r.extra["allocation_bytes"] = std::to_string(bytes * 2);
    r.extra["grid_x"] = std::to_string(v == "naive" ? naive_grid : grid.x);
    r.extra["grid_y"] = std::to_string(v == "naive" ? 1 : grid.y);
    r.extra["block_x"] = std::to_string(v == "naive" ? 256 : 32); r.extra["block_y"] = v == "naive" ? "1" : "8";
    r.extra["shared_pitch_words"] = v == "naive" ? "" : v == "tiled" ? "32" : "33";
    r.extra["model_distinct_banks_transposed_full_warp"] = v == "naive" ? "" : v == "tiled" ? "1" : "32";
    r.extra["e2e_scope"] = "input H2D + transpose + output D2H + sync; reset and allocation excluded";
    auto launch = [&] {
      if (v == "naive") transpose_naive<<<naive_grid, o.block, 0, stream.value>>>(in.data, out.data, rows, cols);
      else if (v == "tiled") transpose_tiled<0><<<grid, block, 0, stream.value>>>(in.data, out.data, rows, cols);
      else transpose_tiled<1><<<grid, block, 0, stream.value>>>(in.data, out.data, rows, cols);
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
    bpg::Arguments args(argc, argv, {{"rows", "1024"}, {"cols", "1024"}}, {"n"});
    if (args.help) { args.usage("all|naive|tiled|padded; fixed 256 threads (tiled: 32x8)"); return 0; }
    auto o = args.common();
    const int rows = static_cast<int>(args.integer("rows", 1, 65536)), cols = static_cast<int>(args.integer("cols", 1, 65536));
    o.n = bpg::checked_product(rows, cols);
    if (o.block != 256) throw std::invalid_argument("transpose fixes --block=256; tiled layout is 32x8");
    bpg::check_allocation(o, bpg::checked_product(o.n, 8));
    const auto variants = bpg::select_variants(o.variant, {"naive", "tiled", "padded"});
    const bpg::Device d(o.device);
    const auto results = execute(o, d, rows, cols, variants);
    bpg::publish("transpose", o, d, results);
    return 0;
  } catch (const std::exception& e) { std::cerr << "FAILED (no valid performance CSV): " << e.what() << '\n'; return 1; }
}

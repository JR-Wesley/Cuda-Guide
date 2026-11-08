#pragma once
#include "bpg_build_info.hpp"
#include "common/benchmark/options.hpp"
#include "common/benchmark/stats.hpp"
#include "common/benchmark/timing.hpp"
#include "common/device_info/device.hpp"
#include "common/validation/compare.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

namespace bpg {
// Each instantiation contains exactly one operation; Copy eliminates b at compile time.
template<bool Copy>
__global__ void scalar_kernel(const float* __restrict__ a, const float* __restrict__ b,
                              float* __restrict__ out, std::size_t n) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    if constexpr (Copy) out[i] = a[i];
    else out[i] = a[i] + b[i];
  }
}
template<bool Copy>
__global__ void grid_stride_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                   float* __restrict__ out, std::size_t n) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += stride) {
    if constexpr (Copy) out[i] = a[i];
    else out[i] = a[i] + b[i];
  }
}
struct Result {
  std::string variant;
  unsigned grid = 0;
  int registers = 0, shared_bytes = 0, active_blocks_per_sm = 0;
  std::vector<Sample> samples;
};

template<bool Copy>
std::vector<Result> execute(const Options& o, const Device& device) {
  const std::size_t bytes = o.n * sizeof(float);
  if (bytes * (Copy ? 2 : 3) > device.free_bytes / 2)
    throw std::runtime_error("requested buffers exceed 50% of currently free device memory");
  if (o.block > device.props.maxThreadsPerBlock || o.block > device.props.maxThreadsDim[0])
    throw std::invalid_argument("block exceeds this device's limit");
  const std::size_t full_grid = (o.n + o.block - 1) / o.block;
  if (full_grid > static_cast<std::size_t>(device.props.maxGridSize[0]))
    throw std::invalid_argument("grid exceeds device limit");

  std::vector<float> a(o.n), b(Copy ? 0 : o.n), reference(o.n), actual(o.n);
  std::uint32_t state = 20261023;
  auto next = [&]() {
    state = state * 1664525u + 1013904223u;
    return (static_cast<int>((state >> 8) % 20001) - 10000) / 1000.0f;
  };
  for (std::size_t i = 0; i < o.n; ++i) {
    a[i] = next();
    if constexpr (Copy) reference[i] = a[i];
    else { b[i] = next(); reference[i] = static_cast<float>(static_cast<double>(a[i]) + b[i]); }
  }
  DeviceBuffer da(bytes), db(Copy ? 0 : bytes), dout(bytes);
  Stream stream;
  Event start, stop;
  auto upload = [&]() {
    CUDA_CHECK(cudaMemcpyAsync(da.data, a.data(), bytes, cudaMemcpyHostToDevice, stream.value));
    if constexpr (!Copy)
      CUDA_CHECK(cudaMemcpyAsync(db.data, b.data(), bytes, cudaMemcpyHostToDevice, stream.value));
  };
  auto download = [&]() {
    CUDA_CHECK(cudaMemcpyAsync(actual.data(), dout.data, bytes, cudaMemcpyDeviceToHost, stream.value));
    CUDA_CHECK(cudaStreamSynchronize(stream.value));
  };
  auto poison = [&]() {
    CUDA_CHECK(cudaMemsetAsync(dout.data, 0xff, bytes, stream.value)); // NaNs expose unwritten output.
    CUDA_CHECK(cudaStreamSynchronize(stream.value));
  };
  upload();
  CUDA_CHECK(cudaStreamSynchronize(stream.value));
  std::vector<std::string> variants = {"scalar", "grid_stride"};
  if constexpr (Copy) variants.push_back("runtime");
  if (o.variant != "all") variants = {o.variant};
  std::vector<Result> results;
  for (const auto& variant : variants) {
    Result r;
    r.variant = variant;
    r.grid = variant == "grid_stride"
      ? static_cast<unsigned>(std::min(full_grid, static_cast<std::size_t>(device.props.multiProcessorCount) * o.grid_multiplier))
      : static_cast<unsigned>(full_grid);
    if (variant != "runtime") {
      auto kernel = variant == "scalar" ? scalar_kernel<Copy> : grid_stride_kernel<Copy>;
      cudaFuncAttributes attributes{};
      CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
      if (o.block > attributes.maxThreadsPerBlock) throw std::invalid_argument("block exceeds kernel limit");
      r.registers = attributes.numRegs;
      r.shared_bytes = static_cast<int>(attributes.sharedSizeBytes);
      CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r.active_blocks_per_sm, kernel, o.block, 0));
    } else { r.grid = 0; }
    auto launch = [&]() {
      if (variant == "scalar") {
        scalar_kernel<Copy><<<r.grid, o.block, 0, stream.value>>>(da.data, db.data, dout.data, o.n);
        CUDA_CHECK(cudaGetLastError());
      } else if (variant == "grid_stride") {
        grid_stride_kernel<Copy><<<r.grid, o.block, 0, stream.value>>>(da.data, db.data, dout.data, o.n);
        CUDA_CHECK(cudaGetLastError());
      } else {
        CUDA_CHECK(cudaMemcpyAsync(dout.data, da.data, bytes, cudaMemcpyDeviceToDevice, stream.value));
      }
    };
    // Establish correctness before collecting any timings.
    poison(); launch(); download(); validate(actual, reference, Copy);
    for (int i = 0; i < o.warmup; ++i) launch();
    CUDA_CHECK(cudaStreamSynchronize(stream.value));
    for (int repeat = 0; repeat < o.repeats; ++repeat) {
      poison();
      // Rewarm after poison; consistent warm/reused buffers, not cold-cache traffic.
      for (int i = 0; i < o.warmup; ++i) launch();
      CUDA_CHECK(cudaStreamSynchronize(stream.value));
      auto sample = measure_batch(stream.value, start, stop, o.iterations, launch);
      download(); validate(actual, reference, Copy);
      // One independent end-to-end operation on existing allocations and pageable host buffers.
      poison();
      const auto begin = Clock::now();
      upload(); launch(); download();
      sample.host_e2e_us = elapsed_us(begin, Clock::now());
      validate(actual, reference, Copy);
      r.samples.push_back(sample);
    }
    results.push_back(std::move(r));
  }
  return results; // RAII cleanup is checked before results are published.
}

inline std::string quoted(const std::string& value) {
  std::string result = "\"";
  for (char c : value) { if (c == '"') result += '"'; result += c; }
  return result + '"';
}
inline std::string number(double x) { std::ostringstream s; s << std::setprecision(12) << x; return s.str(); }
template<bool Copy>
void report(const Options& o, const Device& d, const std::vector<Result>& results) {
  const std::size_t requested = o.n * sizeof(float) * (Copy ? 2 : 3);
  const char* context_env = std::getenv("BPG_MEASUREMENT_CONTEXT");
  const std::string context = context_env ? context_env : "standalone";
  std::ostringstream csv;
  csv << "experiment,variant,statistic,sample,n,block,grid,grid_multiplier,warmup,repeats,iterations,requested_bytes,working_set_bytes,event_span_us_per_op,host_batch_us_per_op,host_e2e_us,effective_GBps_at_event_stat,gpu,cc,sm_count,l2_bytes,device_total_bytes,device_free_bytes,device_ordinal,pci,driver_api_version,runtime_version,compiler,build_arch,build_flags,registers,static_shared_bytes,theoretical_active_blocks_per_sm,measurement_context,validation\n";
  for (const auto& r : results) {
    std::vector<double> events, hosts, e2es;
    for (const auto& s : r.samples) { events.push_back(s.event_us); hosts.push_back(s.host_batch_us); e2es.push_back(s.host_e2e_us); }
    const auto es = statistics(events), hs = statistics(hosts), xs = statistics(e2es);
    auto row = [&](const std::string& statistic, const std::string& index, Sample s) {
      const std::vector<std::string> fields = {
        Copy ? "memory_copy" : "vector_add", r.variant, statistic, index, std::to_string(o.n),
        r.variant == "runtime" ? "" : std::to_string(o.block), r.variant == "runtime" ? "" : std::to_string(r.grid),
        std::to_string(o.grid_multiplier), std::to_string(o.warmup), std::to_string(o.repeats), std::to_string(o.iterations),
        std::to_string(requested), std::to_string(requested), number(s.event_us), number(s.host_batch_us), number(s.host_e2e_us),
        number(requested / (s.event_us * 1000.0)), d.props.name,
        std::to_string(d.props.major) + "." + std::to_string(d.props.minor), std::to_string(d.props.multiProcessorCount),
        std::to_string(d.props.l2CacheSize), std::to_string(d.total_bytes), std::to_string(d.free_bytes), std::to_string(o.device),
        d.pci, std::to_string(d.driver), std::to_string(d.runtime), build_compiler, build_arch, build_flags,
        r.variant == "runtime" ? "" : std::to_string(r.registers), r.variant == "runtime" ? "" : std::to_string(r.shared_bytes),
        r.variant == "runtime" ? "" : std::to_string(r.active_blocks_per_sm), context, "PASS"};
      for (std::size_t i = 0; i < fields.size(); ++i) { if (i) csv << ','; csv << quoted(fields[i]); }
      csv << '\n';
    };
    for (std::size_t i = 0; i < r.samples.size(); ++i) row("raw", std::to_string(i), r.samples[i]);
    row("min", "", {es.min, hs.min, xs.min});
    row("median", "", {es.median, hs.median, xs.median});
    row("p95", "", {es.p95, hs.p95, xs.p95});
    std::cout << r.variant << " PASS event_span_us/op median=" << es.median << " min=" << es.min << " p95=" << es.p95
              << " effective_GB/s=" << requested / (es.median * 1000.0)
              << " host_batch_us/op=" << hs.median << " host_e2e_us=" << xs.median << '\n';
  }
  if (!o.csv.empty()) {
    const auto temp = o.csv + ".tmp";
    if (std::filesystem::exists(o.csv) || std::filesystem::exists(temp)) throw std::runtime_error("refusing to overwrite CSV or temporary file");
    std::ofstream out(temp);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << csv.str(); out.close();
    std::filesystem::rename(temp, o.csv);
  }
}

template<bool Copy>
int run(int argc, char** argv) {
  try {
    const auto o = parse_options(argc, argv, Copy);
    if (o.help) {
      std::cout << "Options: --n 1048576 --block 256 --variant all|scalar|grid_stride" << (Copy ? "|runtime" : "")
                << " --warmup 5 --repeats 21 --iterations 100 --grid-multiplier 4 --device 0 --max-mib 256 --csv PATH\n"
                << "Event span / iterations includes dispatch gaps. Host E2E includes H2D+operation+D2H+sync, excludes allocations.\n";
      return 0;
    }
    if (!o.csv.empty() && (std::filesystem::exists(o.csv) || std::filesystem::exists(o.csv + ".tmp")))
      throw std::invalid_argument("refusing to overwrite CSV");
    const Device device(o.device);
    std::cout << "compiler=" << build_compiler << " arch=" << build_arch << " flags=" << build_flags << '\n';
    const auto results = execute<Copy>(o, device);
    report<Copy>(o, device, results);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAILED (no valid performance CSV): " << e.what() << '\n';
    return 1;
  }
}
}

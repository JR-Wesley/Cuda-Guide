#pragma once
#include "bpg_build_info.hpp"
#include "common/benchmark/cli.hpp"
#include "common/benchmark/stats.hpp"
#include "common/benchmark/timing.hpp"
#include "common/device_info/device.hpp"
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>

namespace bpg {
using Fields = std::map<std::string, std::string>;
inline std::string decimal(double x) { std::ostringstream s; s << std::setprecision(12) << x; return s.str(); }
inline std::string csv_quote(const std::string& value) {
  std::string s = "\"";
  for (char c : value) { if (c == '"') s += '"'; s += c; }
  return s + '"';
}
struct CaseResult {
  std::string variant;
  std::size_t requested_bytes = 0;
  double flops = 0;
  Fields extra;
  std::vector<Sample> samples;
};
inline Fields environment_fields(const Options& o, const Device& d) {
  const auto* context = std::getenv("BPG_MEASUREMENT_CONTEXT");
  return {{"n", std::to_string(o.n)}, {"block", std::to_string(o.block)},
          {"warmup", std::to_string(o.warmup)}, {"repeats", std::to_string(o.repeats)},
          {"iterations", std::to_string(o.iterations)}, {"device_ordinal", std::to_string(o.device)},
          {"gpu", d.props.name}, {"cc", std::to_string(d.props.major) + "." + std::to_string(d.props.minor)},
          {"sm_count", std::to_string(d.props.multiProcessorCount)}, {"l2_bytes", std::to_string(d.props.l2CacheSize)},
          {"device_total_bytes", std::to_string(d.total_bytes)}, {"device_free_bytes", std::to_string(d.free_bytes)},
          {"pci", d.pci}, {"driver_api_version", std::to_string(d.driver)}, {"runtime_version", std::to_string(d.runtime)},
          {"compiler", build_compiler}, {"build_arch", build_arch}, {"build_flags", build_flags},
          {"measurement_context", context ? context : "standalone"}, {"validation", "PASS"}};
}
template<class Kernel>
Fields kernel_fields(Kernel kernel, int block, std::size_t dynamic_shared = 0) {
  cudaFuncAttributes a{};
  CUDA_CHECK(cudaFuncGetAttributes(&a, kernel));
  if (block > a.maxThreadsPerBlock) throw std::invalid_argument("block exceeds kernel limit");
  int blocks = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, block, dynamic_shared));
  return {{"registers", std::to_string(a.numRegs)}, {"static_shared_bytes", std::to_string(a.sharedSizeBytes)},
          {"local_bytes_per_thread", std::to_string(a.localSizeBytes)}, {"dynamic_shared_bytes", std::to_string(dynamic_shared)},
          {"theoretical_active_blocks_per_sm", std::to_string(blocks)}};
}
inline unsigned grid_for(std::size_t work, int block, const Device& d) {
  if (block > d.props.maxThreadsPerBlock || block > d.props.maxThreadsDim[0])
    throw std::invalid_argument("block exceeds device limit");
  const auto grid = (work + block - 1) / block;
  if (!grid || grid > static_cast<std::size_t>(d.props.maxGridSize[0]))
    throw std::invalid_argument("invalid grid dimension");
  return static_cast<unsigned>(grid);
}
// download() completes the stream; check() only inspects the already-copied host output.
// E2E has precisely upload+launch+download; allocations and check are excluded.
template<class Prepare, class Upload, class Launch, class Download, class Check>
std::vector<Sample> collect(const Options& o, cudaStream_t stream, Prepare prepare, Upload upload,
                            Launch launch, Download download, Check check) {
  Event start, stop;
  upload(); prepare(); launch(); download(); check();
  std::vector<Sample> samples;
  for (int r = 0; r < o.repeats; ++r) {
    prepare();
    for (int w = 0; w < o.warmup; ++w) launch();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto s = measure_batch(stream, start, stop, o.iterations, launch);
    download(); check();
    prepare(); CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto begin = Clock::now();
    upload(); launch(); download();
    s.host_e2e_us = elapsed_us(begin, Clock::now());
    check(); samples.push_back(s);
  }
  return samples;
}
inline void publish(const std::string& experiment, const Options& o, const Device& d,
                    const std::vector<CaseResult>& results) {
  std::vector<Fields> rows;
  for (const auto& r : results) {
    std::vector<double> event, host, e2e;
    for (const auto& s : r.samples) { event.push_back(s.event_us); host.push_back(s.host_batch_us); e2e.push_back(s.host_e2e_us); }
    const auto es = statistics(event), hs = statistics(host), xs = statistics(e2e);
    auto append = [&](const std::string& statistic, const std::string& sample, Sample time) {
      auto f = environment_fields(o, d);
      for (const auto& kv : r.extra) f[kv.first] = kv.second;
      f["experiment"] = experiment; f["variant"] = r.variant; f["statistic"] = statistic; f["sample"] = sample;
      f["requested_bytes"] = std::to_string(r.requested_bytes); f["flops"] = decimal(r.flops);
      f["event_span_us_per_op"] = decimal(time.event_us); f["host_batch_us_per_op"] = decimal(time.host_batch_us);
      f["host_e2e_us"] = decimal(time.host_e2e_us);
      f["effective_GBps_at_event_stat"] = decimal(r.requested_bytes / (time.event_us * 1000.0));
      f["GFLOPs_at_event_stat"] = decimal(r.flops / (time.event_us * 1000.0));
      rows.push_back(std::move(f));
    };
    for (std::size_t i = 0; i < r.samples.size(); ++i) append("raw", std::to_string(i), r.samples[i]);
    append("min", "", {es.min, hs.min, xs.min});
    append("median", "", {es.median, hs.median, xs.median});
    append("p95", "", {es.p95, hs.p95, xs.p95});
    std::cout << r.variant << " PASS event_span_us/op median=" << es.median << " min=" << es.min << " p95=" << es.p95
              << " effective_GB/s=" << r.requested_bytes / (es.median * 1000.0)
              << " host_e2e_us=" << xs.median << '\n';
  }
  if (o.csv.empty()) return;
  if (std::filesystem::exists(o.csv) || std::filesystem::exists(o.csv + ".tmp"))
    throw std::runtime_error("refusing to overwrite CSV");
  std::set<std::string> keys;
  for (const auto& row : rows) for (const auto& kv : row) keys.insert(kv.first);
  std::ofstream out(o.csv + ".tmp");
  out.exceptions(std::ios::failbit | std::ios::badbit);
  bool first = true;
  for (const auto& key : keys) { if (!first) out << ','; first = false; out << csv_quote(key); }
  out << '\n';
  for (auto& row : rows) {
    first = true;
    for (const auto& key : keys) { if (!first) out << ','; first = false; out << csv_quote(row[key]); }
    out << '\n';
  }
  out.close();
  std::filesystem::rename(o.csv + ".tmp", o.csv);
}
}

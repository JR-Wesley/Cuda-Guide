#pragma once
#include <charconv>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace bpg {
struct Options {
  std::size_t n = 1 << 20;
  int block = 256, warmup = 5, repeats = 21, iterations = 100, device = 0;
  int grid_multiplier = 4;
  std::size_t max_mib = 256;
  std::string variant = "all", csv;
  bool help = false;
};
inline std::uint64_t unsigned_number(const std::string& text) {
  std::uint64_t result = 0;
  auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
  if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
    throw std::invalid_argument("invalid unsigned integer: " + text);
  return result;
}
inline Options parse_options(int argc, char** argv, bool copy) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help") { o.help = true; continue; }
    if (i + 1 == argc) throw std::invalid_argument("missing value for " + key);
    const std::string value = argv[++i];
    if (key == "--csv") { o.csv = value; continue; }
    if (key == "--variant") { o.variant = value; continue; }
    const auto n = unsigned_number(value);
    if (key == "--n") {
      if (n == 0 || n > std::numeric_limits<std::size_t>::max() / 16)
        throw std::invalid_argument("n must be positive and byte counts must fit size_t");
      o.n = static_cast<std::size_t>(n);
    } else if (key == "--max-mib") {
      if (n == 0 || n > 1024) throw std::invalid_argument("max-mib must be 1..1024 (hard safety cap)");
      o.max_mib = static_cast<std::size_t>(n);
    } else {
      if (n > 1000000) throw std::invalid_argument("numeric option exceeds 1000000");
      const int x = static_cast<int>(n);
      if (key == "--block") o.block = x;
      else if (key == "--warmup") o.warmup = x;
      else if (key == "--repeats") o.repeats = x;
      else if (key == "--iterations") o.iterations = x;
      else if (key == "--device") o.device = x;
      else if (key == "--grid-multiplier") o.grid_multiplier = x;
      else throw std::invalid_argument("unknown option: " + key);
    }
  }
  if (o.block < 1 || o.block > 1024 || o.warmup < 1 || o.repeats < 1 ||
      o.iterations < 1 || o.grid_multiplier < 1)
    throw std::invalid_argument("block: 1..1024; warmup/repeats/iterations/grid-multiplier: positive");
  if (o.variant != "all" && o.variant != "scalar" && o.variant != "grid_stride" &&
      !(copy && o.variant == "runtime")) throw std::invalid_argument("unsupported variant: " + o.variant);
  const std::size_t buffers = copy ? 2 : 3;
  if (o.n * sizeof(float) * buffers > o.max_mib * 1024 * 1024)
    throw std::invalid_argument("device working set exceeds --max-mib; reduce --n");
  return o;
}
}

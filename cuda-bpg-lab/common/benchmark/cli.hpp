#pragma once
#include "common/benchmark/options.hpp"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <map>
#include <vector>

namespace bpg {
// Later stages add named parameters without changing the Stage 1 CLI contract.
class Arguments {
 public:
  std::map<std::string, std::string> values;
  bool help = false;
  Arguments(int argc, char** argv, std::map<std::string, std::string> extras = {}, std::vector<std::string> omitted = {})
      : values{{"n", "1048576"}, {"block", "256"}, {"warmup", "5"}, {"repeats", "21"},
               {"iterations", "100"}, {"device", "0"}, {"max-mib", "256"}, {"variant", "all"}, {"csv", ""}} {
    for (const auto& kv : extras) values[kv.first] = kv.second;
    for (const auto& key : omitted) values.erase(key);
    for (int i = 1; i < argc; ++i) {
      const std::string key = argv[i];
      if (key == "--help") { help = true; continue; }
      if (key.size() < 3 || key.substr(0, 2) != "--" || !values.count(key.substr(2)))
        throw std::invalid_argument("unknown option: " + key);
      if (++i == argc) throw std::invalid_argument("missing value for " + key);
      values[key.substr(2)] = argv[i];
    }
  }
  const std::string& text(const std::string& key) const { return values.at(key); }
  std::size_t integer(const std::string& key, std::size_t lo, std::size_t hi) const {
    const auto n = unsigned_number(text(key));
    if (n < lo || n > hi) throw std::invalid_argument("--" + key + " must be " + std::to_string(lo) + ".." + std::to_string(hi));
    return static_cast<std::size_t>(n);
  }
  void usage(const std::string& variants) const {
    std::cout << "Variants: " << variants << '\n';
    for (const auto& kv : values) std::cout << "--" << kv.first << " " << kv.second << '\n';
  }
  Options common() const {
    Options o;
    o.n = values.count("n") ? integer("n", 1, std::numeric_limits<std::size_t>::max() / 16) : 1;
    o.block = static_cast<int>(integer("block", 1, 1024));
    o.warmup = static_cast<int>(integer("warmup", 1, 1000000));
    o.repeats = static_cast<int>(integer("repeats", 1, 1000000));
    o.iterations = static_cast<int>(integer("iterations", 1, 1000000));
    o.device = static_cast<int>(integer("device", 0, 1024));
    o.max_mib = integer("max-mib", 1, 1024);
    o.variant = text("variant"); o.csv = text("csv");
    if (!o.csv.empty() && (std::filesystem::exists(o.csv) || std::filesystem::exists(o.csv + ".tmp")))
      throw std::invalid_argument("refusing to overwrite CSV");
    return o;
  }
};
inline std::vector<std::string> select_variants(const std::string& requested, std::vector<std::string> supported) {
  if (requested == "all") return supported;
  if (std::find(supported.begin(), supported.end(), requested) == supported.end())
    throw std::invalid_argument("unsupported variant: " + requested);
  return {requested};
}
inline std::size_t checked_product(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b) throw std::invalid_argument("size overflow");
  return a * b;
}
inline std::size_t checked_sum(std::size_t a, std::size_t b) {
  if (a > std::numeric_limits<std::size_t>::max() - b) throw std::invalid_argument("size overflow");
  return a + b;
}
inline void check_allocation(const Options& o, std::size_t bytes) {
  if (bytes > o.max_mib * 1024 * 1024) throw std::invalid_argument("device allocation exceeds --max-mib");
}
}

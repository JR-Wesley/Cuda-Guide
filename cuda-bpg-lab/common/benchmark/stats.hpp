#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace bpg {
struct Stats { double min, median, p95; };
inline Stats statistics(std::vector<double> values) {
  if (values.empty()) throw std::invalid_argument("empty samples");
  for (double x : values)
    if (!std::isfinite(x) || x <= 0) throw std::invalid_argument("nonpositive/nonfinite timing");
  std::sort(values.begin(), values.end());
  const auto n = values.size();
  const double median = n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2;
  // Nearest-rank P95; for fewer than 20 samples this is the maximum.
  return {values.front(), median, values[static_cast<std::size_t>(std::ceil(0.95 * n)) - 1]};
}
}

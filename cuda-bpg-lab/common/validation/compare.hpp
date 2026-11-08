#pragma once
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace bpg {
inline void validate(const std::vector<float>& actual, const std::vector<float>& expected,
                     bool exact, double atol = 1e-6, double rtol = 1e-6) {
  if (actual.size() != expected.size()) throw std::runtime_error("validation size mismatch");
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double a = actual[i], e = expected[i];
    const bool equal = exact ? std::memcmp(&actual[i], &expected[i], sizeof(float)) == 0
                             : std::abs(a - e) <= atol + rtol * std::abs(e);
    if (!std::isfinite(a) || !std::isfinite(e) || !equal) {
      std::ostringstream error;
      error << "validation failed at " << i << ": actual=" << a << " reference=" << e;
      throw std::runtime_error(error.str());
    }
  }
}
}

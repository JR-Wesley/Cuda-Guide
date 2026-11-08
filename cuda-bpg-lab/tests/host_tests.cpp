#include "common/benchmark/options.hpp"
#include "common/benchmark/stats.hpp"
#include "common/validation/compare.hpp"
#include "common/benchmark/cli.hpp"
#include "labs/02_global_memory/model.hpp"
#include <iostream>
#include <limits>

void require(bool condition) { if (!condition) throw std::runtime_error("test failed"); }
template<class F> void must_fail(F f) {
  bool failed = false;
  try { f(); } catch (const std::exception&) { failed = true; }
  require(failed);
}
int main() {
  require(bpg::first_warp_sectors(32, 1, 0, 1) == 4);
  require(bpg::first_warp_sectors(32, 1, 1, 1) == 5);
  require(bpg::first_warp_sectors(32, 2, 0, 1) == 8);
  require(bpg::first_warp_sectors(32, 8, 0, 1) == 32);
  require(bpg::first_warp_sectors(32, 32, 0, 1) == 32);
  require(bpg::first_warp_sectors(64, 1, 0, 2) == 8);
  require(bpg::first_warp_sectors(128, 1, 0, 4) == 16);
  require(bpg::first_warp_sectors(1, 1, 0, 4) == 0);
  must_fail([] { bpg::checked_sum(std::numeric_limits<std::size_t>::max(), 1); });
  must_fail([] { bpg::checked_product(std::numeric_limits<std::size_t>::max(), 2); });
  const auto s = bpg::statistics({4, 1, 3, 2});
  require(s.min == 1 && s.median == 2.5 && s.p95 == 4);
  require(bpg::statistics({1, 8, 3}).median == 3);
  std::vector<double> twenty; for (int i = 1; i <= 20; ++i) twenty.push_back(i);
  require(bpg::statistics(twenty).p95 == 19);
  must_fail([] { bpg::statistics({}); });
  must_fail([] { bpg::statistics({0}); });
  must_fail([] { bpg::statistics({std::numeric_limits<double>::quiet_NaN()}); });
  bpg::validate({1.0000001f, 0}, {1, 0}, false);
  bpg::validate({1, -2}, {1, -2}, true);
  must_fail([] { bpg::validate({0.01f}, {0}, false); });
  must_fail([] { bpg::validate({std::numeric_limits<float>::quiet_NaN()}, {0}, false); });
  must_fail([] { bpg::validate({std::numeric_limits<float>::infinity()}, {1}, false); });
  must_fail([] { bpg::validate({1.0000001f}, {1}, true); });
  require(bpg::unsigned_number("1048576") == 1048576);
  for (const auto* text : {"-1", "1x", "", "18446744073709551616"})
    must_fail([&] { bpg::unsigned_number(text); });
  std::cout << "host statistics / validation / parsing PASS\n";
}

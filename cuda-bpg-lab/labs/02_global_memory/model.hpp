#pragma once
#include <algorithm>
#include <cstddef>
#include <set>

namespace bpg {
// Addresses relative to a cudaMalloc-aligned base. Count one warp's complete vector accesses.
// A scalar remainder in a vector kernel is a separate instruction and excluded here.
inline std::size_t first_warp_sectors(std::size_t n, std::size_t stride, std::size_t offset, std::size_t width) {
  const auto lanes = std::min<std::size_t>(32, n / width);
  std::set<std::size_t> sectors;
  for (std::size_t lane = 0; lane < lanes; ++lane) {
    const auto start = (offset + lane * width * stride) * 4;
    for (std::size_t b = 0; b < width * 4; ++b) sectors.insert((start + b) / 32);
  }
  return sectors.size();
}
}

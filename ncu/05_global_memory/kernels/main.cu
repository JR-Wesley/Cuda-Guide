#include "vector_access.cuh"

constexpr auto access = VectorAccess::contiguous;
constexpr int n = 16;       // 先手算；n=32 观察完整 warp，扩大规模时固定其他条件。
constexpr int block = 256;
constexpr int stride = 8;   // 仅 strided_read / strided_write 使用；可对照 1 / 2 / 4 / 8。
constexpr int offset = 1;   // 仅 offset 使用，单位是 float：1 表示偏移 4B。

int main() {
    return run_vector_access<access, stride, offset>(n, block);
}

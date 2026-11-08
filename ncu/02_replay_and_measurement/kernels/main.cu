#include "../../kernels/vector_multiply.cuh"

// 先核对 N=16，再改为 1048576 并重新编译；测量策略对照期间固定 N。
constexpr int n = 16;
constexpr int block = 256;
constexpr int stride = 1;  // 本课不改变布局，只改变采集条件。

int main() {
    return run_vector<stride>(n, block);
}

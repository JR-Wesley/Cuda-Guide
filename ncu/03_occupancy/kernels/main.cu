#include "../../kernels/vector_multiply.cuh"
#include "../../kernels/fma.cuh"

constexpr bool use_fma = false;  // false: A 乘法；true: D 寄存器/依赖链补例。
constexpr int n = 16;           // 先核对；性能阶段改为 262144，然后固定。
constexpr int block = 256;      // A 的单变量：128 / 256 / 512。
constexpr int stride = 1;       // A 始终连续读取。
constexpr int chains = 1;       // D 的单变量：1 / 4 / 8；不同时改变 block。
constexpr int total_fmas_per_thread = 1024;  // D 总 FMA 数固定，不是每条链的次数。

int main() {
    if constexpr (use_fma) {
        return run_fma<chains>(n, block, total_fmas_per_thread);
    } else {
        return run_vector<stride>(n, block);
    }
}

#include "../../kernels/fma.cuh"

int main() {
    constexpr int n = 1 << 14;
    constexpr int block = 256;
    constexpr int chains = 1;
    constexpr int total_fmas_per_thread = 1024;

    return run_fma<chains>(n, block, total_fmas_per_thread);
}

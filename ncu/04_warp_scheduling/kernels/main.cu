#include "../../kernels/fma.cuh"
#include "../../kernels/reduction.cuh"

int main() {
    constexpr bool use_reduction = false;  // D by default; true selects C.
    constexpr int n = 1 << 14;
    constexpr int block = 256;
    constexpr int chains = 1;
    constexpr int total_fmas_per_thread = 1024;
    constexpr bool shuffle = false;

    if constexpr (use_reduction) {
        return run_reduction(n, shuffle);
    } else {
        return run_fma<chains>(n, block, total_fmas_per_thread);
    }
}

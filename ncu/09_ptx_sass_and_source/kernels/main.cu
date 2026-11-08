#include "../../kernels/fma.cuh"
#include "../../kernels/matrix_transpose.cuh"
#include "../../kernels/reduction.cuh"

int main() {
    constexpr int example = 0;  // 0: D (FMA), 1: B (transpose), 2: C (reduction).
    constexpr int n = 1 << 14;
    constexpr int block = 256;
    constexpr int chains = 1;
    constexpr int total_fmas_per_thread = 1024;
    constexpr int matrix_n = 1024;
    constexpr int transpose_variant = 0;
    constexpr bool shuffle = false;
    static_assert(example >= 0 && example <= 2, "example must be 0, 1 or 2");

    if constexpr (example == 0) {
        return run_fma<chains>(n, block, total_fmas_per_thread);
    } else if constexpr (example == 1) {
        return run_transpose(matrix_n, transpose_variant);
    } else {
        return run_reduction(n, shuffle);
    }
}

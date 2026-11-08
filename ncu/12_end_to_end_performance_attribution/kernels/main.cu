#include "../../kernels/matrix_transpose.cuh"
#include "../../kernels/reduction.cuh"
#include "../../kernels/fma.cuh"

int main() {
    constexpr int example = 0;  // 0: B transpose, 1: C reduction, 2: D FMA
    constexpr int transpose_n = 65;
    constexpr int transpose_variant = 0;
    constexpr int reduction_n = 65;
    constexpr bool use_shuffle = false;
    constexpr int fma_n = 65;
    constexpr int fma_block = 256;
    constexpr int fma_chains = 1;
    constexpr int total_fmas_per_thread = 256;

    static_assert(example >= 0 && example <= 2, "example must be 0, 1, or 2");
    if constexpr (example == 0) {
        return run_transpose(transpose_n, transpose_variant);
    } else if constexpr (example == 1) {
        return run_reduction(reduction_n, use_shuffle);
    } else {

        return run_fma<fma_chains>(fma_n, fma_block, total_fmas_per_thread);
    }
}

#include "../../kernels/vector_multiply.cuh"
#include "../../kernels/fma.cuh"
#include "../../kernels/matrix_transpose.cuh"

int main() {
    constexpr int example = 0;  // 0: A (vector), 1: D (FMA), 2: B (transpose).
    constexpr int n = 1 << 14;
    constexpr int block = 256;
    constexpr int stride = 1;
    constexpr int chains = 1;
    constexpr int total_fmas_per_thread = 1024;
    constexpr int matrix_n = 1024;  // Matrix side length, not element count.
    constexpr int transpose_variant = 0;  // 0: naive, 1: tiled, 2: padded.
    static_assert(example >= 0 && example <= 2, "example must be 0, 1 or 2");

    if constexpr (example == 0) {
        return run_vector<stride>(n, block);
    } else if constexpr (example == 1) {
        return run_fma<chains>(n, block, total_fmas_per_thread);
    } else {
        return run_transpose(matrix_n, transpose_variant);
    }
}

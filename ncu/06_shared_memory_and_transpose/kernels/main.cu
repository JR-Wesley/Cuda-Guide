#include "../../kernels/matrix_transpose.cuh"

int main() {
    constexpr int n = 65;
    constexpr int variant = 0;  // 0: naive, 1: tiled, 2: padded
    return run_transpose(n, variant);
}

#include "../../kernels/vector_multiply.cuh"

#include <cerrno>
#include <cstdio>
#include <cstdlib>

// 先用 16 个输出手算；性能阶段可通过唯一可选参数 N 改为 1048576。
constexpr int default_n = 16;
constexpr int block = 256;  // 每块线程数；本课固定。
constexpr int stride = 1;  // 连续输入；本课固定。

int main(int argc, char** argv) {
    int n = default_n;
    if (argc > 2) {
        std::fprintf(stderr, "Usage: %s [N: 1..67108864]\n", argv[0]);
        return 1;
    }
    if (argc == 2) {
        char* end = nullptr;
        errno = 0;
        const long value = std::strtol(argv[1], &end, 10);
        if (errno == ERANGE || end == argv[1] || *end != '\0' ||
            value < 1 || value > (1L << 26)) {
            std::fprintf(stderr, "N must be an integer in [1, 67108864]\n");
            return 1;
        }
        n = static_cast<int>(value);
    }
    return run_vector<stride>(n, block);
}

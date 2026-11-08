#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>

// 定义矩阵类型（使用二维数组）
using Matrix = std::vector<std::vector<float>>;

// 初始化矩阵为随机值
void init_matrix(Matrix& mat, const int rows, const int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    mat.resize(rows, std::vector<float>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat[i][j] = dist(gen);
        }
    }
}

// 方法1：内积法（i-j-k）——经典三重循环
void matmul_ijk(Matrix& C, const Matrix& A, const Matrix& B, const int M, const int N, const int K) {
    // 初始化 C 为 0
    C.assign(M, std::vector<float>(N, 0.0f));

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                C[m][n] += A[m][k] * B[k][n];
            }
        }
    }
}

// 方法2：外积法（k-i-j）——K 维度在外层
void matmul_kij(Matrix& C, const Matrix& A, const Matrix& B, const int M, const int N, const int K) {
    // 初始化 C 为 0
    C.assign(M, std::vector<float>(N, 0.0f));

    for (int k = 0; k < K; ++k) {
        for (int m = 0; m < M; ++m) {
            float a_mk = A[m][k];  // 提取到寄存器
            for (int n = 0; n < N; ++n) {
                C[m][n] += a_mk * B[k][n];
            }
        }
    }
}

// 高精度计时执行函数
template<typename Func>
double benchmark(Func&& func, const int warmup = 1, const int iters = 5) {
    // 预热
    for (int i = 0; i < warmup; ++i) {
        func();
    }

    // 正式计时
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration_us.count() / (double)iters;  // 平均微秒
}

int main() {
    // 测试不同的矩阵大小
    std::vector<int> sizes = {64, 128, 256, 512};  // M=N=K
    constexpr int warmup = 1;
    constexpr int iters = 5;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Matrix Multiplication Performance Comparison\n";
    std::cout << "--------------------------------------------\n";
    std::cout << "Size\tIJK (GFLOPS)\tKIJ (GFLOPS)\tSpeedup\n";

    for (int N : sizes) {
        int M = N, K = N;
        Matrix A, B, C1, C2;

        init_matrix(A, M, K);
        init_matrix(B, K, N);

        // 测试 IJK（内积法）
        double time_ijk = benchmark([&]() { matmul_ijk(C1, A, B, M, N, K); },
                                 warmup, iters);

        // 测试 KIJ（外积法）
        double time_kij = benchmark([&]() { matmul_kij(C2, A, B, M, N, K); },
                                 warmup, iters);

        // 计算 GFLOPS：2*M*N*K / (time in seconds) / 1e9
        double flops = 2.0 * M * N * K;
        double gflops_ijk = flops / (time_ijk * 1e-6) / 1e9;
        double gflops_kij = flops / (time_kij * 1e-6) / 1e9;

        double speedup = time_ijk / time_kij;

        std::cout << N << "\t" 
                  << gflops_ijk << "\t\t" 
                  << gflops_kij << "\t\t" 
                  << speedup << "x\n";
    }

    return 0;
}
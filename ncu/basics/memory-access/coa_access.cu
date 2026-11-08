#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>

struct golden_data_t {
  std::vector<float> A;
  float* A_d;
  std::vector<float> B;
  float* B_d;
  std::vector<float> C;
  float* C_d;
  int N;

  golden_data_t(int N) : N(N), A(N), B(N), C(N), A_d(nullptr), B_d(nullptr), C_d(nullptr) {}

  ~golden_data_t() {        
    if (A_d)
      cudaFree(A_d);
    if (B_d)
      cudaFree(B_d);
    if (C_d)
      cudaFree(C_d);
  }

  void init() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < N; ++i) {
        A[i] = dis(gen);
        B[i] = dis(gen);
    }
    std::transform(A.begin(), A.end(), B.begin(), C.begin(), std::plus<float>());

    cudaMalloc(&A_d, N * sizeof(float));
    cudaMalloc(&B_d, N * sizeof(float));
    cudaMalloc(&C_d, N * sizeof(float));
    cudaMemcpy(A_d, A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B.data(), N * sizeof(float), cudaMemcpyHostToDevice);
  }

  bool verify_result() {
    std::vector<float> C_host(N);
    cudaMemcpy(C_host.data(), C_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    return std::equal(C_host.begin(), C_host.end(), C.begin());
  }
};

__global__ void vec_add(const float* A, const float* B, float* C, const int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

__global__ void vec_add_offset(const float* A, const float* B, float* C, const int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

__global__ void vec_add_interleave(const float* A, const float* B, float* C, const int N) {
    const int i = blockIdx.x * blockDim.x + (threadIdx.x ^ 0x1);
    if (i < N) {
        C[i] = A[i] + B[i];
    }   
}

__global__ void vec_add_broadcast(const float* A, const float* B, float* C, const int N) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (i < N) {
        C[i] = A[i] + B[i];
    }   
}

__global__ void vec_add_multi(const float* A, const float* B, float* C, const int N) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i < N) {
        C[i] = A[i] + B[i];
    }   
}

int main(void) {
    constexpr int N = 32 * 1024 * 1024;
    constexpr int blockSize = 256;
    constexpr int numBlocks = (N + blockSize - 1) / blockSize;

    golden_data_t data(N);
    data.init();

    vec_add<<<numBlocks, blockSize>>>(data.A_d, data.B_d, data.C_d, N);
    cudaDeviceSynchronize();
    bool is_correct = data.verify_result();
    std::cout << "is_correct: " << is_correct << std::endl;

    for (const auto& func: {vec_add, vec_add_offset, vec_add_interleave, vec_add_broadcast, vec_add_multi}) {
        func<<<numBlocks, blockSize>>>(data.A_d, data.B_d, data.C_d, N);
        cudaDeviceSynchronize();
    }

    return 0;
}
#include "gemm.h"
#include <iostream>
#include <vector>

int main(int argc, char *argv[]) {
  if (argc == 4) {
    std::cout << atoi(argv[1]) << atoi(argv[2]) << atoi(argv[3]) << "\n";
  }
  if (!torch::cuda::is_available()) {
    std::cerr << "CUDA is not available. Exiting." << std::endl;
    return 1;
  }
  torch::manual_seed(42);
  torch::cuda::manual_seed(42);

  std::vector<int> sizes = {4096, 4096, 4096};
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);

  int M = sizes[0];
  int N = sizes[1];
  int K = sizes[2];

  // Declare and initialize missing variables
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;

  auto A = torch::randn({M, K}, options);
  auto B = torch::randn({K, N}, options);
  auto C = torch::randn({M, N}, options);
  auto golden = alpha * torch::matmul(A, B) + beta * C;
  std::cout << "golden: " << golden.select(1, 0).slice(0, 0, 5) << "\n\n";

  for (auto kernel : {sgemm_naive, sgemm_shmem, sgemm_2dtiling}) {
    // for (auto kernel : {sgemm_naive, sgemm_shmem, sgemm_2dtiling,
    // SgemmWarptiling, Sgemm2DBlocktiling}) {
    auto _C = C.clone();
    kernel(A, B, _C, alpha, beta);
    bool is_correct = torch::allclose(_C, golden, 1e-4, 1e-4);
    std::cout << ">>> M: " << M << ", N: " << N << ", K: " << K
              << "\n>>> is_correct: " << is_correct << std::endl;
    std::cout << "C: " << _C.select(1, 0).slice(0, 0, 5) << "\n";
  }

  return 0;
}

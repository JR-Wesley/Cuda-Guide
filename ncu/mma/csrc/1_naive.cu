#include "gemm.h"

__global__ void sgemm_naive_kernel(const int M, const int N, const int K,
                                   const float alpha, const float *A,
                                   const float *B, const float beta, float *C) {

  // compute position in C that this thread is responsible for
  const uint n = blockIdx.x * blockDim.x + threadIdx.x;
  const uint m = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (n < N && m < M) {
    float sum = 0.0f;
    for (uint i = 0; i < K; ++i) {
      sum += A[m * K + i] * B[i * N + n];
    }

    // C = α*(A@B)+β*C
    C[m * N + n] = alpha * sum + beta * C[m * N + n];
  }
}

void sgemm_naive(torch::Tensor A, torch::Tensor B, torch::Tensor C, float alpha,
                 float beta) {
  TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
  TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
  TORCH_CHECK(C.device().is_cuda(), "C must be a CUDA tensor");
  TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be a float32 tensor");
  TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be a float32 tensor");
  TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be a float32 tensor");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");

  const int M = A.size(0);
  const int K = A.size(1);
  const int N = B.size(1);
  TORCH_CHECK(A.size(1) == B.size(0),
              "A and B must have compatible dimensions");
  TORCH_CHECK(C.size(0) == A.size(0) && C.size(1) == B.size(1),
              "C must have the same dimensions as A@B");

  // create as many blocks as necessary to map all of C
  // 32 * 32 = 1024 thread per block
  constexpr int block_x = 32;
  constexpr int block_y = 32;
  dim3 gridDim(CEIL_DIV(N, block_x), CEIL_DIV(M, block_y), 1);
  dim3 blockDim(block_x, block_y, 1);
  sgemm_naive_kernel<<<gridDim, blockDim>>>(M, N, K, alpha, A.data_ptr<float>(),
                                            B.data_ptr<float>(), beta,
                                            C.data_ptr<float>());
}

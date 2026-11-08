#include "gemm.h"

// tile size equals to block size
static constexpr uint BM = 32;
static constexpr uint BN = 32;
static constexpr uint BK = 32;

__global__ void sgemm_shmem_kernel(const int M, const int N, const int K,
                                   const float alpha, const float *A,
                                   const float *B, const float beta, float *C) {
  // shared memory for a row tile in A and a column tile in B
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  const uint tx = threadIdx.x;
  const uint ty = threadIdx.y;

  // indices for this tile
  // note that TILE_SIZE = blockDim for simplicity
  const uint n = blockIdx.x * BN + tx;
  const uint m = blockIdx.y * BM + ty;

  // Each thread computes one element of C
  float sum = 0.0f;

  // Loop over K dimension
  for (uint k = 0; k < K; k += BK) {
    // Have each thread load its own element in A & B into shared memory
    // Load A[m][k+tx]
    if (m < M && (k + tx) < K) {
      As[ty][tx] = A[m * K + k + tx];
    } else {
      As[ty][tx] = 0.0f;
    }
    // Load B[k+ty][n]
    if (n < N && (k + ty) < K) {
      Bs[ty][tx] = B[(k + ty) * N + n];
    } else {
      Bs[ty][tx] = 0.0f;
    }

    __syncthreads();

    // Compute partial dot products
    for (uint i = 0; i < BK; ++i) {
      sum += As[ty][i] * Bs[i][tx];
    }
    __syncthreads();
  }

  // Write result to C
  if (m < M && n < N) {
    C[m * N + n] = alpha * sum + beta * C[m * N + n];
  }
}

void sgemm_shmem(torch::Tensor A, torch::Tensor B, torch::Tensor C, float alpha,
                 float beta) {
  TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
  TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
  TORCH_CHECK(C.device().is_cuda(), "C must be a CUDA tensor");
  TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be a float32 tensor");
  TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be a float32 tensor");
  TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be a float32 tensor");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");

  const int M = A.size(0);
  const int N = B.size(1);
  const int K = A.size(1);
  TORCH_CHECK(A.size(1) == B.size(0),
              "A and B must have compatible dimensions");
  TORCH_CHECK(C.size(0) == A.size(0) && C.size(1) == B.size(1),
              "C must have the same dimensions as A@B");

  // gridDim stays the same
  // make blockDim 1-dimensional, but don't change number of threads
  constexpr int block_x = 32;
  constexpr int block_y = 32;
  dim3 gridDim(CEIL_DIV(N, block_x), CEIL_DIV(M, block_y), 1);
  dim3 blockDim(block_x, block_y, 1);
  sgemm_shmem_kernel<<<gridDim, blockDim>>>(M, N, K, alpha, A.data_ptr<float>(),
                                            B.data_ptr<float>(), beta,
                                            C.data_ptr<float>());
}

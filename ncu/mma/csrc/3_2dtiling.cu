#include "gemm.h"

template <uint BM, uint BN, uint BK, uint TM, uint TN>
__global__ void sgemm_2dtiling_kernel(const int M, const int N, const int K,
                                      const float alpha, const float *A,
                                      const float *B, const float beta,
                                      float *C) {
  // shared memory for a block tile in A & B
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  // block tile
  const uint bx = blockIdx.x;
  const uint by = blockIdx.y;
  // thread tile
  const uint tx = threadIdx.x;
  const uint ty = threadIdx.y;

  // Each thread computes a TMxTN tile of C
  float res[TM][TN] = {0.0f};
  float fragA[TM] = {0.0f};
  float fragB[TN] = {0.0f};

  for (uint kIdx = 0; kIdx < K; kIdx += BK) {
    // load A B into shared memory
    // Each thread loads a partion of a tile of shmem
    // TN = BK & TM = BK so there is only one loop
    // Load A[by*BM+ty*TM + m][k*BK+tx]
    for (uint m = 0; m < TM; ++m) {
      if (by * BM + ty * TM + m < M && kIdx + tx < K) {
        As[ty * TM + m][tx] = A[(by * BM + ty * TM + m) * K + kIdx + tx];
      } else {
        As[ty * TM + m][tx] = 0.0f;
      }
    }
    // Load B[k*BK+ty][bx*BN+tx*TN+n]
    for (uint n = 0; n < TN; ++n) {
      if (kIdx + ty < K && bx * BN + tx * TN + n < N) {
        Bs[ty][tx * TN + n] = B[(kIdx + ty) * N + bx * BN + tx * TN + n];
      } else {
        Bs[ty][tx * TN + n] = 0.0f;
      }
    }
    __syncthreads();

    // calculate per-thread results
    // note that this loop is outer dot product
    for (uint bk = 0; bk < BK; ++bk) {
      // load fragments from shared memory into registers
      for (uint i = 0; i < TM; ++i) {
        fragA[i] = As[ty * TM + i][bk];
      }
      for (uint j = 0; j < TN; ++j) {
        fragB[j] = Bs[bk][tx * TN + j];
      }

      // compute partial dot products
      for (uint i = 0; i < TM; ++i) {
        for (uint j = 0; j < TN; ++j) {
          res[i][j] += fragA[i] * fragB[j];
        }
      }
    }
    __syncthreads();
  }
  // write back to global memory
  for (uint i = 0; i < TM; ++i) {
    for (uint j = 0; j < TN; ++j) {
      // offset calculation: block offset + thread tile offset + element offset
      if (by * BM + ty * TM + i < M && bx * BN + tx * TN + j < N) {
        C[(by * BM + ty * TM + i) * N + (bx * BN + tx * TN + j)] =
            alpha * res[i][j] +
            beta * C[(by * BM + ty * TM + i) * N + (bx * BN + tx * TN + j)];
      }
    }
  }
}

void sgemm_2dtiling(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                    float alpha, float beta) {
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

  // each thread computed a TM * TN in C
  constexpr uint TM = 8;
  constexpr uint TN = 8;
  constexpr uint BDM = 16;
  constexpr uint BDN = 16;
  // block tile size corresponds to the dim of input matrix
  constexpr uint BM = TM * BDM;
  constexpr uint BN = TN * BDN;
  // For simplicity, BK = BDM = BDN
  constexpr uint BK = 16;

  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);
  dim3 blockDim(BDN, BDM, 1);

  sgemm_2dtiling_kernel<BM, BN, BK, TM, TN>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A.data_ptr<float>(),
                              B.data_ptr<float>(), beta, C.data_ptr<float>());
}

#include <cuda_runtime.h>
#include <cstdio>

// Small on purpose: the generated host stub, PTX, cubin and fatbin are easier to read.
extern "C" __global__ void saxpy_probe(const float* x, float* y, float a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
  float *x = nullptr, *y = nullptr;
  constexpr int n = 256;
  float host_x[n], host_y[n];
  for (int i = 0; i < n; ++i) {
    host_x[i] = 1.0f;
    host_y[i] = 1.0f;
  }
  if (cudaMalloc(&x, n * sizeof(float)) != cudaSuccess) return 1;
  if (cudaMalloc(&y, n * sizeof(float)) != cudaSuccess) return 1;
  if (cudaMemcpy(x, host_x, n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
    return 1;
  if (cudaMemcpy(y, host_y, n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
    return 1;
  saxpy_probe<<<1, n>>>(x, y, 2.0f, n);
  cudaError_t result = cudaGetLastError();
  if (result == cudaSuccess) result = cudaDeviceSynchronize();
  if (result == cudaSuccess)
    result = cudaMemcpy(host_y, y, n * sizeof(float), cudaMemcpyDeviceToHost);
  bool correct = result == cudaSuccess;
  if (result == cudaSuccess) {
    int mismatches = 0;
    for (int i = 0; i < n; ++i) {
      if (host_y[i] != 3.0f) ++mismatches;
    }
    correct = mismatches == 0;
    std::printf("saxpy_probe correct=%s, mismatches=%d, first=%.9g\n",
                correct ? "yes" : "no", mismatches, host_y[0]);
  } else {
    std::printf("saxpy_probe correct=no, CUDA=%s\n", cudaGetErrorString(result));
  }
  cudaFree(x);
  cudaFree(y);
  return correct ? 0 : 1;
}

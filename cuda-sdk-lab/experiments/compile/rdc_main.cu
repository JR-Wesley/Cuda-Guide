#include <cuda_runtime.h>

extern "C" __device__ float rdc_scale(float x);

extern "C" __global__ void rdc_kernel(float* out) {
  if (threadIdx.x == 0) out[0] = rdc_scale(3.0f);
}

int main() {
  float* out = nullptr;
  if (cudaMalloc(&out, sizeof(float)) != cudaSuccess) return 1;
  rdc_kernel<<<1, 1>>>(out);
  float value = 0;
  if (cudaMemcpy(&value, out, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
    return 1;
  cudaFree(out);
  return value == 6.0f ? 0 : 1;
}

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call) do {                                      \
  cudaError_t err = (call);                                         \
  if (err != cudaSuccess) {                                         \
    std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,  \
                 __LINE__, cudaGetErrorString(err));               \
    std::exit(EXIT_FAILURE);                                        \
  }                                                                \
} while (0)

extern "C" __global__ void vector_add(const float* a, const float* b,
                                        float* out, size_t count) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) out[i] = a[i] + b[i];
}

extern "C" __global__ void transpose_naive(const float* in, float* out,
                                             int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < width) out[x * width + y] = in[y * width + x];
}

// The extra column avoids shared-memory bank conflicts during the transposed read.
extern "C" __global__ void transpose_tiled(const float* in, float* out,
                                             int width) {
  __shared__ float tile[32][33];
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;
  for (int j = 0; j < 32; j += 8) {
    if (x < width && y + j < width)
      tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
  }
  __syncthreads();
  x = blockIdx.y * 32 + threadIdx.x;
  y = blockIdx.x * 32 + threadIdx.y;
  for (int j = 0; j < 32; j += 8) {
    if (x < width && y + j < width)
      out[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
  }
}

static void usage(const char* program) {
  std::fprintf(stderr,
      "Usage: %s --case vector|naive|tiled [--n N] [--repeat R]\n"
      "  N is vector length, or square-matrix width for transpose.\n",
      program);
}

int main(int argc, char** argv) {
  const char* which = "vector";
  int n = 1024;
  int repeat = 100;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--case") && ++i < argc) which = argv[i];
    else if (!std::strcmp(argv[i], "--n") && ++i < argc) n = std::atoi(argv[i]);
    else if (!std::strcmp(argv[i], "--repeat") && ++i < argc) repeat = std::atoi(argv[i]);
    else { usage(argv[0]); return 2; }
  }
  if (n <= 0 || repeat <= 0 ||
      (std::strcmp(which, "vector") && std::strcmp(which, "naive") &&
       std::strcmp(which, "tiled"))) {
    usage(argv[0]); return 2;
  }
  // Keep the beginner lab's allocations and grid dimensions modest.
  int max_n = !std::strcmp(which, "vector") ? 16777216 : 8192;
  if (n > max_n) {
    std::fprintf(stderr, "N must be <= %d for this case\n", max_n);
    return 2;
  }

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  size_t count = !std::strcmp(which, "vector") ? size_t(n) : size_t(n) * n;
  size_t bytes = count * sizeof(float);
  std::vector<float> input(count), second(count), result(count);
  for (size_t i = 0; i < count; ++i) {
    input[i] = float(i % 97) * 0.125f;
    second[i] = float(i % 71) * 0.25f;
  }
  float *d_in = nullptr, *d_second = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice));
  if (!std::strcmp(which, "vector")) {
    CUDA_CHECK(cudaMalloc(&d_second, bytes));
    CUDA_CHECK(cudaMemcpy(d_second, second.data(), bytes, cudaMemcpyHostToDevice));
  }

  auto launch = [&] {
    if (!std::strcmp(which, "vector"))
      vector_add<<<(count + 255) / 256, 256>>>(d_in, d_second, d_out, count);
    else if (!std::strcmp(which, "naive"))
      transpose_naive<<<dim3((n + 31) / 32, (n + 7) / 8), dim3(32, 8)>>>(d_in, d_out, n);
    else
      transpose_tiled<<<dim3((n + 31) / 32, (n + 31) / 32), dim3(32, 8)>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());
  };

  launch();  // Warm up JIT compilation and the device.
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repeat; ++i) launch();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float total_ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  CUDA_CHECK(cudaMemcpy(result.data(), d_out, bytes, cudaMemcpyDeviceToHost));

  size_t errors = 0;
  double sum = 0;
  for (size_t i = 0; i < count; ++i) {
    float expected = !std::strcmp(which, "vector")
        ? input[i] + second[i]
        : input[(i % n) * n + (i / n)];
    if (std::fabs(result[i] - expected) > 1e-6f) {
      if (errors < 3) std::fprintf(stderr, "mismatch[%zu]: got %.9g expected %.9g\n",
                                   i, result[i], expected);
      ++errors;
    }
    sum += result[i];
  }
  double us = double(total_ms) * 1000.0 / repeat;
  // One read + one write for transpose; two reads + one write for vector add.
  double traffic = double(bytes) * (!std::strcmp(which, "vector") ? 3 : 2);
  std::printf("device=%s, case=%s, n=%d, repeat=%d, elements=%zu\n",
              prop.name, which, n, repeat, count);
  std::printf("correct=%s, errors=%zu, checksum=%.6f\n",
              errors ? "no" : "yes", errors, sum);
  std::printf("batch_us_per_launch=%.3f, effective_GBps=%.3f\n", us, traffic / (us * 1000.0));
  std::printf("timing_scope=CUDA event around all launches; may include launch gaps; excludes allocation and copies\n");

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_in));
  if (d_second) CUDA_CHECK(cudaFree(d_second));
  CUDA_CHECK(cudaFree(d_out));
  return errors ? 1 : 0;
}

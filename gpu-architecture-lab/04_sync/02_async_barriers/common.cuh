#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>

#define CUDA(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d %s: %s (%s)\n", __FILE__, __LINE__, #call, \
               cudaGetErrorName(e_), cudaGetErrorString(e_)); std::exit(1); } } while (0)
inline cudaDeviceProp device() {
  int count=0; auto e=cudaGetDeviceCount(&count);
  if (e!=cudaSuccess || !count) { std::printf("SKIP no usable GPU: %s\n",cudaGetErrorString(e)); std::exit(77); }
  cudaDeviceProp p{}; CUDA(cudaGetDeviceProperties(&p,0));
  int driver=0,runtime=0; CUDA(cudaDriverGetVersion(&driver)); CUDA(cudaRuntimeGetVersion(&runtime));
  std::printf("GPU=%s sm=%d%d driver=%d runtime=%d\n",p.name,p.major,p.minor,driver,runtime);
  return p;
}
inline void finish() { CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize()); }
template<class T> struct Buffer {
  T* p; size_t n;
  explicit Buffer(size_t count):n(count) { CUDA(cudaMalloc(&p,n*sizeof(T))); CUDA(cudaMemset(p,0,n*sizeof(T))); }
  ~Buffer() { cudaFree(p); }
  Buffer(const Buffer&)=delete;
  Buffer& operator=(const Buffer&)=delete;
  std::vector<T> read() const { std::vector<T> h(n); CUDA(cudaMemcpy(h.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost)); return h; }
};
inline void require(bool ok,const char* name) {
  std::printf("%s %s\n",ok?"PASS":"FAIL",name);
  if(!ok) std::exit(1);
}

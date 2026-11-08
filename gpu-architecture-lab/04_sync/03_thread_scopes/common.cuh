#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cstdint>
#define CUDA(call) do { auto e_=(call); if(e_!=cudaSuccess) { \
  std::fprintf(stderr,"%s:%d %s -> %s: %s\n",__FILE__,__LINE__,#call,cudaGetErrorName(e_),cudaGetErrorString(e_)); std::exit(1); } } while(0)
inline cudaDeviceProp device() {
  int count=0; auto e=cudaGetDeviceCount(&count);
  if(e!=cudaSuccess || count==0) { std::printf("SKIP no GPU: %s\n",cudaGetErrorString(e)); std::exit(77); }
  cudaDeviceProp p{}; CUDA(cudaGetDeviceProperties(&p,0));
  int driver=0,runtime=0; CUDA(cudaDriverGetVersion(&driver)); CUDA(cudaRuntimeGetVersion(&runtime));
  std::printf("GPU=%s sm=%d%d driver=%d runtime=%d cooperative=%d concurrentManagedAccess=%d hostNativeAtomic=%d\n",
    p.name,p.major,p.minor,driver,runtime,p.cooperativeLaunch,p.concurrentManagedAccess,p.hostNativeAtomicSupported);
  if(p.major<8) { std::puts("SKIP this lab targets sm_80+"); std::exit(77); }
  return p;
}
inline void finish() { CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize()); }
inline void check(bool ok,const char* name) { std::printf("%s %s\n",ok?"PASS":"FAIL",name); if(!ok) std::exit(1); }
template<class T> struct Buffer {
  T* p; size_t n;
  explicit Buffer(size_t count):n(count) { CUDA(cudaMalloc(&p,n*sizeof(T))); clear(); }
  ~Buffer() { cudaFree(p); }
  Buffer(const Buffer&)=delete; Buffer& operator=(const Buffer&)=delete;
  void clear() { CUDA(cudaMemset(p,0,n*sizeof(T))); }
  std::vector<T> read() const { std::vector<T> h(n); CUDA(cudaMemcpy(h.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost)); return h; }
};
constexpr int Threads=64,Blocks=8,Tile=128;
inline bool tickets_ok(std::vector<unsigned> tickets,int groups,int size) {
  for(int g=0;g<groups;++g) {
    auto b=tickets.begin()+g*size; std::sort(b,b+size);
    for(int i=0;i<size;++i) if(b[i]!=unsigned(i)) return false;
  }
  return true;
}

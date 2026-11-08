#include "common.cuh"
#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda/std/utility>

// Shared object initialization is performed explicitly by thread 0.
#pragma nv_diag_suppress static_var_with_dynamic_init
// Compile separately for sm_90. Never silently replace transaction tracking.
__global__ void transaction_copy(const unsigned* in,unsigned* out) {
  __shared__ alignas(16) unsigned data[256];
  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  int t=threadIdx.x;
  if(t==0) init(&bar,blockDim.x);
  __syncthreads();
  if(t==0) {
    cuda::device::memcpy_async_tx(data,in,cuda::aligned_size_t<16>(sizeof(data)),bar);
    auto token=cuda::device::barrier_arrive_tx(bar,1,sizeof(data));
    bar.wait(cuda::std::move(token));
  } else {
    bar.arrive_and_wait();
  }
  for(int j=t;j<256;j+=blockDim.x) out[j]=data[j]*3u+7u;
}
int main() {
  auto p=device();
  if(p.major<9) { std::puts("SKIP transaction_copy requires sm_90+; no kernel launched"); return 77; }
  Buffer<unsigned> in(256),out(256);
  std::vector<unsigned> h(256);
  for(unsigned i=0;i<h.size();++i) h[i]=i*17u+11u;
  CUDA(cudaMemcpy(in.p,h.data(),h.size()*sizeof(unsigned),cudaMemcpyHostToDevice));
  transaction_copy<<<1,128>>>(in.p,out.p); finish(); auto got=out.read();
  bool ok=true; for(int i=0;i<256;++i) ok &= got[i]==h[i]*3u+7u;
  require(ok,"transaction_copy_1024_bytes");
}

#include "common.cuh"
__global__ void shared_capacity(unsigned* out,int count) {
  extern __shared__ unsigned data[];
  for(int i=threadIdx.x;i<count;i+=blockDim.x) data[i]=i+1;
  __syncthreads();
  unsigned sum=0;
  for(int i=threadIdx.x;i<count;i+=blockDim.x) sum+=data[(i+1)%count];
  out[threadIdx.x]=sum;
}
void validate(Buffer<unsigned>& out,int count) {
  auto h=out.read(); bool ok=true;
  for(int t=0;t<Threads;++t) { unsigned e=0; for(int i=t;i<count;i+=Threads) e+=(i+1)%count+1; ok &= h[t]==e; }
  check(ok,"dynamic_shared_neighbor_sum");
}
int main(int argc,char** argv) {
  auto p=device(); bool debug=argc==2 && std::string(argv[1])=="debug";
  std::printf("SHARED default_limit=%zu optin_limit=%zu per_sm=%zu\n",p.sharedMemPerBlock,p.sharedMemPerBlockOptin,p.sharedMemPerMultiprocessor);
  Buffer<unsigned> out(Threads);
  int bytes=std::min<size_t>(64*1024,p.sharedMemPerBlockOptin),count=bytes/sizeof(unsigned);
  if(debug) {
    if(bytes<=int(p.sharedMemPerBlock)) { std::puts("SKIP no allocation above default limit available"); return 77; }
    shared_capacity<<<1,Threads,bytes>>>(out.p,count);
    auto e=cudaGetLastError();
    std::printf("EXPECTED_LAUNCH_FAILURE before_optin=%s\n",cudaGetErrorName(e));
    check(e==cudaErrorInvalidValue || e==cudaErrorInvalidConfiguration,"large_shared_requires_optin");
  }
  CUDA(cudaFuncSetAttribute(shared_capacity,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
  const int hints[]={cudaSharedmemCarveoutDefault,cudaSharedmemCarveoutMaxL1,50,cudaSharedmemCarveoutMaxShared};
  for(int hint:hints) {
    CUDA(cudaFuncSetAttribute(shared_capacity,cudaFuncAttributePreferredSharedMemoryCarveout,hint));
    cudaFuncAttributes a{}; CUDA(cudaFuncGetAttributes(&a,shared_capacity));
    int active=0; CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,shared_capacity,Threads,bytes));
    std::printf("CARVEOUT request=%d reported_preference=%d optin_attribute=%d static_bytes=%zu launch_bytes=%d predicted_blocks_per_sm=%d\n",
      hint,a.preferredShmemCarveout,a.maxDynamicSharedSizeBytes,a.sharedSizeBytes,bytes,active);
    shared_capacity<<<1,Threads,bytes>>>(out.p,count); finish(); validate(out,count);
  }
  std::puts("SUMMARY 4 carveout preferences verified; physical partition and performance not measured");
}

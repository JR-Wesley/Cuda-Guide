#include "common.cuh"
#include <cuda/barrier>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda_pipeline.h>
#pragma nv_diag_suppress static_var_with_dynamic_init

// Read a neighbor as well as our own element: completion must cover the group.
__device__ void consume(const unsigned* tile,unsigned* out,int offset,int n) {
  for(int j=threadIdx.x;j<Tile && offset+j<n;j+=blockDim.x)
    out[offset+j]=tile[j]+3u*tile[(j+1)%Tile];
}
template<int Mode> __global__ void copy_single(const unsigned* in,unsigned* out,int n) {
  __shared__ alignas(16) unsigned tile[Tile];
  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  auto group=cooperative_groups::this_thread_block();
  if constexpr(Mode==3) { if(threadIdx.x==0) init(&bar,blockDim.x); group.sync(); }
  for(int offset=0;offset<n;offset+=Tile) {
    if constexpr(Mode==0) {
      for(int j=threadIdx.x;j<Tile;j+=blockDim.x) tile[j]=in[offset+j];
      group.sync();
    } else if constexpr(Mode==1) {
      cooperative_groups::memcpy_async(group,tile,in+offset,sizeof(tile));
      cooperative_groups::wait(group);
    } else if constexpr(Mode==2) {
      for(int j=threadIdx.x;j<Tile;j+=blockDim.x) __pipeline_memcpy_async(&tile[j],&in[offset+j],sizeof(unsigned));
      __pipeline_commit();
      __pipeline_wait_prior(0); // Only this thread's committed operations.
      group.sync(); // Needed before reading slots copied by other threads.
    } else {
      cuda::memcpy_async(group,tile,in+offset,cuda::aligned_size_t<16>(sizeof(tile)),bar);
      bar.arrive_and_wait();
    }
    consume(tile,out,offset,n);
    if constexpr(Mode==3) bar.arrive_and_wait();
    else group.sync(); // Readers done before overwriting the shared tile.
  }
}
extern "C" __global__ void copy_pipeline(const unsigned* in,unsigned* out,int n) {
  __shared__ alignas(16) unsigned tile[2][Tile];
  __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block,2> state;
  auto group=cooperative_groups::this_thread_block();
  auto pipe=cuda::make_pipeline(group,&state);
  int tiles=(n+Tile-1)/Tile,next=0;
  for(int used=0;used<tiles;++used) {
    while(next<tiles && next<used+2) {
      pipe.producer_acquire();
      cuda::memcpy_async(group,tile[next%2],in+next*Tile,cuda::aligned_size_t<16>(sizeof(tile[0])),pipe);
      pipe.producer_commit(); ++next;
    }
    pipe.consumer_wait();
    consume(tile[used%2],out,used*Tile,n);
    pipe.consumer_release(); // All participants release before this slot is reused.
  }
}
__global__ void wrong_bytes(const unsigned* in,unsigned* out) {
  __shared__ unsigned tile[Tile];
  auto group=cooperative_groups::this_thread_block();
  for(int j=threadIdx.x;j<Tile;j+=blockDim.x) tile[j]=0xdeadbeefu;
  group.sync();
  // Four-argument overload counts BYTES, not unsigned elements.
  cooperative_groups::memcpy_async(group,tile,in,size_t(Tile));
  cooperative_groups::wait(group);
  for(int j=threadIdx.x;j<Tile;j+=blockDim.x) out[j]=tile[j];
}
int main(int argc,char** argv) {
  device(); bool debug=argc==2 && std::string(argv[1])=="debug";
  if(debug) {
    Buffer<unsigned> in(Tile),out(Tile); std::vector<unsigned> h(Tile);
    for(int i=0;i<Tile;++i) h[i]=i+1;
    CUDA(cudaMemcpy(in.p,h.data(),sizeof(unsigned)*Tile,cudaMemcpyHostToDevice));
    wrong_bytes<<<1,Threads>>>(in.p,out.p); finish(); auto got=out.read();
    int bad=0,poison=0; for(int i=0;i<Tile;++i) { bad+=got[i]!=h[i]; poison+=got[i]==0xdeadbeefu; }
    std::printf("EXPECTED_MISMATCH requested_bytes=%d required_bytes=%zu mismatches=%d poison=%d\n",Tile,Tile*sizeof(unsigned),bad,poison);
    check(bad==96 && poison==96,"byte_count_bug_detected");
    return 0;
  }
  const char* names[]={"synchronous","cooperative_groups_bytes","thread_primitives_plus_block_sync","barrier_bound","two_stage_pipeline"};
  for(int n:{1,127,128,129,1027}) {
    int padded=(n+Tile-1)/Tile*Tile; std::vector<unsigned> h(padded,0);
    for(int i=0;i<n;++i) h[i]=17u*i+11u;
    Buffer<unsigned> in(padded),out(n);
    CUDA(cudaMemcpy(in.p,h.data(),padded*sizeof(unsigned),cudaMemcpyHostToDevice));
    for(int mode=0;mode<5;++mode) {
      out.clear();
      if(mode==0) copy_single<0><<<1,Threads>>>(in.p,out.p,n);
      if(mode==1) copy_single<1><<<1,Threads>>>(in.p,out.p,n);
      if(mode==2) copy_single<2><<<1,Threads>>>(in.p,out.p,n);
      if(mode==3) copy_single<3><<<1,Threads>>>(in.p,out.p,n);
      if(mode==4) copy_pipeline<<<1,Threads>>>(in.p,out.p,n);
      finish(); auto got=out.read(); bool ok=true;
      for(int i=0;i<n;++i) ok &= got[i]==h[i]+3u*h[i/Tile*Tile+(i%Tile+1)%Tile];
      std::printf("COPY n=%d padded=%d ",n,padded); check(ok,names[mode]);
    }
  }
  std::puts("SUMMARY copies 25 cases passed; overlap/speedup not measured");
}

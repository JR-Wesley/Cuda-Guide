#include "common.cuh"
#include <cuda/atomic>
#include <cooperative_groups.h>

template<cuda::thread_scope Scope> using Ref=cuda::atomic_ref<unsigned,Scope>;
// Kernel boundaries provide host observation, never cross-block spin scheduling.
extern "C" __global__ void count_thread(unsigned* counters,unsigned* tickets,int steps) {
  int id=blockIdx.x*blockDim.x+threadIdx.x;
  Ref<cuda::thread_scope_thread> a(counters[id]);
  for(int k=0;k<steps;++k) tickets[id*steps+k]=a.fetch_add(1,cuda::memory_order_relaxed);
}
extern "C" __global__ void count_cta(unsigned* counters,unsigned* tickets,int steps) {
  int id=blockIdx.x*blockDim.x+threadIdx.x;
  Ref<cuda::thread_scope_block> a(counters[blockIdx.x]);
  for(int k=0;k<steps;++k) tickets[id*steps+k]=a.fetch_add(1,cuda::memory_order_relaxed);
}
extern "C" __global__ void count_shared(unsigned* counters,unsigned* tickets,int steps) {
  __shared__ alignas(Ref<cuda::thread_scope_block>::required_alignment) unsigned counter;
  if(threadIdx.x==0) counter=0;
  __syncthreads();
  { // End all atomic_ref lifetimes before the final ordinary read.
    Ref<cuda::thread_scope_block> a(counter);
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    for(int k=0;k<steps;++k) tickets[id*steps+k]=a.fetch_add(1,cuda::memory_order_relaxed);
  }
  __syncthreads();
  if(threadIdx.x==0) counters[blockIdx.x]=counter;
}
extern "C" __global__ void count_gpu(unsigned* counters,unsigned* tickets,int steps) {
  int id=blockIdx.x*blockDim.x+threadIdx.x;
  Ref<cuda::thread_scope_device> a(counters[0]);
  for(int k=0;k<steps;++k) tickets[id*steps+k]=a.fetch_add(1,cuda::memory_order_relaxed);
}
extern "C" __global__ void count_sys(unsigned* counters,unsigned* tickets,int steps) {
  int id=blockIdx.x*blockDim.x+threadIdx.x;
  Ref<cuda::thread_scope_system> a(counters[0]);
  for(int k=0;k<steps;++k) tickets[id*steps+k]=a.fetch_add(1,cuda::memory_order_relaxed);
}
extern "C" __global__ void count_builtin(unsigned* counters,unsigned* tickets,int steps) {
  int id=blockIdx.x*blockDim.x+threadIdx.x;
  for(int k=0;k<steps;++k) tickets[id*steps+k]=__nv_atomic_fetch_add(counters,1u,__NV_ATOMIC_RELAXED,__NV_THREAD_SCOPE_DEVICE);
}

struct Mailbox {
  alignas(Ref<cuda::thread_scope_system>::required_alignment) unsigned flag;
  unsigned payload;
};
// Each block has its OWN global-memory mailbox. Location does not imply scope.
template<bool Builtin> __global__ void publish_block(Mailbox* boxes,unsigned* outputs,int rounds) {
  int t=threadIdx.x,b=blockIdx.x; auto& box=boxes[b];
  for(int r=0;r<rounds;++r) {
    if(t==0) { box.flag=0; box.payload=0; }
    __syncthreads();
    {
      if(t==0) {
        box.payload=1000*r+b+42;
        if constexpr(Builtin) __nv_atomic_store_n(&box.flag,1u,__NV_ATOMIC_RELEASE,__NV_THREAD_SCOPE_BLOCK);
        else Ref<cuda::thread_scope_block>(box.flag).store(1,cuda::memory_order_release);
      }
      if(t==32) {
        bool ready=false;
        for(int i=0;i<1000000;++i) {
          unsigned v;
          if constexpr(Builtin) v=__nv_atomic_load_n(&box.flag,__NV_ATOMIC_ACQUIRE,__NV_THREAD_SCOPE_BLOCK);
          else v=Ref<cuda::thread_scope_block>(box.flag).load(cuda::memory_order_acquire);
          if(v==1) { ready=true; break; }
          __nanosleep(64);
        }
        outputs[b*rounds+r]=ready?box.payload:0xffffffffu;
      }
    }
    __syncthreads(); // All readers done before the next non-atomic reset.
  }
}
__global__ void publish_device(Mailbox* box,unsigned* out,int rounds) {
  auto grid=cooperative_groups::this_grid();
  for(int r=0;r<rounds;++r) {
    if(grid.thread_rank()==0) { box->flag=0; box->payload=0; }
    grid.sync();
    {
      Ref<cuda::thread_scope_device> flag(box->flag);
      if(blockIdx.x==0 && threadIdx.x==0) {
        box->payload=1000*r+42; flag.store(1,cuda::memory_order_release);
      }
      if(blockIdx.x==1 && threadIdx.x==0) {
        bool ready=false;
        for(int i=0;i<1000000;++i) {
          if(flag.load(cuda::memory_order_acquire)==1) { ready=true; break; }
          __nanosleep(64);
        }
        out[r]=ready?box->payload:0xffffffffu;
      }
    }
    grid.sync();
  }
}

int main(int argc,char** argv) {
  auto p=device(); int steps=argc>1?std::stoi(argv[1]):7;
  if(steps<1 || steps>1000) return 2;
  Buffer<unsigned> counters(Blocks*Threads),tickets(Blocks*Threads*steps);
  const char* names[]={"thread_own_global_slot","block_global_partition","block_shared",
                       "device_global","system_gpu_only","builtin_device"};
  for(int mode=0;mode<6;++mode) {
    counters.clear(); tickets.clear();
    if(mode==0) count_thread<<<Blocks,Threads>>>(counters.p,tickets.p,steps);
    if(mode==1) count_cta<<<Blocks,Threads>>>(counters.p,tickets.p,steps);
    if(mode==2) count_shared<<<Blocks,Threads>>>(counters.p,tickets.p,steps);
    if(mode==3) count_gpu<<<Blocks,Threads>>>(counters.p,tickets.p,steps);
    if(mode==4) count_sys<<<Blocks,Threads>>>(counters.p,tickets.p,steps);
    if(mode==5) count_builtin<<<Blocks,Threads>>>(counters.p,tickets.p,steps);
    finish(); auto c=counters.read();
    int groups=mode==0?Blocks*Threads:mode<3?Blocks:1;
    int per=Blocks*Threads*steps/groups; bool ok=tickets_ok(tickets.read(),groups,per);
    for(int g=0;g<groups;++g) ok &= c[g]==unsigned(per);
    std::printf("COUNTER groups=%d increments_per_group=%d ",groups,per); check(ok,names[mode]);
  }
  constexpr int rounds=19;
  Buffer<Mailbox> boxes(Blocks); Buffer<unsigned> values(Blocks*rounds);
  for(int mode=0;mode<2;++mode) {
    if(mode==0) publish_block<false><<<Blocks,Threads>>>(boxes.p,values.p,rounds);
    else publish_block<true><<<Blocks,Threads>>>(boxes.p,values.p,rounds);
    finish(); auto h=values.read(); bool ok=true;
    for(int b=0;b<Blocks;++b) for(int r=0;r<rounds;++r) ok &= h[b*rounds+r]==unsigned(1000*r+b+42);
    check(ok,mode==0?"block_release_acquire":"builtin_block_release_acquire");
  }
  int active=0; CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,publish_device,Threads,0));
  if(!p.cooperativeLaunch || active*p.multiProcessorCount<2) { std::puts("SKIP cooperative device publication unavailable"); return 77; }
  Mailbox* b=boxes.p; unsigned* v=values.p; int r=rounds;
  void* args[]={&b,&v,&r};
  CUDA(cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(publish_device),2,Threads,args)); finish();
  auto h=values.read(); bool ok=true;
  for(int i=0;i<rounds;++i) ok &= h[i]==unsigned(1000*i+42);
  check(ok,"device_release_acquire_cooperative");
  std::puts("SUMMARY scopes 9 cases passed; timings not measured");
}

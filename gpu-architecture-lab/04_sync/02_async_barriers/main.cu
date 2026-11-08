#include "common.cuh"
#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cuda/std/utility>
#include <new>
#include <string>

// Shared barriers are explicitly initialized by one thread before participation.
#pragma nv_diag_suppress static_var_with_dynamic_init
using Barrier = cuda::barrier<cuda::thread_scope_block>;
constexpr int Threads=128, Blocks=4, Tile=256;
__host__ __device__ unsigned independent(unsigned x) {
  for(int i=0;i<16;++i) x=x*1664525u+1013904223u;
  return x;
}

// Two phases per round: publish, then acknowledge that all readers are done.
template<int Mode> __global__ void exchange(unsigned* out,int rounds) {
  __shared__ unsigned data[Threads];
  __shared__ Barrier bar;
  int t=threadIdx.x, id=blockIdx.x*Threads+t;
  if(t==0) init(&bar,Threads);
  __syncthreads();
  unsigned sum=0,parity=0;
  for(int r=0;r<rounds;++r) {
    data[t]=r*1000u+id;
    unsigned work;
    if constexpr(Mode==0) {
      __syncthreads(); work=independent(id+r);
    } else if constexpr(Mode==1) {
      auto token=bar.arrive();
      work=independent(id+r); // Registers only; does not touch data[].
      bar.wait(cuda::std::move(token));
    } else {
      auto* native=cuda::device::barrier_native_handle(bar);
      (void)cuda::ptx::mbarrier_arrive(native);
      work=independent(id+r);
#if defined(LAB_API_PARITY)
      bar.wait_parity(parity); // Diagnostic A/B: same semantics, library polling.
#elif __CUDA_ARCH__ >= 900
      while(!cuda::ptx::mbarrier_try_wait_parity(native,parity)) {}
#else
      while(!cuda::ptx::mbarrier_test_wait_parity(native,parity)) {}
#endif
      parity^=1;
    }
    sum+=data[(t+1)%Threads]+work;
    // A publish barrier alone does not prevent the next round overwriting data.
    if constexpr(Mode==0) __syncthreads();
    else if constexpr(Mode==1) bar.arrive_and_wait();
    else {
      auto* native=cuda::device::barrier_native_handle(bar);
      (void)cuda::ptx::mbarrier_arrive(native);
#if defined(LAB_API_PARITY)
      bar.wait_parity(parity);
#elif __CUDA_ARCH__ >= 900
      while(!cuda::ptx::mbarrier_try_wait_parity(native,parity)) {}
#else
      while(!cuda::ptx::mbarrier_test_wait_parity(native,parity)) {}
#endif
      parity^=1; // Track BOTH phases, not merely the loop index.
    }
  }
  out[id]=sum;
}

__global__ void early_exit(unsigned* out,int rounds) {
  __shared__ Barrier bar;
  __shared__ unsigned values[Threads];
  int t=threadIdx.x,limit=2*(t/32+1);
  if(t==0) init(&bar,Threads);
  __syncthreads();
  unsigned sum=0;
  for(int r=0;r<rounds;++r) {
    if(r==limit) { out[t]=sum; bar.arrive_and_drop(); return; }
    values[t]=1000*r+t;
    auto token=bar.arrive();
    bar.wait(cuda::std::move(token));
    for(int j=0;j<Threads;++j) if(r<2*(j/32+1)) sum+=values[j];
    bar.arrive_and_wait(); // Only remaining threads participate in this phase.
  }
  out[t]=sum;
}

struct SumCompletion {
  unsigned* values; unsigned* total; unsigned* phases;
  __device__ void operator()() noexcept {
    unsigned sum=0;
    for(int i=0;i<Threads;++i) sum+=values[i];
    *total+=sum; ++*phases;
  }
};
__global__ void completion(unsigned* out,int rounds) {
  using B=cuda::barrier<cuda::thread_scope_block,SumCompletion>;
  __shared__ unsigned values[Threads],total,phases;
  __shared__ alignas(B) unsigned char storage[sizeof(B)];
  auto* bar=reinterpret_cast<B*>(storage);
  int t=threadIdx.x;
  if(t==0) { total=0; phases=0; new(bar) B(Threads,SumCompletion{values,&total,&phases}); }
  __syncthreads();
  for(int r=0;r<rounds;++r) {
    values[t]=r*Threads+t+1;
    auto token=bar->arrive();
    bar->wait(cuda::std::move(token));
    out[r*Threads+t]=total;
    // Next completion cannot start until every thread has read total and arrived.
  }
  __syncthreads();
  if(t==0) { out[rounds*Threads]=phases; bar->~B(); }
}

template<bool Async> __global__ void tiled_copy(const unsigned* in,unsigned* out,unsigned* scratch,int n) {
  __shared__ alignas(16) unsigned tile[Tile];
  __shared__ Barrier bar;
  int t=threadIdx.x;
  if(t==0) init(&bar,Threads);
  __syncthreads();
  auto group=cooperative_groups::this_thread_block();
  unsigned work=0;
  for(int offset=0;offset<n;offset+=Tile) {
    if constexpr(Async) {
      // Source is padded to full tiles; all pointers and sizes are 16-byte aligned.
      cuda::memcpy_async(group,tile,in+offset,cuda::aligned_size_t<16>(sizeof(tile)),bar);
      auto token=bar.arrive();
      work+=independent(t+offset);
      bar.wait(cuda::std::move(token));
    } else {
      for(int j=t;j<Tile;j+=Threads) tile[j]=in[offset+j];
      __syncthreads(); work+=independent(t+offset);
    }
    for(int j=t;j<Tile && offset+j<n;j+=Threads) out[offset+j]=tile[j]*3u+7u;
    if constexpr(Async) bar.arrive_and_wait();
    else __syncthreads();
  }
  scratch[t]=work;
}

__global__ void producer_consumer(const unsigned* in,unsigned* out,int n) {
  __shared__ unsigned buffers[2][Tile];
  __shared__ Barrier ready[2],filled[2];
  int t=threadIdx.x,tiles=(n+Tile-1)/Tile;
  if(t<2) { init(&ready[t],Threads); init(&filled[t],Threads); }
  __syncthreads();
  if(t<32) {
    for(int k=0;k<tiles;++k) {
      int b=k%2;
      ready[b].arrive_and_wait();
      for(int j=t;j<Tile;j+=32) buffers[b][j]=in[k*Tile+j];
      (void)filled[b].arrive();
    }
  } else {
    for(int b=0;b<2 && b<tiles;++b) (void)ready[b].arrive();
    for(int k=0;k<tiles;++k) {
      int b=k%2;
      filled[b].arrive_and_wait();
      for(int j=t-32;j<Tile && k*Tile+j<n;j+=Threads-32) out[k*Tile+j]=buffers[b][j]*3u+7u;
      if(k+2<tiles) (void)ready[b].arrive(); // No unused final partial phase.
    }
  }
}

int main(int argc,char** argv) {
  int rounds=argc>1?std::stoi(argv[1]):17;
  if(rounds<1 || rounds>1000) { std::fprintf(stderr,"rounds must be 1..1000\n"); return 2; }
  auto p=device(); if(p.major<8) { std::puts("SKIP this lab requires sm_80+"); return 77; }
  std::printf("CONFIG rounds=%d blocks=%d threads=%d parity=%s\n",rounds,Blocks,Threads,
              p.major>=9?"try_wait (when compiled sm_90+)":"test_wait");
  Buffer<unsigned> out(Blocks*Threads);
  for(int mode=0;mode<3;++mode) {
    if(mode==0) exchange<0><<<Blocks,Threads>>>(out.p,rounds);
    if(mode==1) exchange<1><<<Blocks,Threads>>>(out.p,rounds);
    if(mode==2) exchange<2><<<Blocks,Threads>>>(out.p,rounds);
    finish(); auto h=out.read(); bool ok=true;
    for(int id=0;id<Blocks*Threads;++id) {
      unsigned expected=0;
      for(int r=0;r<rounds;++r) expected+=1000u*r+(id/Threads)*Threads+(id+1)%Threads+independent(id+r);
      ok &= h[id]==expected;
    }
    require(ok,mode==0?"syncthreads_baseline":mode==1?"split_arrive_wait":"explicit_parity");
  }
  int drop_rounds=rounds<8?rounds:8;
  early_exit<<<1,Threads>>>(out.p,drop_rounds); finish(); auto drops=out.read(); bool ok=true;
  for(int t=0;t<Threads;++t) {
    unsigned expected=0;
    for(int r=0;r<drop_rounds && r<2*(t/32+1);++r)
      for(int j=0;j<Threads;++j) if(r<2*(j/32+1)) expected+=1000*r+j;
    ok &= drops[t]==expected;
  }
  require(ok,"arrive_and_drop");
  Buffer<unsigned> sums(rounds*Threads+1);
  completion<<<1,Threads>>>(sums.p,rounds); finish(); auto h=sums.read(); ok=h.back()==unsigned(rounds);
  for(int r=0;r<rounds;++r) {
    unsigned count=(r+1)*Threads,expected;
    // Use a wide host reference before narrowing the modulo-2^32 sum.
    expected=unsigned(uint64_t(count)*(count+1)/2);
    for(int t=0;t<Threads;++t) ok &= h[r*Threads+t]==expected;
  }
  require(ok,"completion_once_per_phase");
  for(int n:{1,255,256,257,4099}) {
    int padded=(n+Tile-1)/Tile*Tile;
    std::vector<unsigned> input(padded,0);
    for(int i=0;i<n;++i) input[i]=i*17u+11u;
    Buffer<unsigned> src(padded),dst(n),work(Threads);
    CUDA(cudaMemcpy(src.p,input.data(),padded*sizeof(unsigned),cudaMemcpyHostToDevice));
    for(int mode=0;mode<3;++mode) {
      CUDA(cudaMemset(dst.p,0,n*sizeof(unsigned)));
      if(mode==0) tiled_copy<false><<<1,Threads>>>(src.p,dst.p,work.p,n);
      if(mode==1) tiled_copy<true><<<1,Threads>>>(src.p,dst.p,work.p,n);
      if(mode==2) producer_consumer<<<1,Threads>>>(src.p,dst.p,n);
      finish(); auto got=dst.read(); ok=true;
      for(int i=0;i<n;++i) ok &= got[i]==input[i]*3u+7u;
      if(mode<2) {
        auto w=work.read();
        for(int t=0;t<Threads;++t) { unsigned e=0; for(int off=0;off<n;off+=Tile) e+=independent(t+off); ok &= w[t]==e; }
      }
      std::printf("CASE n=%d padded=%d ",n,padded);
      require(ok,mode==0?"sync_copy":mode==1?"async_copy_barrier":"double_buffer");
    }
  }
  std::puts("SUMMARY 20 cases passed; performance not measured");
}

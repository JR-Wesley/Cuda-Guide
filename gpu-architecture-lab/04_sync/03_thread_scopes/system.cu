#include "common.cuh"
#include <cuda/atomic>
#include <chrono>
#include <thread>
#include <new>
using Atomic=cuda::atomic_ref<unsigned,cuda::thread_scope_system>;
struct Box {
  alignas(Atomic::required_alignment) unsigned request=0,ack=0,stop=0;
  unsigned payload=0,reply=0,done=0;
};
__global__ void host_device_exchange(Box* b,int rounds) {
  Atomic request(b->request),ack(b->ack),stop(b->stop);
  for(int r=1;r<=rounds;++r) {
    bool ready=false;
    for(int spin=0;spin<10000000;++spin) {
      if(stop.load(cuda::memory_order_relaxed)) return;
      if(request.load(cuda::memory_order_acquire)==unsigned(r)) { ready=true; break; }
      __nanosleep(128);
    }
    if(!ready) return;
    b->reply=b->payload*3u+7u;
    ack.store(r,cuda::memory_order_release);
  }
  b->done=rounds; // Host reads only after kernel synchronization.
}
int main() {
  auto p=device();
  if(!p.managedMemory || !p.concurrentManagedAccess) {
    std::puts("SKIP CPU/GPU simultaneous managed-memory access requires concurrentManagedAccess=1"); return 77;
  }
  Box* b=nullptr; CUDA(cudaMallocManaged(&b,sizeof(Box))); new(b) Box;
  if(!Atomic(b->request).is_lock_free()) { std::puts("SKIP host atomic_ref is not lock-free"); b->~Box(); CUDA(cudaFree(b)); return 77; }
  constexpr int rounds=32; bool ok=true;
  {
    Atomic request(b->request),ack(b->ack),stop(b->stop);
    host_device_exchange<<<1,1>>>(b,rounds); CUDA(cudaGetLastError());
    auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
    for(unsigned r=1;r<=rounds;++r) {
      b->payload=r+100; request.store(r,cuda::memory_order_release);
      while(ack.load(cuda::memory_order_acquire)!=r) {
        if(std::chrono::steady_clock::now()>deadline) { ok=false; break; }
        std::this_thread::yield();
      }
      if(!ok) break;
      if(b->reply!=(r+100)*3+7) { ok=false; break; }
    }
    stop.store(1,cuda::memory_order_relaxed);
    CUDA(cudaDeviceSynchronize()); ok &= b->done==rounds;
  }
  b->~Box(); CUDA(cudaFree(b));
  check(ok,"system_CPU_GPU_release_acquire_32_rounds");
}

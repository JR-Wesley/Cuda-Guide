#include "common.cuh"
#include <cuda/barrier>
#include <cuda/std/chrono>
#include <cuda/std/utility>
#include <cstring>
// Every shared barrier below has an explicit init before its first arrival.
#pragma nv_diag_suppress static_var_with_dynamic_init
using Barrier=cuda::barrier<cuda::thread_scope_block>;

// Count obligations, not necessarily unique threads. No intentional deadlock.
__global__ void missing_arrival(unsigned* out) {
  __shared__ Barrier bar;
  init(&bar,2); // This kernel has exactly ONE thread.
  auto token=bar.arrive(); // One obligation remains outstanding.
  bool premature=bar.try_wait_for(cuda::std::move(token),cuda::std::chrono::microseconds(100));
  out[0]=premature;
  (void)bar.arrive(); // Supply the missing obligation in the SAME phase.
  bar.wait(cuda::std::move(token));
  out[1]=1;
  // Reset restored expected count=2; complete both obligations explicitly.
  bar.wait(bar.arrive(2));
  out[2]=1;
}

// Deliberately wrong: the writes AFTER arrive are not published by that arrival.
__global__ void late_publish(unsigned* out) {
  __shared__ volatile unsigned data[64];
  __shared__ Barrier bar;
  int t=threadIdx.x;
  if(t==0) init(&bar,64);
  data[t]=0;
  __syncthreads();
  auto token=bar.arrive();
  data[t]=t+1; // BUG: should precede arrive().
  bar.wait(cuda::std::move(token));
  out[t]=data[(t+32)%64]; // Cross-warp read with no happens-before edge.
}
int main(int argc,char** argv) {
  auto p=device(); if(p.major<8) return 77;
  Buffer<unsigned> out(64);
  if(argc==2 && std::strcmp(argv[1],"count")==0) {
    missing_arrival<<<1,1>>>(out.p); finish(); auto h=out.read();
    std::printf("COUNT completed_before_repair=%u after_repair=%u next_phase=%u\n",h[0],h[1],h[2]);
    require(h[0]==0 && h[1]==1 && h[2]==1,"missing_arrival_detected_and_repaired");
  } else if(argc==2 && std::strcmp(argv[1],"race")==0) {
    late_publish<<<1,64>>>(out.p); finish();
    std::puts("EXPECTED_RACE: output is not a correctness test; require racecheck diagnostic");
  } else { std::fprintf(stderr,"usage: debug count|race\n"); return 2; }
}

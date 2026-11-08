#include "common.cuh"
// Two clusters, each with two blocks. Counters never cross cluster boundaries.
extern "C" __global__ void __cluster_dims__(2,1,1) count_cluster(unsigned* counters,unsigned* tickets) {
  unsigned cluster=blockIdx.x/2,id=blockIdx.x*blockDim.x+threadIdx.x;
  tickets[id]=__nv_atomic_fetch_add(&counters[cluster],1u,__NV_ATOMIC_RELAXED,__NV_THREAD_SCOPE_CLUSTER);
}
int main() {
  auto p=device(); int supported=0; CUDA(cudaDeviceGetAttribute(&supported,cudaDevAttrClusterLaunch,0));
  if(p.major<9 || !supported) { std::puts("SKIP cluster scope requires sm_90+ and cluster launch support"); return 77; }
  Buffer<unsigned> counts(2),tickets(4*Threads);
  count_cluster<<<4,Threads>>>(counts.p,tickets.p); finish(); auto h=counts.read();
  check(h[0]==2*Threads && h[1]==2*Threads && tickets_ok(tickets.read(),2,2*Threads),"cluster_atomic_unique_tickets");
}

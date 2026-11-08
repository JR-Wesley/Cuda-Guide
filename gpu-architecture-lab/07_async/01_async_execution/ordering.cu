#include "common.cuh"
#ifndef LAB_PER_THREAD
#define LAB_PER_THREAD 0
#endif

static void ordering(bool nonblocking,bool explicit_legacy) {
    Device<unsigned long long> times(6);
    Stream a(nonblocking?cudaStreamNonBlocking:cudaStreamDefault);
    Stream b(nonblocking?cudaStreamNonBlocking:cudaStreamDefault);
    CUDA_CHECK(cudaDeviceSynchronize());
    delay_kernel<<<1,1,0,a>>>(times.p,2000000);
    CUDA_CHECK(cudaGetLastError());
    if(explicit_legacy) delay_kernel<<<1,1,0,cudaStreamLegacy>>>(times.p+2,2000000);
    else delay_kernel<<<1,1>>>(times.p+2,2000000);
    CUDA_CHECK(cudaGetLastError());
    delay_kernel<<<1,1,0,b>>>(times.p+4,2000000);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long t[6];
    CUDA_CHECK(cudaMemcpy(t,times.p,sizeof(t),cudaMemcpyDeviceToHost));
    for(int i=0;i<3;++i) require(t[2*i+1]>=t[2*i]+2000000,"invalid GPU timestamp interval");
    const bool ordered=t[1]<=t[2] && t[3]<=t[4];
    const bool required=!nonblocking && (explicit_legacy || !LAB_PER_THREAD);
    if(required) require(ordered,"legacy ordering violated");
    unsigned long long pair_overlap=0;
    for(int i=0;i<3;++i) for(int j=i+1;j<3;++j) {
        const auto lo=std::max(t[2*i],t[2*j]),hi=std::min(t[2*i+1],t[2*j+1]);
        if(hi>lo) pair_overlap+=hi-lo;
    }
    std::printf("ORDER,compiled=%s,workers=%s,middle=%s,ordered=%d,required=%d,pair_overlap_ns=%llu,PASS\n",
        LAB_PER_THREAD?"per-thread":"legacy",nonblocking?"nonblocking":"blocking",
        explicit_legacy?"explicit_legacy":"implicit_default",ordered,required,pair_overlap);
    for(int i=0;i<3;++i) std::printf("GPU_INTERVAL,kernel=%d,start_ns=%llu,end_ns=%llu\n",i,t[2*i],t[2*i+1]);
}
static void priorities_and_destroy() {
    Device<unsigned long long> times(4);
    int low,high;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&low,&high));
    cudaStream_t a,b;
    CUDA_CHECK(cudaStreamCreateWithPriority(&a,cudaStreamNonBlocking,low));
    CUDA_CHECK(cudaStreamCreateWithPriority(&b,cudaStreamNonBlocking,high));
    delay_kernel<<<1,1,0,a>>>(times.p,1000000);
    delay_kernel<<<1,1,0,b>>>(times.p+2,1000000);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long t[4];
    CUDA_CHECK(cudaMemcpy(t,times.p,sizeof(t),cudaMemcpyDeviceToHost));
    require(t[1]>=t[0]+1000000 && t[3]>=t[2]+1000000,"priority work incomplete");
    std::printf("PRIORITY,lowest=%d,highest=%d,low_start=%llu,high_start=%llu,PASS_no_order_assertion\n",low,high,t[0],t[2]);
    CUDA_CHECK(cudaStreamDestroy(a)); CUDA_CHECK(cudaStreamDestroy(b));

    Event done(cudaEventDisableTiming);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    delay_kernel<<<1,1,0,stream>>>(times.p,2000000);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(done,stream));
    CUDA_CHECK(cudaStreamDestroy(stream)); // Returns without requiring queued work to finish.
    auto status=cudaEventQuery(done);
    if(status!=cudaSuccess && status!=cudaErrorNotReady) CUDA_CHECK(status);
    std::printf("DESTROY,event_immediately_after_destroy=%s\n",cudaGetErrorName(status));
    CUDA_CHECK(cudaEventSynchronize(done)); // Buffer is kept alive until here.
    std::puts("PASS destroyed_stream_work_completed");
}
int main() {
    try {
        if(!initialize()) return 77;
        for(bool explicit_legacy:{false,true}) for(bool nonblocking:{false,true})
            ordering(nonblocking,explicit_legacy);
        priorities_and_destroy();
        std::puts("SUMMARY ordering failed=0");
    } catch(const std::exception& e) { std::fprintf(stderr,"FAIL %s\n",e.what()); return 1; }
}

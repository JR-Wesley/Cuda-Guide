#include "common.cuh"
#include <nvtx3/nvToolsExt.h>

struct CallbackState { std::atomic<int> calls{0}; };
static void CUDART_CB completed(void* p) {
    // No CUDA APIs, exceptions, or waits inside the host function.
    static_cast<CallbackState*>(p)->calls.fetch_add(1, std::memory_order_release);
}
struct Marks { Event h0,h1,k0,k1,d0,d1; };

static void run(const char* name, bool pinned, int stream_count, size_t n, int samples) {
    nvtxRangePushA(name);
    constexpr int chunks=7, steps=64;
    Host<uint32_t> input(n,pinned), output(n,pinned);
    Device<uint32_t> device(n);
    std::vector<uint32_t> expected(n);
    for(size_t i=0;i<n;++i) { input.p[i]=static_cast<uint32_t>(i); expected[i]=transform(input.p[i],steps); }
    CallbackState callback;
    std::vector<std::unique_ptr<Marks>> marks;
    for(int i=0;i<chunks;++i) marks.push_back(std::make_unique<Marks>());
    Event start,stop;
    Stream control;
    std::vector<std::unique_ptr<Stream>> streams;
    for(int i=0;i<stream_count;++i) streams.push_back(std::make_unique<Stream>());
    std::vector<double> walls;
    for(int sample=-1;sample<samples;++sample) {
        callback.calls.store(0);
        std::fill(output.p,output.p+n,0u);
        auto begin=Clock::now();
        CUDA_CHECK(cudaEventRecord(start,control));
        for(auto& s:streams) CUDA_CHECK(cudaStreamWaitEvent(*s,start,0));
        for(int c=0;c<chunks;++c) {
            const size_t lo=n*c/chunks, hi=n*(c+1)/chunks, bytes=(hi-lo)*4;
            auto s=streams[c%stream_count]->s;
            auto& m=*marks[c];
            CUDA_CHECK(cudaEventRecord(m.h0,s));
            CUDA_CHECK(cudaMemcpyAsync(device.p+lo,input.p+lo,bytes,cudaMemcpyHostToDevice,s));
            CUDA_CHECK(cudaEventRecord(m.h1,s));
            CUDA_CHECK(cudaEventRecord(m.k0,s));
            compute<<<128,256,0,s>>>(device.p+lo,hi-lo,steps);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(m.k1,s));
            CUDA_CHECK(cudaEventRecord(m.d0,s));
            CUDA_CHECK(cudaMemcpyAsync(output.p+lo,device.p+lo,bytes,cudaMemcpyDeviceToHost,s));
            CUDA_CHECK(cudaEventRecord(m.d1,s));
            // Fan-in: never time multiple workers by waiting for only one.
            CUDA_CHECK(cudaStreamWaitEvent(control,m.d1,0));
        }
        CUDA_CHECK(cudaEventRecord(stop,control));
        CUDA_CHECK(cudaLaunchHostFunc(control,completed,&callback));
        auto submitted=Clock::now();
        uint64_t cpu_checksum=1, polls=0;
        for(;;) {
            auto state=cudaEventQuery(stop);
            if(state==cudaSuccess) break;
            if(state!=cudaErrorNotReady) CUDA_CHECK(state);
            // Independent CPU work; never inspect unfinished output buffers.
            for(int j=0;j<1024;++j) cpu_checksum=cpu_checksum*6364136223846793005ull+1;
            ++polls;
            if(Clock::now()-begin>std::chrono::seconds(30)) throw std::runtime_error("poll timeout");
            std::this_thread::yield();
        }
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaStreamSynchronize(control)); // Also waits for the callback.
        auto finished=Clock::now();
        require(callback.calls.load(std::memory_order_acquire)==1,"callback count mismatch");
        for(auto& s:streams) CUDA_CHECK(cudaStreamQuery(*s));
        validate(output.p,expected);
        if(sample>=0) {
            walls.push_back(ms(begin,finished));
            std::printf("SAMPLE,%s,%d,enqueue_ms=%.6f,wall_ms=%.6f,gpu_span_ms=%.6f,polls=%llu,cpu_checksum=%llu,PASS\n",
                name,sample,ms(begin,submitted),ms(begin,finished),elapsed(start,stop),
                static_cast<unsigned long long>(polls),static_cast<unsigned long long>(cpu_checksum));
            for(int c=0;c<chunks;++c) {
                auto& m=*marks[c];
                std::printf("INTERVAL,%s,%d,chunk=%d,stream=%d,H2D,%.6f,%.6f\n",name,sample,c,c%stream_count,elapsed(start,m.h0),elapsed(start,m.h1));
                std::printf("INTERVAL,%s,%d,chunk=%d,stream=%d,KERNEL,%.6f,%.6f\n",name,sample,c,c%stream_count,elapsed(start,m.k0),elapsed(start,m.k1));
                std::printf("INTERVAL,%s,%d,chunk=%d,stream=%d,D2H,%.6f,%.6f\n",name,sample,c,c%stream_count,elapsed(start,m.d0),elapsed(start,m.d1));
            }
        }
    }
    std::sort(walls.begin(),walls.end());
    std::printf("RESULT,%s,min_wall_ms=%.6f,median_wall_ms=%.6f,max_wall_ms=%.6f,PASS\n",
        name,walls.front(),(walls[(walls.size()-1)/2]+walls[walls.size()/2])/2,walls.back());
    nvtxRangePop();
}
int main(int argc,char** argv) {
    try {
        if(argc>3) throw std::runtime_error("Usage: pipeline [samples=5] [elements=1048613]");
        int samples=argc>1?number(argv[1],31):5;
        size_t n=argc>2?number(argv[2],16777216):1048613;
        require(n>=7,"need at least seven elements");
        if(!initialize()) return 77;
        run("pinned_one_stream",true,1,n,samples);
        run("pinned_three_streams",true,3,n,samples);
        run("pageable_three_streams",false,3,n,samples);
        std::puts("SUMMARY pipeline cases=3 failed=0");
    } catch(const std::exception& e) { std::fprintf(stderr,"FAIL %s\n",e.what()); return 1; }
}

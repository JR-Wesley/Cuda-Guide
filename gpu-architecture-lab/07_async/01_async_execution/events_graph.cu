#include "common.cuh"

static void event_dependency() {
    constexpr size_t n=65573;
    Host<uint32_t> input(n),output(n);
    Device<uint32_t> data(n);
    Device<unsigned long long> tail(2);
    std::vector<uint32_t> expected(n);
    for(size_t i=0;i<n;++i) {input.p[i]=i; expected[i]=transform(i,2);}
    Event produced(cudaEventDisableTiming),copied(cudaEventDisableTiming);
    Stream producer,consumer;
    CUDA_CHECK(cudaMemcpyAsync(data.p,input.p,n*4,cudaMemcpyHostToDevice,producer));
    compute<<<32,256,0,producer>>>(data.p,n,1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(produced,producer));
    delay_kernel<<<1,1,0,producer>>>(tail.p,5000000); // Does not access data.
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamWaitEvent(consumer,produced,0));
    compute<<<32,256,0,consumer>>>(data.p,n,1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(output.p,data.p,n*4,cudaMemcpyDeviceToHost,consumer));
    CUDA_CHECK(cudaEventRecord(copied,consumer));
    // Waiting for the copy marker need not wait for unrelated producer tail work.
    CUDA_CHECK(cudaEventSynchronize(copied));
    auto state=cudaStreamQuery(producer);
    if(state!=cudaSuccess && state!=cudaErrorNotReady) CUDA_CHECK(state);
    validate(output.p,expected);
    std::printf("PASS event_dependency producer_status_when_copy_done=%s\n",cudaGetErrorName(state));
    CUDA_CHECK(cudaStreamSynchronize(producer));
    CUDA_CHECK(cudaStreamSynchronize(consumer));
}

static void graph_demo() {
    constexpr size_t n=4099;
    constexpr int kernels=8,replays=20,samples=5;
    Host<uint32_t> input(n),output(n);
    Device<uint32_t> data(n);
    std::vector<uint32_t> expected(n);
    for(size_t i=0;i<n;++i) {input.p[i]=i; expected[i]=transform(i,kernels);}
    Event start,stop;
    Stream stream;
    auto enqueue=[&] {
        CUDA_CHECK(cudaMemcpyAsync(data.p,input.p,n*4,cudaMemcpyHostToDevice,stream));
        for(int k=0;k<kernels;++k) compute<<<17,256,0,stream>>>(data.p,n,1);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(output.p,data.p,n*4,cudaMemcpyDeviceToHost,stream));
    };
    cudaGraph_t graph=nullptr;
    cudaGraphExec_t instance=nullptr;
    auto setup=Clock::now();
    CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    enqueue();
    CUDA_CHECK(cudaStreamEndCapture(stream,&graph));
    CUDA_CHECK(cudaGraphInstantiate(&instance,graph,0));
    size_t nodes=0;
    CUDA_CHECK(cudaGraphGetNodes(graph,nullptr,&nodes));
    std::printf("GRAPH_SETUP,nodes=%zu,capture_instantiate_ms=%.6f\n",nodes,ms(setup,Clock::now()));
    for(int mode=0;mode<2;++mode) {
        if(mode) CUDA_CHECK(cudaGraphLaunch(instance,stream)); else enqueue();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        validate(output.p,expected);
    }
    std::vector<double> wall[2];
    for(int sample=0;sample<samples;++sample) for(int offset=0;offset<2;++offset) {
        const int mode=(sample+offset)%2;
        auto begin=Clock::now();
        CUDA_CHECK(cudaEventRecord(start,stream));
        for(int r=0;r<replays;++r) {
            if(mode) CUDA_CHECK(cudaGraphLaunch(instance,stream)); else enqueue();
        }
        CUDA_CHECK(cudaEventRecord(stop,stream));
        auto submitted=Clock::now();
        CUDA_CHECK(cudaEventSynchronize(stop));
        auto finished=Clock::now();
        validate(output.p,expected);
        wall[mode].push_back(ms(begin,finished));
        std::printf("GRAPH_SAMPLE,%s,%d,replays=%d,enqueue_ms=%.6f,wall_ms=%.6f,gpu_span_ms=%.6f,PASS\n",
            mode?"graph":"ordinary",sample,replays,ms(begin,submitted),ms(begin,finished),elapsed(start,stop));
    }
    for(int mode=0;mode<2;++mode) {
        std::sort(wall[mode].begin(),wall[mode].end());
        std::printf("GRAPH_RESULT,%s,median_wall_ms=%.6f\n",mode?"graph":"ordinary",wall[mode][samples/2]);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGraphExecDestroy(instance)); CUDA_CHECK(cudaGraphDestroy(graph));
}
int main() {
    try {
        if(!initialize()) return 77;
        event_dependency(); graph_demo();
        std::puts("SUMMARY events_graph failed=0");
    } catch(const std::exception& e) { std::fprintf(stderr,"FAIL %s\n",e.what()); return 1; }
}

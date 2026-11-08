#pragma once
#include "common/benchmark/experiment.hpp"
#include "common/validation/compare.hpp"
#include <memory>

namespace asynchronous {
struct HostBuffer {
  float* data{};
  bool pinned;
  explicit HostBuffer(std::size_t n,bool pin):pinned(pin) {
    if(pin)CUDA_CHECK(cudaMallocHost(&data,n*sizeof(float)));
    else data=new float[n];
  }
  ~HostBuffer() {
    if(pinned)bpg::cleanup(cudaFreeHost(data),"cudaFreeHost");else delete[] data;
  }
  HostBuffer(const HostBuffer&)=delete;
  HostBuffer& operator=(const HostBuffer&)=delete;
};
struct Signal {
  cudaEvent_t value{};
  Signal(){CUDA_CHECK(cudaEventCreateWithFlags(&value,cudaEventDisableTiming));}
  ~Signal(){bpg::cleanup(cudaEventDestroy(value),"cudaEventDestroy signal");}
  Signal(const Signal&)=delete;
  Signal& operator=(const Signal&)=delete;
};
// Gate and completion records are captured by each wait at enqueue time.
// Re-recording the same event later does not retarget a previously enqueued wait.
struct Workers {
  bpg::Stream coordinator;
  Signal gate;
  std::vector<std::unique_ptr<bpg::Stream>> streams;
  std::vector<std::unique_ptr<Signal>> done;
  explicit Workers(int count) {
    for(int i=0;i<count;++i){streams.push_back(std::make_unique<bpg::Stream>());done.push_back(std::make_unique<Signal>());}
  }
  void begin() {
    CUDA_CHECK(cudaEventRecord(gate.value,coordinator.value));
    for(const auto& stream:streams)CUDA_CHECK(cudaStreamWaitEvent(stream->value,gate.value,0));
  }
  void join() {
    for(std::size_t i=0;i<streams.size();++i) {
      CUDA_CHECK(cudaEventRecord(done[i]->value,streams[i]->value));
      CUDA_CHECK(cudaStreamWaitEvent(coordinator.value,done[i]->value,0));
    }
  }
  void synchronize(){CUDA_CHECK(cudaStreamSynchronize(coordinator.value));}
};
__global__ void transform(const float* input,float* output,std::size_t n,int rounds) {
  const auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n) {
    float x=input[i];
    #pragma unroll 1
    for(int r=0;r<rounds;++r)x=fmaf(x,1.0001f,.01f);
    output[i]=x;
  }
}
inline float value(std::size_t i){return (static_cast<int>(i%257)-128)/32.0f;}
inline float reference(float x,int rounds){for(int r=0;r<rounds;++r)x=std::fma(x,1.0001f,.01f);return x;}
inline void check(const float* actual,const std::vector<float>& expected,bool exact=false) {
  for(std::size_t i=0;i<expected.size();++i) {
    const double a=actual[i],e=expected[i];
    if(!std::isfinite(a)||!std::isfinite(e)||(exact?std::memcmp(actual+i,expected.data()+i,sizeof(float))!=0:std::abs(a-e)>1e-5+1e-5*std::abs(e)))
      throw std::runtime_error("validation mismatch at "+std::to_string(i)+" actual="+bpg::decimal(a)+" expected="+bpg::decimal(e));
  }
}
inline void limits(const bpg::Options& o,int rounds=1) {
  if(o.n>(1ULL<<23)||bpg::checked_product(o.n,rounds)>(1ULL<<29))throw std::invalid_argument("n <= 2^23; CPU-reference work <= 2^29 updates");
  // Up to three full arrays: input/output/reference or transfer/reference/verification.
  if(o.n*12>128ULL*1024*1024)throw std::invalid_argument("host storage exceeds 128 MiB lab limit");
}
inline void capabilities(bpg::Fields& fields,const bpg::Device& d) {
  fields["async_engine_count"]=std::to_string(d.props.asyncEngineCount);
  fields["concurrent_kernels_capability"]=std::to_string(d.props.concurrentKernels);
  fields["stream_flags"]="cudaStreamNonBlocking";
  fields["host_allocation_timed"]="no; pin/unpin and allocation excluded";
}
}

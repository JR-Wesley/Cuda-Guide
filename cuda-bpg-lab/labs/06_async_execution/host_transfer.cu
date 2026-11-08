#include "async_common.cuh"

namespace {
bpg::CaseResult execute(const bpg::Options& o,const bpg::Device& d,const std::string& variant,bool h2d) {
  const auto bytes=o.n*sizeof(float);
  bpg::check_allocation(o,bytes);
  if(bytes>d.free_bytes/2)throw std::invalid_argument("exceeds 50% of free GPU memory");
  const bool pinned=variant=="pinned";
  asynchronous::HostBuffer host(o.n,pinned);
  bpg::DeviceBuffer device(bytes);
  bpg::Stream stream;
  std::vector<float> expected(o.n),actual(o.n);
  for(std::size_t i=0;i<o.n;++i){expected[i]=asynchronous::value(i);host.data[i]=expected[i];}
  if(!h2d) {
    CUDA_CHECK(cudaMemcpyAsync(device.data,expected.data(),bytes,cudaMemcpyHostToDevice,stream.value));
    CUDA_CHECK(cudaStreamSynchronize(stream.value));
  }
  auto reset=[&] {
    if(h2d)CUDA_CHECK(cudaMemsetAsync(device.data,0xff,bytes,stream.value));
    else std::fill(host.data,host.data+o.n,std::numeric_limits<float>::quiet_NaN());
  };
  auto copy=[&] {
    if(h2d)CUDA_CHECK(cudaMemcpyAsync(device.data,host.data,bytes,cudaMemcpyHostToDevice,stream.value));
    else CUDA_CHECK(cudaMemcpyAsync(host.data,device.data,bytes,cudaMemcpyDeviceToHost,stream.value));
  };
  auto verify=[&] {
    if(h2d) {
      CUDA_CHECK(cudaMemcpyAsync(actual.data(),device.data,bytes,cudaMemcpyDeviceToHost,stream.value));
      CUDA_CHECK(cudaStreamSynchronize(stream.value));
      asynchronous::check(actual.data(),expected,true);
    } else asynchronous::check(host.data,expected,true);
  };
  reset();copy();CUDA_CHECK(cudaStreamSynchronize(stream.value));verify();
  bpg::CaseResult result;result.variant=variant;result.requested_bytes=bytes;
  result.extra={{"direction",h2d?"h2d":"d2h"},{"host_memory",pinned?"pinned":"pageable"},
    {"allocation_bytes",std::to_string(bytes)},{"host_array_bytes",std::to_string(bytes*3)},
    {"working_set_bytes",std::to_string(bytes)},{"h2d_bytes_per_op",h2d?std::to_string(bytes):"0"},
    {"d2h_bytes_per_op",h2d?"0":std::to_string(bytes)},
    {"requested_bytes_scope","one-direction host/device transfer; not device-kernel requested bytes"},
    {"event_scope","batch of cudaMemcpyAsync on one nonblocking stream; may include host submission gaps"},
    {"e2e_scope","one cudaMemcpyAsync + stream synchronization; verification transfer and allocation excluded"}};
  asynchronous::capabilities(result.extra,d);
  bpg::Event start,stop;
  for(int r=0;r<o.repeats;++r) {
    reset();
    for(int w=0;w<o.warmup;++w)copy();
    CUDA_CHECK(cudaStreamSynchronize(stream.value));
    auto sample=bpg::measure_batch(stream.value,start,stop,o.iterations,copy);
    verify();
    reset();CUDA_CHECK(cudaStreamSynchronize(stream.value));
    const auto begin=bpg::Clock::now();copy();CUDA_CHECK(cudaStreamSynchronize(stream.value));
    sample.host_e2e_us=bpg::elapsed_us(begin,bpg::Clock::now());
    verify();result.samples.push_back(sample);
  }
  return result;
}
}
int main(int argc,char** argv) {
  try {
    bpg::Arguments args(argc,argv,{{"n","4194304"},{"direction","h2d"},{"iterations","10"},{"repeats","11"},{"warmup","3"}});
    if(args.help){args.usage("all|pageable|pinned; --direction h2d|d2h; cudaMemcpyAsync + explicit completion");return 0;}
    const auto o=args.common();asynchronous::limits(o);
    const auto direction=args.text("direction");
    if(direction!="h2d"&&direction!="d2h")throw std::invalid_argument("direction must be h2d or d2h");
    const auto variants=bpg::select_variants(o.variant,{"pageable","pinned"});
    const bpg::Device d(o.device);
    std::vector<bpg::CaseResult> results;
    for(const auto& variant:variants)results.push_back(execute(o,d,variant,direction=="h2d"));
    bpg::publish("host_transfer",o,d,results);return 0;
  } catch(const std::exception& e){std::cerr<<"FAILED (no valid performance CSV): "<<e.what()<<'\n';return 1;}
}

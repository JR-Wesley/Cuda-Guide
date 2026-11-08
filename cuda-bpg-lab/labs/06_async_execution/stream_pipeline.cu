#include "async_common.cuh"

namespace {
struct Slot {
  bpg::DeviceBuffer input,output;
  asynchronous::Signal done;
  bool used=false;
  explicit Slot(std::size_t bytes):input(bytes),output(bytes){}
};
bpg::CaseResult execute(const bpg::Options& o,const bpg::Device& d,const std::string& variant,
                        std::size_t chunk,int stream_count,int buffers,int rounds,const std::string& order,
                        const std::vector<float>& expected) {
  const bool pinned=variant.rfind("pinned_",0)==0;
  const int workers=variant.find("serial")!=std::string::npos?1:stream_count;
  const auto slot_n=std::min(chunk,o.n);
  const auto allocation=slot_n*8*buffers;
  bpg::check_allocation(o,allocation);
  if(allocation>d.free_bytes/2)throw std::invalid_argument("exceeds 50% of free GPU memory");
  const auto chunks=(o.n+chunk-1)/chunk;
  bpg::grid_for(slot_n,o.block,d);
  asynchronous::HostBuffer input(o.n,pinned),output(o.n,pinned);
  for(std::size_t i=0;i<o.n;++i)input.data[i]=asynchronous::value(i);
  asynchronous::Workers team(workers);
  std::vector<std::unique_ptr<Slot>> slots;
  for(int i=0;i<buffers;++i)slots.push_back(std::make_unique<Slot>(slot_n*4));
  auto prepare=[&] {
    team.synchronize();
    std::fill(output.data,output.data+o.n,std::numeric_limits<float>::quiet_NaN());
    for(auto& slot:slots) {
      CUDA_CHECK(cudaMemsetAsync(slot->input.data,0xff,slot_n*4,team.coordinator.value));
      CUDA_CHECK(cudaMemsetAsync(slot->output.data,0xff,slot_n*4,team.coordinator.value));
    }
  };
  auto submit=[&] {
    team.begin();
    auto stage=[&](std::size_t c,int phase) {
      const auto offset=c*chunk;
      auto& slot=*slots[c%buffers];
      const auto stream=team.streams[c%workers]->value;
      const auto count=std::min(chunk,o.n-offset);
      if(phase==0) {
        if(slot.used)CUDA_CHECK(cudaStreamWaitEvent(stream,slot.done.value,0));
        CUDA_CHECK(cudaMemcpyAsync(slot.input.data,input.data+offset,count*4,cudaMemcpyHostToDevice,stream));
      } else if(phase==1) {
        asynchronous::transform<<<(count+o.block-1)/o.block,o.block,0,stream>>>(slot.input.data,slot.output.data,count,rounds);
        CUDA_CHECK(cudaGetLastError());
      } else {
        CUDA_CHECK(cudaMemcpyAsync(output.data+offset,slot.output.data,count*4,cudaMemcpyDeviceToHost,stream));
        CUDA_CHECK(cudaEventRecord(slot.done.value,stream));slot.used=true;
      }
    };
    if(order=="interleaved") {
      for(std::size_t c=0;c<chunks;++c)for(int phase=0;phase<3;++phase)stage(c,phase);
    } else {
      // Complete submission of one buffer-sized wave before any slot can be reused.
      // Each chunk's H/K/D remain ordered within its own worker stream.
      for(std::size_t base=0;base<chunks;base+=buffers)
        for(int phase=0;phase<3;++phase)
          for(std::size_t c=base;c<std::min(chunks,base+buffers);++c)stage(c,phase);
    }
    team.join();
  };
  auto check=[&]{asynchronous::check(output.data,expected);};
  prepare();submit();team.synchronize();check();
  bpg::CaseResult result;result.variant=variant;result.requested_bytes=o.n*8;result.flops=2.0*o.n*rounds;
  result.extra=bpg::kernel_fields(asynchronous::transform,o.block);
  asynchronous::capabilities(result.extra,d);
  result.extra["chunk_elements"]=std::to_string(chunk);result.extra["chunks_per_op"]=std::to_string(chunks);
  result.extra["buffers"]=std::to_string(buffers);result.extra["streams_requested"]=std::to_string(stream_count);
  result.extra["worker_streams"]=std::to_string(workers);result.extra["rounds"]=std::to_string(rounds);
  result.extra["host_memory"]=pinned?"pinned":"pageable";
  result.extra["submission_order"]=order;
  result.extra["allocation_bytes"]=std::to_string(allocation);result.extra["host_array_bytes"]=std::to_string(o.n*12);
  result.extra["device_buffer_working_set_bytes"]=std::to_string(allocation);
  result.extra["working_set_bytes"]=std::to_string(o.n*8);
  result.extra["working_set_scope"]="logical host input/output per op; device ring capacity reported separately";
  result.extra["h2d_bytes_per_op"]=std::to_string(o.n*4);result.extra["d2h_bytes_per_op"]=std::to_string(o.n*4);
  result.extra["kernel_requested_bytes_per_op"]=std::to_string(o.n*8);
  result.extra["requested_bytes_scope"]="H2D + D2H bytes; kernel global I/O is a separate memory path";
  result.extra["event_scope"]="complete per-op chunk pipeline + gate/join; global join between ops";
  result.extra["e2e_scope"]="one complete pipeline submission + coordinator synchronization; allocation/pinning/reset/reference excluded";
  bpg::Event start,stop;
  for(int r=0;r<o.repeats;++r) {
    prepare();for(int w=0;w<o.warmup;++w)submit();team.synchronize();
    auto sample=bpg::measure_batch(team.coordinator.value,start,stop,o.iterations,submit);check();
    prepare();team.synchronize();
    const auto begin=bpg::Clock::now();submit();team.synchronize();
    sample.host_e2e_us=bpg::elapsed_us(begin,bpg::Clock::now());check();result.samples.push_back(sample);
  }
  return result;
}
}
int main(int argc,char** argv) {
  try {
    bpg::Arguments args(argc,argv,{{"n","1048576"},{"chunk","65536"},{"streams","3"},{"buffers","3"},{"rounds","128"},
                                  {"warmup","3"},{"repeats","11"},{"iterations","10"},{"order","interleaved"}});
    if(args.help){args.usage("all|pageable_serial|pinned_serial|pinned_pipeline|pageable_pipeline; --order interleaved|phased; independent buffers/streams");return 0;}
    const auto o=args.common();
    const auto order=args.text("order");
    if(order!="interleaved"&&order!="phased")throw std::invalid_argument("order must be interleaved or phased");
    const int streams=static_cast<int>(args.integer("streams",1,8)),buffers=static_cast<int>(args.integer("buffers",1,8)),
              rounds=static_cast<int>(args.integer("rounds",1,4096));
    const auto chunk=args.integer("chunk",1,1ULL<<23);
    asynchronous::limits(o,rounds);
    if((o.n+chunk-1)/chunk>4096)throw std::invalid_argument("at most 4096 chunks/op to bound host submission and trace size");
    const auto variants=bpg::select_variants(o.variant,{"pageable_serial","pinned_serial","pinned_pipeline","pageable_pipeline"});
    const bpg::Device d(o.device);
    std::vector<float> expected(o.n);for(std::size_t i=0;i<o.n;++i)expected[i]=asynchronous::reference(asynchronous::value(i),rounds);
    std::vector<bpg::CaseResult> results;
    for(const auto& variant:variants)results.push_back(execute(o,d,variant,chunk,streams,buffers,rounds,order,expected));
    bpg::publish("stream_pipeline",o,d,results);return 0;
  } catch(const std::exception& e){std::cerr<<"FAILED (no valid performance CSV): "<<e.what()<<'\n';return 1;}
}

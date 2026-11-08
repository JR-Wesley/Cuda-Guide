#include "async_common.cuh"

namespace {
bpg::CaseResult execute(const bpg::Options& o,const bpg::Device& d,const std::string& variant,int jobs,int rounds,
                        const std::vector<float>& input,const std::vector<float>& expected) {
  const auto bytes=o.n*4;
  bpg::check_allocation(o,bytes*2);
  if(bytes*2>d.free_bytes/2)throw std::invalid_argument("exceeds 50% of free GPU memory");
  const int workers=variant=="serial"?1:jobs;
  asynchronous::Workers team(workers);
  bpg::DeviceBuffer in(bytes),out(bytes);
  std::vector<float> actual(o.n);
  auto upload=[&]{CUDA_CHECK(cudaMemcpyAsync(in.data,input.data(),bytes,cudaMemcpyHostToDevice,team.coordinator.value));};
  auto prepare=[&]{CUDA_CHECK(cudaMemsetAsync(out.data,0xff,bytes,team.coordinator.value));};
  auto submit=[&] {
    team.begin();
    for(int job=0;job<jobs;++job) {
      const auto offset=o.n*job/jobs,end=o.n*(job+1)/jobs,count=end-offset;
      const auto stream=team.streams[job%workers]->value;
      asynchronous::transform<<<(count+o.block-1)/o.block,o.block,0,stream>>>(in.data+offset,out.data+offset,count,rounds);
      CUDA_CHECK(cudaGetLastError());
    }
    team.join();
  };
  auto download=[&] {
    CUDA_CHECK(cudaMemcpyAsync(actual.data(),out.data,bytes,cudaMemcpyDeviceToHost,team.coordinator.value));
    team.synchronize();
  };
  auto check=[&]{bpg::validate(actual,expected,false,1e-5,1e-5);};
  bpg::CaseResult result;result.variant=variant;result.requested_bytes=bytes*2;result.flops=2.0*o.n*rounds;
  result.extra=bpg::kernel_fields(asynchronous::transform,o.block);
  asynchronous::capabilities(result.extra,d);
  result.extra["jobs"]=std::to_string(jobs);result.extra["rounds"]=std::to_string(rounds);
  result.extra["worker_streams"]=std::to_string(workers);result.extra["allocation_bytes"]=std::to_string(bytes*2);
  result.extra["working_set_bytes"]=std::to_string(bytes*2);
  result.extra["min_elements_per_job"]=std::to_string(o.n/jobs);
  result.extra["max_elements_per_job"]=std::to_string((o.n+jobs-1)/jobs);
  result.extra["event_scope"]="jobs kernels with input resident; coordinator gate/join; no host transfers";
  result.extra["e2e_scope"]="full input H2D + same jobs group + full output D2H + sync; allocation/reference/reset excluded";
  result.samples=bpg::collect(o,team.coordinator.value,prepare,upload,submit,download,check);
  return result;
}
}
int main(int argc,char** argv) {
  try {
    bpg::Arguments args(argc,argv,{{"n","4096"},{"jobs","4"},{"rounds","4096"}});
    if(args.help){args.usage("all|serial|concurrent; n is total elements across all jobs; both variants launch the same jobs");return 0;}
    const auto o=args.common();
    const int jobs=static_cast<int>(args.integer("jobs",1,8)),rounds=static_cast<int>(args.integer("rounds",1,16384));
    if(o.n<static_cast<std::size_t>(jobs))throw std::invalid_argument("jobs <= n; each job must be nonempty");
    asynchronous::limits(o,rounds);
    const bpg::Device d(o.device);
    bpg::grid_for((o.n+jobs-1)/jobs,o.block,d);
    std::vector<std::string> supported{"serial"};
    if(d.props.concurrentKernels)supported.push_back("concurrent");
    else std::cerr<<"Not Tested: concurrent kernel capability unavailable; serial baseline remains supported\n";
    const auto variants=bpg::select_variants(o.variant,supported);
    std::vector<float> input(o.n),expected(o.n);
    for(std::size_t i=0;i<o.n;++i){input[i]=asynchronous::value(i);expected[i]=asynchronous::reference(input[i],rounds);}
    std::vector<bpg::CaseResult> results;
    for(const auto& variant:variants)results.push_back(execute(o,d,variant,jobs,rounds,input,expected));
    bpg::publish("concurrent_kernels",o,d,results);return 0;
  }catch(const std::exception& e){std::cerr<<"FAILED (no valid performance CSV): "<<e.what()<<'\n';return 1;}
}

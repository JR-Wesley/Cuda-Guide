#include "word_benchmark.cuh"

namespace {
using instructions::Word;
template<int Chains,int Unroll>
__global__ void update_chain(const Word* input,Word* output,std::size_t n,int rounds) {
  const auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=n)return;
  Word x[Chains];
  #pragma unroll
  for(int c=0;c<Chains;++c)x[c]=input[i]+c*0x9e3779b9u;
  #pragma unroll Unroll
  for(int r=0;r<rounds;++r) {
    #pragma unroll
    for(int c=0;c<Chains;++c) {
      x[c]=x[c]*1664525u+1013904223u;
      x[c]^=x[c]>>13;
    }
  }
  #pragma unroll
  for(int c=0;c<Chains;++c)output[static_cast<std::size_t>(c)*n+i]=x[c];
}
template<int Chains>
std::vector<bpg::CaseResult> execute(const bpg::Options& o,const bpg::Device& d,int rounds) {
  const auto variants=bpg::select_variants(o.variant,{"u1","u4","u8"});
  const auto grid=bpg::grid_for(o.n,o.block,d);
  std::vector<Word> input(o.n),expected(o.n*Chains);
  for(std::size_t i=0;i<o.n;++i) {
    input[i]=instructions::seed(i);
    for(int c=0;c<Chains;++c) {
      Word state=input[i]+c*0x9e3779b9u;
      for(int r=0;r<rounds;++r) {
        state=state*1664525u+1013904223u;
        state^=state>>13;
      }
      expected[c*o.n+i]=state;
    }
  }
  std::vector<bpg::CaseResult> results;
  for(const auto& v:variants) {
    auto kernel = v=="u1" ? update_chain<Chains,1> : v=="u4" ? update_chain<Chains,4> : update_chain<Chains,8>;
    bpg::CaseResult r;r.variant=v;
    r.extra={{"rounds",std::to_string(rounds)},{"chains",std::to_string(Chains)},
      {"unroll_factor",v.substr(1)},{"logical_updates",std::to_string(o.n*Chains*rounds)},
      {"grid",std::to_string(grid)},{"integer_semantics","uint32 multiplication/addition wraps modulo 2^32"}};
    auto launch = [&](Word* in,Word* out,cudaStream_t stream) {kernel<<<grid,o.block,0,stream>>>(in,out,o.n,rounds);};
    results.push_back(instructions::run(o,d,input,expected,std::move(r),kernel,launch));
  }
  return results;
}
}
int main(int argc,char** argv) {
  try {
    bpg::Arguments args(argc,argv,{{"n","65536"},{"rounds","128"},{"chains","4"}});
    if(args.help) {args.usage("all|u1|u4|u8; --chains 1 or 4; nonmultiple rounds allowed");return 0;}
    const auto o=args.common();
    const int rounds=static_cast<int>(args.integer("rounds",1,1024)),chains=static_cast<int>(args.integer("chains",1,4));
    if(chains!=1&&chains!=4)throw std::invalid_argument("chains must be 1 or 4");
    instructions::cap_work(o,bpg::checked_product(o.n,rounds*chains));
    const bpg::Device d(o.device);
    const auto results=chains==1?execute<1>(o,d,rounds):execute<4>(o,d,rounds);
    bpg::publish("loop_unroll",o,d,results);return 0;
  } catch(const std::exception& e) {std::cerr<<"FAILED (no valid performance CSV): "<<e.what()<<'\n';return 1;}
}

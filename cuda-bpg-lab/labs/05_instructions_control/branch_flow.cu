#include "word_benchmark.cuh"

namespace {
using instructions::Word;
__host__ __device__ inline Word path_a(Word x) {x=x*1664525u+1013904223u;return x^(x>>13);}
__host__ __device__ inline Word path_b(Word x) {x=x*22695477u+1u;return x^(x<<7);}
template<bool Short>
__global__ void branch_kernel(const Word* input,Word* output,std::size_t n,int rounds) {
  const auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=n)return;
  const Word packed=input[i];Word x=packed>>1;
  if constexpr (Short) {
    if(packed&1)x=path_b(x);else x=path_a(x);
  } else {
    if(packed&1) {
      #pragma unroll 1
      for(int r=0;r<rounds;++r)x=path_b(x);
    } else {
      #pragma unroll 1
      for(int r=0;r<rounds;++r)x=path_a(x);
    }
  }
  output[i]=x;
}
std::vector<bpg::CaseResult> execute(const bpg::Options& o,const bpg::Device& d,int rounds,bool is_short) {
  const auto variants=bpg::select_variants(o.variant,{"uniform","interleaved"});
  const auto grid=bpg::grid_for(o.n,o.block,d);
  std::vector<Word> canonical(o.n),reference(o.n);
  for(std::size_t task=0;task<o.n;++task) {
    const Word flag=(task/32)&1;
    canonical[task]=((instructions::seed(task)&0x7fffffffu)<<1)|flag;
    Word x=canonical[task]>>1;
    for(int r=0;r<rounds;++r)x=flag?path_b(x):path_a(x);
    reference[task]=x;
  }
  std::vector<bpg::CaseResult> results;
  for(const auto& v:variants) {
    std::vector<std::size_t> order;order.reserve(o.n);
    if(v=="uniform") {for(std::size_t i=0;i<o.n;++i)order.push_back(i);}
    else for(std::size_t base=0;base<o.n;base+=64)for(std::size_t lane=0;lane<32;++lane)for(std::size_t half=0;half<2;++half) {
      const auto task=base+half*32+lane;if(task<o.n)order.push_back(task);
    }
    std::vector<Word> input(o.n),expected(o.n);
    std::size_t mixed=0,path_b_count=0;
    for(std::size_t i=0;i<o.n;++i) {input[i]=canonical[order[i]];expected[i]=reference[order[i]];path_b_count+=input[i]&1;}
    for(std::size_t first=0;first<o.n;first+=32) {
      const auto end=std::min(first+32,o.n);bool zero=false,one=false;
      for(auto i=first;i<end;++i) {one|=(input[i]&1)!=0;zero|=(input[i]&1)==0;}
      mixed+=zero&&one;
    }
    auto kernel=is_short?branch_kernel<true>:branch_kernel<false>;
    bpg::CaseResult r;r.variant=v;
    r.extra={{"rounds",std::to_string(rounds)},{"body",is_short?"short":"loop"},
      {"logical_updates",std::to_string(o.n*rounds)},{"path_b_tasks",std::to_string(path_b_count)},
      {"model_mixed_warps",std::to_string(mixed)},{"grid",std::to_string(grid)},
      {"layout_semantics","same task multiset permuted within groups of 64; contiguous global input/output"}};
    auto launch=[&](Word* in,Word* out,cudaStream_t stream) {kernel<<<grid,o.block,0,stream>>>(in,out,o.n,rounds);};
    results.push_back(instructions::run(o,d,input,expected,std::move(r),kernel,launch));
  }
  return results;
}
}
int main(int argc,char** argv) {
  try {
    bpg::Arguments args(argc,argv,{{"n","65536"},{"rounds","128"},{"body","loop"}});
    if(args.help) {args.usage("all|uniform|interleaved; --body loop|short; short requires --rounds 1; block multiple of 32");return 0;}
    const auto o=args.common();
    if(o.block%32)throw std::invalid_argument("block must be a multiple of 32");
    const int rounds=static_cast<int>(args.integer("rounds",1,1024));
    const auto body=args.text("body");
    if(body!="short"&&body!="loop")throw std::invalid_argument("invalid body");
    if(body=="short"&&rounds!=1)throw std::invalid_argument("short body requires rounds=1");
    instructions::cap_work(o,bpg::checked_product(o.n,rounds));
    const bpg::Device d(o.device);
    const auto results=execute(o,d,rounds,body=="short");
    bpg::publish("branch_flow",o,d,results);return 0;
  } catch(const std::exception& e) {std::cerr<<"FAILED (no valid performance CSV): "<<e.what()<<'\n';return 1;}
}

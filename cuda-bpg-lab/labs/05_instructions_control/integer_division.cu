#include "word_benchmark.cuh"

namespace {
using instructions::Word;
template<bool Signed, unsigned Divisor, bool Shift = false>
__global__ void divide_chain(const Word* input, Word* output, std::size_t n, int rounds, Word divisor) {
  const auto i = static_cast<std::size_t>(blockIdx.x)*blockDim.x + threadIdx.x;
  if (i >= n) return;
  Word state = input[i];
  #pragma unroll 1
  for (int r=0; r<rounds; ++r) {
    Word q;
    if constexpr (Shift) {
      if constexpr (Signed) {
        const bool negative = (state & 0x80000000u) != 0;
        const Word magnitude = negative ? 0u-state : state;
        q = magnitude >> 3;
        if (negative) q = 0u-q;
      } else q = state >> 3;
    } else if constexpr (Signed) {
      q = static_cast<Word>(instructions::signed_word(state) / static_cast<std::int32_t>(Divisor ? Divisor : divisor));
    } else q = state / (Divisor ? Divisor : divisor);
    state = instructions::mix(state) ^ q;
  }
  output[i] = state;
}
std::vector<bpg::CaseResult> execute(const bpg::Options& o, const bpg::Device& d, int rounds, Word divisor, bool is_signed) {
  std::vector<std::string> available{"runtime"};
  if (divisor==7 || divisor==8) available.push_back("constant");
  if (divisor==8) available.push_back("manual_shift");
  const auto variants=bpg::select_variants(o.variant,available);
  const auto grid=bpg::grid_for(o.n,o.block,d);
  std::vector<Word> input(o.n),expected(o.n);
  for (std::size_t i=0;i<o.n;++i) {
    input[i]=instructions::seed(i); Word state=input[i];
    for (int r=0;r<rounds;++r) {
      const std::int64_t value = state <= 0x7fffffffu ? state : static_cast<std::int64_t>(state)-4294967296LL;
      const Word q = is_signed ? static_cast<Word>(value / divisor) : state / divisor;
      state = instructions::mix(state) ^ q;
    }
    expected[i]=state;
  }
  std::vector<bpg::CaseResult> results;
  for (const auto& v: variants) {
    auto kernel = is_signed ? divide_chain<true,0> : divide_chain<false,0>;
    if (v=="constant" && divisor==7) kernel = is_signed ? divide_chain<true,7> : divide_chain<false,7>;
    if (v=="constant" && divisor==8) kernel = is_signed ? divide_chain<true,8> : divide_chain<false,8>;
    if (v=="manual_shift") kernel = is_signed ? divide_chain<true,8,true> : divide_chain<false,8,true>;
    bpg::CaseResult result;result.variant=v;
    result.extra={{"rounds",std::to_string(rounds)},{"divisor",std::to_string(divisor)},
      {"signed_input",is_signed?"1":"0"},{"logical_divisions",std::to_string(o.n*rounds)},
      {"dependency_chains_per_thread","1"},{"grid",std::to_string(grid)},
      {"integer_semantics","signed quotient truncates toward zero; mixing wraps modulo 2^32"}};
    auto launch = [&](Word* in,Word* out,cudaStream_t stream) {kernel<<<grid,o.block,0,stream>>>(in,out,o.n,rounds,divisor);};
    results.push_back(instructions::run(o,d,input,expected,std::move(result),kernel,launch));
  }
  return results;
}
}
int main(int argc,char** argv) {
  try {
    bpg::Arguments args(argc,argv,{{"n","65536"},{"rounds","128"},{"divisor","7"},{"signed","0"}});
    if (args.help) {args.usage("all|runtime|constant (d=7/8)|manual_shift (d=8); --signed 0/1; positive divisor only");return 0;}
    const auto o=args.common();
    const int rounds=static_cast<int>(args.integer("rounds",1,1024));
    const auto divisor=static_cast<Word>(args.integer("divisor",1,0x7fffffffu));
    const bool is_signed=args.integer("signed",0,1)!=0;
    instructions::cap_work(o,bpg::checked_product(o.n,rounds));
    const bpg::Device d(o.device);
    const auto results=execute(o,d,rounds,divisor,is_signed);
    bpg::publish("integer_division",o,d,results);return 0;
  } catch(const std::exception& e) {std::cerr<<"FAILED (no valid performance CSV): "<<e.what()<<'\n';return 1;}
}

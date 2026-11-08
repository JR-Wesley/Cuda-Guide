#include "common/benchmark/experiment.hpp"
#include <cmath>
#include <cstdint>

namespace {
template<int T,bool Coalesced>
__global__ void gemm_direct(const float* __restrict__ a,const float* __restrict__ b,float* __restrict__ c,
                            int m,int n,int depth) {
  const int row=Coalesced?blockIdx.y*T+threadIdx.y:blockIdx.x*T+threadIdx.x;
  const int col=Coalesced?blockIdx.x*T+threadIdx.x:blockIdx.y*T+threadIdx.y;
  if(row>=m||col>=n)return;
  float acc=0;
  #pragma unroll 1
  for(int k=0;k<depth;++k)acc=fmaf(a[row*depth+k],b[k*n+col],acc);
  c[row*n+col]=acc;
}
template<int T,int Rows,int Unroll>
__global__ void gemm_tiled(const float* __restrict__ a,const float* __restrict__ b,float* __restrict__ c,
                           int m,int n,int depth) {
  __shared__ float as[Rows*T][T],bs[T][T];
  const int tx=threadIdx.x,ty=threadIdx.y,base_row=blockIdx.y*(Rows*T)+ty,col=blockIdx.x*T+tx;
  float acc[Rows]={};
  for(int base=0;base<depth;base+=T) {
    #pragma unroll
    for(int r=0;r<Rows;++r) {
      const int row=base_row+r*T;
      as[ty+r*T][tx]=(row<m&&base+tx<depth)?a[row*depth+base+tx]:0.0f;
    }
    bs[ty][tx]=(base+ty<depth&&col<n)?b[(base+ty)*n+col]:0.0f;
    __syncthreads();
    #pragma unroll Unroll
    for(int k=0;k<T;++k) {
      const float bv=bs[k][tx];
      #pragma unroll
      for(int r=0;r<Rows;++r)acc[r]=fmaf(as[ty+r*T][k],bv,acc[r]);
    }
    // All threads finish reading the current tiles before the next load overwrites them.
    __syncthreads();
  }
  #pragma unroll
  for(int r=0;r<Rows;++r)if(base_row+r*T<m&&col<n)c[(base_row+r*T)*n+col]=acc[r];
}
struct Reference {
  std::vector<float> a,b,actual;
  std::vector<double> expected,sum_abs;
  Reference(int m,int n,int depth,const std::string& pattern)
      :a(static_cast<std::size_t>(m)*depth),b(static_cast<std::size_t>(depth)*n),
       actual(static_cast<std::size_t>(m)*n),expected(actual.size(),0),sum_abs(actual.size(),0) {
    std::uint32_t seed=20261024;
    auto random=[&]{seed=seed*1664525u+1013904223u;return (static_cast<int>((seed>>8)%2001)-1000)/1000.0f;};
    for(auto& x:a)x=random();for(auto& x:b)x=random();
    if(pattern=="identity") {
      for(int k=0;k<depth;++k)for(int j=0;j<n;++j)b[k*n+j]=k==j?1.0f:0.0f;
    } else if(pattern=="cancellation") {
      for(int i=0;i<m;++i)for(int k=0;k<depth;++k)
        a[i*depth+k]=(1+(i%13)+(k/2)%11)/37.0f;
      for(int k=0;k<depth;++k)for(int j=0;j<n;++j)b[k*n+j]=(k%2?-1.0f:1.0f)*(1+j%17)/17.0f;
    }
    // All outputs, FP64 products/sums; contiguous j permits efficient CPU reference.
    for(int i=0;i<m;++i)for(int k=0;k<depth;++k) {
      const double av=a[i*depth+k];
      for(int j=0;j<n;++j) {
        const double product=av*static_cast<double>(b[k*n+j]);
        expected[i*n+j]+=product;sum_abs[i*n+j]+=std::abs(product);
      }
    }
  }
};
struct Error {
  double abs=0,scaled=0,relative=0;
  void check(const Reference& ref,int depth) {
    const double ku=depth*std::ldexp(1.0,-24),gamma=ku/(1-ku);
    for(std::size_t i=0;i<ref.actual.size();++i) {
      const double got=ref.actual[i],want=ref.expected[i];
      const double delta=std::abs(got-want),limit=2*gamma*ref.sum_abs[i]+1e-7;
      if(!std::isfinite(got)||!std::isfinite(want)||delta>limit)
        throw std::runtime_error("GEMM validation mismatch at "+std::to_string(i)+" actual="+bpg::decimal(got)+
                                 " expected="+bpg::decimal(want)+" tolerance="+bpg::decimal(limit));
      abs=std::max(abs,delta);scaled=std::max(scaled,delta/limit);
      relative=std::max(relative,delta/std::max(std::abs(want),1e-12));
    }
  }
};
std::size_t first_warp_sectors(int m,int n,int depth,int tile,bool coalesced,char operand) {
  std::set<std::size_t> sectors;
  for(int lane=0;lane<32;++lane) {
    const int tx=lane%tile,ty=lane/tile,row=coalesced?ty:tx,col=coalesced?tx:ty;
    if(row>=m||col>=n)continue;
    const auto index=operand=='A'?static_cast<std::size_t>(row)*depth:
                     operand=='B'?static_cast<std::size_t>(col):static_cast<std::size_t>(row)*n+col;
    sectors.insert(index*4/32);
  }
  return sectors.size();
}
template<int T,int Rows>
std::vector<bpg::CaseResult> execute(const bpg::Options& o,const bpg::Device& d,int m,int n,int depth,
                                    const std::string& pattern,const std::vector<std::string>& variants) {
  const auto mk=static_cast<std::size_t>(m)*depth,kn=static_cast<std::size_t>(depth)*n,mn=o.n;
  const auto allocation=4*(mk+kn+mn);
  if(allocation>d.free_bytes/2)throw std::invalid_argument("exceeds 50% of free GPU memory");
  if(T>d.props.maxThreadsDim[0]||T>d.props.maxThreadsDim[1]||T*T>d.props.maxThreadsPerBlock)
    throw std::invalid_argument("tile thread dimensions exceed device limits");
  Reference reference(m,n,depth,pattern);
  bpg::DeviceBuffer a(mk*4),b(kn*4),c(mn*4);
  bpg::Stream stream;
  auto upload=[&] {
    CUDA_CHECK(cudaMemcpyAsync(a.data,reference.a.data(),mk*4,cudaMemcpyHostToDevice,stream.value));
    CUDA_CHECK(cudaMemcpyAsync(b.data,reference.b.data(),kn*4,cudaMemcpyHostToDevice,stream.value));
  };
  auto prepare=[&]{CUDA_CHECK(cudaMemsetAsync(c.data,0xff,mn*4,stream.value));};
  auto download=[&] {
    CUDA_CHECK(cudaMemcpyAsync(reference.actual.data(),c.data,mn*4,cudaMemcpyDeviceToHost,stream.value));
    CUDA_CHECK(cudaStreamSynchronize(stream.value));
  };
  std::vector<bpg::CaseResult> results;
  for(const auto& variant:variants) {
    const bool direct=variant=="naive"||variant=="coalesced",coalesced=variant!="naive";
    const int accumulators=direct||variant=="shared_tiled"?1:Rows,bm=T*accumulators;
    const dim3 block(T,T),grid(coalesced?(n+T-1)/T:(m+T-1)/T,coalesced?(m+bm-1)/bm:(n+T-1)/T);
    if(grid.x>static_cast<unsigned>(d.props.maxGridSize[0])||grid.y>static_cast<unsigned>(d.props.maxGridSize[1]))
      throw std::invalid_argument("GEMM grid exceeds device limit");
    bpg::CaseResult r;r.variant=variant;r.flops=2.0*mn*depth;
    const auto loads=direct?2*mn*depth:mk*((n+T-1)/T)+kn*((m+bm-1)/bm);
    r.requested_bytes=4*(loads+mn);
    if(variant=="naive")r.extra=bpg::kernel_fields(gemm_direct<T,false>,T*T);
    else if(variant=="coalesced")r.extra=bpg::kernel_fields(gemm_direct<T,true>,T*T);
    else if(variant=="shared_tiled")r.extra=bpg::kernel_fields(gemm_tiled<T,1,1>,T*T);
    else if(variant=="register_tiled")r.extra=bpg::kernel_fields(gemm_tiled<T,Rows,1>,T*T);
    else r.extra=bpg::kernel_fields(gemm_tiled<T,Rows,T>,T*T);
    r.extra["matrix_m"]=std::to_string(m);r.extra["matrix_n"]=std::to_string(n);r.extra["matrix_k"]=std::to_string(depth);
    r.extra["tile"]=std::to_string(T);r.extra["input_pattern"]=pattern;
    r.extra["grid_x"]=std::to_string(grid.x);r.extra["grid_y"]=std::to_string(grid.y);
    r.extra["block_x"]=std::to_string(T);r.extra["block_y"]=std::to_string(T);
    r.extra["output_tile_rows"]=std::to_string(bm);r.extra["output_tile_cols"]=std::to_string(T);
    r.extra["accumulators_per_thread"]=std::to_string(accumulators);
    r.extra["reg_rows_requested"]=std::to_string(Rows);
    r.extra["k_tile_unroll_factor"]=std::to_string(variant=="unrolled"?T:1);
    r.extra["allocation_bytes"]=std::to_string(allocation);r.extra["working_set_bytes"]=std::to_string(allocation);
    r.extra["host_array_bytes"]=std::to_string(4*(mk+kn)+20*mn);
    r.extra["model_global_load_elements"]=std::to_string(loads);
    r.extra["model_executed_fma"]=std::to_string(direct?mn*depth:static_cast<std::size_t>(grid.x)*grid.y*bm*T*((depth+T-1)/T)*T);
    r.extra["model_global_requested_AI_FLOP_per_byte"]=bpg::decimal(r.flops/r.requested_bytes);
    r.extra["minimum_data_AI_FLOP_per_byte"]=bpg::decimal(r.flops/allocation);
    if(direct)for(char operand:std::string("ABC"))r.extra[std::string("model_first_warp_")+operand+"_sectors_k0"]=
      std::to_string(first_warp_sectors(m,n,depth,T,coalesced,operand));
    r.extra["requested_bytes_scope"]="source-level lane global requests; broadcast/cache/DRAM traffic not measured";
    r.extra["event_scope"]="one GEMM kernel; inputs resident; batch span may include launch gaps";
    r.extra["e2e_scope"]="A/B pageable H2D + GEMM + C D2H + sync; allocations/reset/FP64 reference excluded";
    r.extra["validation_mode"]="all outputs FP64 reference; abs error <= 2*gamma_K*sum_abs_products + 1e-7";
    auto launch=[&] {
      if(variant=="naive")gemm_direct<T,false><<<grid,block,0,stream.value>>>(a.data,b.data,c.data,m,n,depth);
      else if(variant=="coalesced")gemm_direct<T,true><<<grid,block,0,stream.value>>>(a.data,b.data,c.data,m,n,depth);
      else if(variant=="shared_tiled")gemm_tiled<T,1,1><<<grid,block,0,stream.value>>>(a.data,b.data,c.data,m,n,depth);
      else if(variant=="register_tiled")gemm_tiled<T,Rows,1><<<grid,block,0,stream.value>>>(a.data,b.data,c.data,m,n,depth);
      else gemm_tiled<T,Rows,T><<<grid,block,0,stream.value>>>(a.data,b.data,c.data,m,n,depth);
      CUDA_CHECK(cudaGetLastError());
    };
    Error errors;
    r.samples=bpg::collect(o,stream.value,prepare,upload,launch,download,[&]{errors.check(reference,depth);});
    r.extra["validation_max_abs_error"]=bpg::decimal(errors.abs);
    r.extra["validation_max_scaled_error"]=bpg::decimal(errors.scaled);
    r.extra["validation_max_relative_error_denominator_floor_1e_minus_12"]=bpg::decimal(errors.relative);
    results.push_back(std::move(r));
  }
  return results;
}
}
int main(int argc,char** argv) {
  try {
    bpg::Arguments args(argc,argv,{{"rows","256"},{"cols","256"},{"depth","256"},{"tile","16"},{"pattern","random"},{"reg-rows","4"},
                                  {"warmup","3"},{"repeats","7"},{"iterations","5"}},{"n"});
    if(args.help){args.usage("all|naive|coalesced|shared_tiled|register_tiled|unrolled; row-major C=A*B; block=tile*tile; reg-rows=2 or 4");return 0;}
    auto o=args.common();
    const int m=args.integer("rows",1,4096),n=args.integer("cols",1,4096),depth=args.integer("depth",1,4096),tile=args.integer("tile",16,32);
    if((tile!=16&&tile!=32)||o.block!=tile*tile)throw std::invalid_argument("tile must be 16 or 32; --block must equal tile*tile");
    const int rows=args.integer("reg-rows",2,4);
    if(rows!=2&&rows!=4)throw std::invalid_argument("reg-rows must be 2 or 4");
    const auto pattern=args.text("pattern");
    if(pattern!="random"&&pattern!="identity"&&pattern!="cancellation")throw std::invalid_argument("pattern must be random, identity or cancellation");
    o.n=bpg::checked_product(m,n);
    if(o.n*depth>(1ULL<<30))throw std::invalid_argument("full FP64 CPU reference limited to 2^30 multiply-accumulate pairs");
    const auto mk=static_cast<std::size_t>(m)*depth,kn=static_cast<std::size_t>(depth)*n;
    bpg::check_allocation(o,4*(mk+kn+o.n));
    if(4*(mk+kn)+20*o.n>256ULL*1024*1024)throw std::invalid_argument("host arrays exceed 256 MiB lab limit");
    const auto variants=bpg::select_variants(o.variant,{"naive","coalesced","shared_tiled","register_tiled","unrolled"});
    const bpg::Device d(o.device);
    auto results=tile==16?(rows==2?execute<16,2>(o,d,m,n,depth,pattern,variants):execute<16,4>(o,d,m,n,depth,pattern,variants)):
                          (rows==2?execute<32,2>(o,d,m,n,depth,pattern,variants):execute<32,4>(o,d,m,n,depth,pattern,variants));
    bpg::publish("gemm",o,d,results);return 0;
  }catch(const std::exception& e){std::cerr<<"FAILED (no valid performance CSV): "<<e.what()<<'\n';return 1;}
}

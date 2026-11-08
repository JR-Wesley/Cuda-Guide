#include "ops.cuh"

// Single-operation kernels use the same wrappers as the executable.
// These are compiled for inspection, never launched with multiple threads.
#define LOAD(op) extern "C" __global__ void probe_load_##op( \
    const unsigned* src, unsigned* dst) { *dst = read_value<Load::op>(src); }
LOAD(plain) LOAD(ldg) LOAD(ca) LOAD(cg) LOAD(cs) LOAD(lu) LOAD(cv)
#undef LOAD
extern "C" __global__ void probe_load_restrict(
    const unsigned* __restrict__ src, unsigned* __restrict__ dst) { *dst = *src; }
#define STORE(op) extern "C" __global__ void probe_store_##op( \
    unsigned* dst, unsigned value) { write_value<Store::op>(dst,value); }
STORE(plain) STORE(wb) STORE(cg) STORE(cs) STORE(wt)
#undef STORE
extern "C" __global__ void probe_float4_ca(const float4* src, float4* dst) {
    *dst = read_value<Load::ca>(src);
}
extern "C" __global__ void probe_half2_ldg(const __half2* src, __half2* dst) {
    *dst = read_value<Load::ldg>(src);
}
extern "C" __global__ void probe_bfloat162_cs(__nv_bfloat162* dst, __nv_bfloat162 value) {
    write_value<Store::cs>(dst,value);
}

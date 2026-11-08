#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

enum class Load { plain, ldg, ca, cg, cs, lu, cv };
enum class Store { plain, wb, cg, cs, wt };

// Addresses must designate correctly aligned global memory objects of type T.
// ldg additionally requires the source data to stay read-only for the kernel.
template <Load Op, class T>
__device__ __forceinline__ T read_value(const T* p) {
    if constexpr (Op == Load::plain) return *p;
    else if constexpr (Op == Load::ldg) return __ldg(p);
    else if constexpr (Op == Load::ca) return __ldca(p);
    else if constexpr (Op == Load::cg) return __ldcg(p);
    else if constexpr (Op == Load::cs) return __ldcs(p);
    else if constexpr (Op == Load::lu) return __ldlu(p);
    else return __ldcv(p);
}

template <Store Op, class T>
__device__ __forceinline__ void write_value(T* p, T value) {
    if constexpr (Op == Store::plain) *p = value;
    else if constexpr (Op == Store::wb) __stwb(p, value);
    else if constexpr (Op == Store::cg) __stcg(p, value);
    else if constexpr (Op == Store::cs) __stcs(p, value);
    else __stwt(p, value);
}

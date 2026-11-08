#include "ops.cuh"
struct Pair { float x, y; };
#ifndef PROBE_TYPE
#define PROBE_TYPE unsigned
#endif
using T = PROBE_TYPE;
// Compile each candidate independently so one unsupported overload does not
// hide support for another. Ordinary dereference is the control.
extern "C" __global__ void candidate(const T* src, T* dst) {
#if PROBE_OP == 0
    *dst = *src;
#elif PROBE_OP == 1
    *dst = __ldg(src);
#elif PROBE_OP == 2
    *dst = __ldca(src);
#else
    __stwb(dst, *src);
#endif
}

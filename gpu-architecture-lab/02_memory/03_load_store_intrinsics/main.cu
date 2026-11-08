#include "ops.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cudaError_t e, const char* expr, int line) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(expr) + ":" + std::to_string(line) +
                                 ": " + cudaGetErrorString(e));
}
#define CUDA_CHECK(expr) check((expr), #expr, __LINE__)

template <class T> struct DeviceBuffer {
    T* p = nullptr;
    explicit DeviceBuffer(size_t n) { CUDA_CHECK(cudaMalloc(&p, n * sizeof(T))); }
    ~DeviceBuffer() { if (p) cudaFree(p); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// Small finite values avoid NaN canonicalization, rounding and signed overflow
// obscuring a transport test. Each vector lane has a different value.
template <class T> T value_at(size_t i) { return static_cast<T>(1 + i % 97); }
#define VECTOR2(T) template <> T value_at<T>(size_t i) { \
    T v{}; v.x = value_at<decltype(v.x)>(2*i); \
    v.y = value_at<decltype(v.y)>(2*i+1); return v; }
#define VECTOR4(T) template <> T value_at<T>(size_t i) { \
    T v{}; v.x = value_at<decltype(v.x)>(4*i); \
    v.y = value_at<decltype(v.y)>(4*i+1); \
    v.z = value_at<decltype(v.z)>(4*i+2); \
    v.w = value_at<decltype(v.w)>(4*i+3); return v; }
VECTOR2(char2) VECTOR4(char4) VECTOR2(uchar2) VECTOR4(uchar4)
VECTOR2(short2) VECTOR4(short4) VECTOR2(ushort2) VECTOR4(ushort4)
VECTOR2(int2) VECTOR4(int4) VECTOR2(uint2) VECTOR4(uint4)
VECTOR2(longlong2) VECTOR2(ulonglong2)
VECTOR2(float2) VECTOR4(float4) VECTOR2(double2)
#undef VECTOR2
#undef VECTOR4
template <> __half value_at<__half>(size_t i) {
    return __float2half_rn(static_cast<float>(1 + i % 97) / 4);
}
template <> __half2 value_at<__half2>(size_t i) {
    return __halves2half2(value_at<__half>(2*i), value_at<__half>(2*i+1));
}
template <> __nv_bfloat16 value_at<__nv_bfloat16>(size_t i) {
    return __float2bfloat16_rn(static_cast<float>(1 + i % 97) / 4);
}
template <> __nv_bfloat162 value_at<__nv_bfloat162>(size_t i) {
    return __halves2bfloat162(value_at<__nv_bfloat16>(2*i),
                             value_at<__nv_bfloat16>(2*i+1));
}

template <Load L, Store S, class T>
__global__ void copy_values(const T* src, T* dst, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(blockDim.x) * gridDim.x)
        write_value<S>(dst + i, read_value<L>(src + i));
}

static int passed = 0, skipped = 0;
template <class T>
static void equal_bytes(const T* actual, const T* expected, size_t n,
                        const std::string& label) {
    // The selected scalar/vector types have no padding; bitwise equality is
    // stricter than a tolerance-based floating-point comparison here.
    for (size_t i = 0; i < n; ++i)
        if (std::memcmp(actual + i, expected + i, sizeof(T)) != 0)
            throw std::runtime_error(label + ": mismatch at element " + std::to_string(i));
}

template <class T>
static void type_matrix(const char* name, size_t n) {
    constexpr size_t guard = 4;
    const size_t total = n + 2 * guard;
    std::vector<T> input(total), expected(total), actual(total);
    for (size_t i = 0; i < total; ++i) input[i] = value_at<T>(i);
    // Guard bytes are compared as bytes only; the kernel never reads them.
    std::memset(expected.data(), 0xa5, total * sizeof(T));
    std::memcpy(expected.data() + guard, input.data() + guard, n * sizeof(T));
    DeviceBuffer<T> src(total), dst(total);
    CUDA_CHECK(cudaMemcpy(src.p, input.data(), total*sizeof(T), cudaMemcpyHostToDevice));
    const int blocks = static_cast<int>((n + 127) / 128);
    auto run = [&](auto launch, const char* op) {
        CUDA_CHECK(cudaMemset(dst.p, 0xa5, total*sizeof(T)));
        launch(src.p + guard, dst.p + guard);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(actual.data(), dst.p, total*sizeof(T), cudaMemcpyDeviceToHost));
        equal_bytes(actual.data(), expected.data(), total, std::string(name) + "/" + op);
        ++passed;
    };
#define LOAD(op) run([&](const T* s, T* d) { \
    copy_values<Load::op, Store::plain><<<blocks,128>>>(s,d,n); }, "load/" #op)
    LOAD(plain); LOAD(ldg); LOAD(ca); LOAD(cg); LOAD(cs); LOAD(lu); LOAD(cv);
#undef LOAD
#define STORE(op) run([&](const T* s, T* d) { \
    copy_values<Load::plain, Store::op><<<blocks,128>>>(s,d,n); }, "store/" #op)
    STORE(plain); STORE(wb); STORE(cg); STORE(cs); STORE(wt);
#undef STORE
    CUDA_CHECK(cudaMemcpy(actual.data(), src.p, total*sizeof(T), cudaMemcpyDeviceToHost));
    equal_bytes(actual.data(), input.data(), total, std::string(name) + "/source unchanged");
    std::printf("PASS type=%-18s sizeof=%2zu alignof=%2zu n=%zu variants=12 guards=OK source=OK\n",
                name, sizeof(T), alignof(T), n);
}

__host__ __device__ unsigned pattern(size_t i, unsigned generation) {
    return static_cast<unsigned>(i) * 1664525u + 1013904223u + generation * 747796405u;
}
__global__ void produce(unsigned* data, size_t n, unsigned generation) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) __stwb(data + i, pattern(i, generation));
}

// The allocation changes between kernels, but is read-only during each ldg
// kernel. Stream ordering, not the intrinsic, establishes the dependency.
static void ordered_generations(size_t n) {
    DeviceBuffer<unsigned> src(n), dst(n);
    std::vector<unsigned> expected(n), actual(n);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const int blocks = static_cast<int>((n + 127) / 128);
    for (unsigned generation = 1; generation <= 3; ++generation) {
        produce<<<blocks,128,0,stream>>>(src.p,n,generation);
        CUDA_CHECK(cudaGetLastError());
        copy_values<Load::ldg,Store::plain><<<blocks,128,0,stream>>>(src.p,dst.p,n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(actual.data(),dst.p,n*sizeof(unsigned),cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; ++i) expected[i] = pattern(i,generation);
        equal_bytes(actual.data(),expected.data(),n,"ordered generation");
        ++passed;
        std::printf("PASS ordered __stwb -> __ldg generation=%u n=%zu\n",generation,n);
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// CPU/GPU ownership is handed off only at kernel completion. No CPU polling,
// no concurrently modified ldg source, and no claim that cv/wt supplies fences.
static void mapped_handoff(const cudaDeviceProp& prop) {
    if (!prop.canMapHostMemory) {
        std::puts("SKIP mapped handoff: canMapHostMemory=0");
        ++skipped;
        return;
    }
    constexpr size_t n = 1027;
    unsigned *host = nullptr, *gpu = nullptr;
    CUDA_CHECK(cudaHostAlloc(&host,n*sizeof(unsigned),cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&gpu,host,0));
    DeviceBuffer<unsigned> src(n), dst(n);
    std::vector<unsigned> expected(n), actual(n);
    for (unsigned generation = 1; generation <= 2; ++generation) {
        for (size_t i = 0; i < n; ++i) host[i] = expected[i] = pattern(i,generation);
        copy_values<Load::cv,Store::plain><<<9,128>>>(gpu,dst.p,n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(actual.data(),dst.p,n*sizeof(unsigned),cudaMemcpyDeviceToHost));
        equal_bytes(actual.data(),expected.data(),n,"mapped __ldcv");
        ++passed;
        for (size_t i = 0; i < n; ++i) expected[i] = pattern(i,generation + 100);
        CUDA_CHECK(cudaMemcpy(src.p,expected.data(),n*sizeof(unsigned),cudaMemcpyHostToDevice));
        copy_values<Load::plain,Store::wt><<<9,128>>>(src.p,gpu,n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        equal_bytes(host,expected.data(),n,"mapped __stwt");
        ++passed;
        std::printf("PASS mapped __ldcv + __stwt generation=%u n=%zu\n",generation,n);
    }
    CUDA_CHECK(cudaFreeHost(host));
}

int main(int argc, char** argv) {
    try {
        if (argc > 2) throw std::runtime_error("usage: intrinsics [elements:1..1048576]");
        size_t n = 4099;
        if (argc == 2) {
            const std::string arg = argv[1];
            if (arg.empty() || arg.find_first_not_of("0123456789") != std::string::npos)
                throw std::runtime_error("elements must be a positive integer");
            n = std::stoull(arg);
            if (n == 0 || n > 1048576) throw std::runtime_error("elements must be 1..1048576");
        }
        // Set before CUDA context initialization for mapped-host allocation.
        CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
        int driver = 0, runtime = 0;
        CUDA_CHECK(cudaDriverGetVersion(&driver));
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
        std::printf("GPU=%s sm_%d%d SMs=%d driver=%d runtime=%d canMapHostMemory=%d\n",
                    prop.name,prop.major,prop.minor,prop.multiProcessorCount,
                    driver,runtime,prop.canMapHostMemory);
#define TYPE(T) type_matrix<T>(#T,n);
#include "types.inc"
#undef TYPE
        ordered_generations(n);
        mapped_handoff(prop);
        std::printf("SUMMARY passed=%d failed=0 skipped=%d\n",passed,skipped);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr,"FAIL: %s\nSUMMARY passed=%d failed=1 skipped=%d\n",
                     e.what(),passed,skipped);
        return 1;
    }
}

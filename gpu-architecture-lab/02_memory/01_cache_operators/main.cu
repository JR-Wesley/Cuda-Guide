#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cudaError_t error, const char* expr, int line) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(expr) + ":" + std::to_string(line) +
                                 ": " + cudaGetErrorString(error));
}
#define CUDA_CHECK(expr) check((expr), #expr, __LINE__)

enum class Load { default_op, ca, cg, cs, lu, cv };
enum class Store { default_op, wb, cg, cs, wt };

// volatile here is a compiler qualifier, NOT the PTX .volatile memory qualifier.
// The memory clobber describes hidden memory effects; neither creates GPU fences.
template <Load Op>
__device__ __forceinline__ uint32_t load(const uint32_t* pointer) {
    uint32_t value;
    const auto address = __cvta_generic_to_global(pointer);
#define LD(op) asm volatile("ld.global" op ".u32 %0, [%1];" \
                           : "=r"(value) : "l"(address) : "memory")
    if constexpr (Op == Load::default_op) { LD(""); }
    if constexpr (Op == Load::ca) { LD(".ca"); }
    if constexpr (Op == Load::cg) { LD(".cg"); }
    if constexpr (Op == Load::cs) { LD(".cs"); }
    if constexpr (Op == Load::lu) { LD(".lu"); }
    if constexpr (Op == Load::cv) { LD(".cv"); }
#undef LD
    return value;
}

template <Store Op>
__device__ __forceinline__ void store(uint32_t* pointer, uint32_t value) {
    const auto address = __cvta_generic_to_global(pointer);
#define ST(op) asm volatile("st.global" op ".u32 [%0], %1;" \
                           : : "l"(address), "r"(value) : "memory")
    if constexpr (Op == Store::default_op) { ST(""); }
    if constexpr (Op == Store::wb) { ST(".wb"); }
    if constexpr (Op == Store::cg) { ST(".cg"); }
    if constexpr (Op == Store::cs) { ST(".cs"); }
    if constexpr (Op == Store::wt) { ST(".wt"); }
#undef ST
}

__host__ __device__ uint32_t pattern(size_t i, uint32_t generation) {
    // Exact unsigned arithmetic; wraparound is intentional and defined.
    return static_cast<uint32_t>(i) * 1664525u + 1013904223u + generation * 747796405u;
}

template <Store Op>
__global__ void write_stream(uint32_t* data, size_t count, uint32_t generation) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += static_cast<size_t>(gridDim.x) * blockDim.x)
        store<Op>(data + i, pattern(i, generation));
}

template <Load Op>
__global__ void read_stream(const uint32_t* data, uint32_t* sums, size_t count) {
    const size_t rank = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t sum = 0;
    for (size_t i = rank; i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x)
        sum += load<Op>(data + i);
    sums[rank] = sum;
}

template <Load Op>
__global__ void read_reuse(const uint32_t* data, uint32_t* sums, int passes) {
    const size_t base = static_cast<size_t>(blockIdx.x) * blockDim.x * 8;
    uint32_t index = static_cast<uint32_t>(base + threadIdx.x);
    uint32_t sum = 0;
#pragma unroll 1
    for (int pass = 0; pass < passes; ++pass) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            // The next address depends on the loaded value. asm volatile alone
            // cannot prevent ptxas from hoisting repeated invariant loads.
            index = load<Op>(data + index);
            sum += index;
        }
    }
    sums[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

__global__ void initialize_ring(uint32_t* data) {
    const uint32_t base = blockIdx.x * blockDim.x * 8;
    for (uint32_t j = 0; j < 8; ++j)
        data[base + j * blockDim.x + threadIdx.x] =
            base + ((j + 1) % 8) * blockDim.x + threadIdx.x;
}

// A separate dependent grid, in the same stream, validates each store policy
// with .ca loads. Kernel ordering supplies visibility, not the cache operator.
__global__ void verify_store(const uint32_t* data, size_t count,
                             uint32_t generation, unsigned* errors) {
    unsigned local = 0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += static_cast<size_t>(gridDim.x) * blockDim.x)
        local += load<Load::ca>(data + i) != pattern(i, generation);
    if (local) atomicAdd(errors, local);
}

static std::vector<uint32_t> expected_stream(size_t count, int threads, uint32_t gen) {
    std::vector<uint32_t> expected(threads, 0);
    for (int t = 0; t < threads; ++t)
        for (size_t i = t; i < count; i += threads) expected[t] += pattern(i, gen);
    return expected;
}

static void verify_sums(uint32_t* sums, const std::vector<uint32_t>& expected) {
    std::vector<uint32_t> actual(expected.size());
    CUDA_CHECK(cudaMemcpy(actual.data(), sums, actual.size() * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < actual.size(); ++i)
        if (actual[i] != expected[i])
            throw std::runtime_error("checksum mismatch at thread " + std::to_string(i));
}

struct Variant {
    const char* name;
    std::function<void(uint32_t)> launch;
    std::function<void(uint32_t)> verify;
    std::vector<float> milliseconds;
};

// Warm up every variant; rotate the starting variant each sample. Validation
// and CPU reference construction are outside the CUDA event interval.
static void benchmark(const char* scenario, double logical_bytes, int samples,
                      std::vector<Variant>& variants) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (auto& v : variants) {
        for (int warm = 0; warm < 3; ++warm) v.launch(0);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        v.verify(0);
    }
    for (int sample = 0; sample < samples; ++sample) {
        for (size_t offset = 0; offset < variants.size(); ++offset) {
            const size_t index = (sample + offset) % variants.size();
            auto& v = variants[index];
            const uint32_t generation = 1 + sample * variants.size() + offset;
            CUDA_CHECK(cudaEventRecord(start));
            v.launch(generation);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            v.milliseconds.push_back(ms);
            v.verify(generation);
            std::printf("SAMPLE,%s,%s,%d,%.6f\n", scenario, v.name, sample, ms);
        }
    }
    for (auto& v : variants) {
        std::sort(v.milliseconds.begin(), v.milliseconds.end());
        const size_t mid = v.milliseconds.size() / 2;
        const float median = v.milliseconds.size() % 2 ? v.milliseconds[mid] :
            (v.milliseconds[mid - 1] + v.milliseconds[mid]) * 0.5f;
        const double gbps = logical_bytes / (median * 1e6);
        std::printf("RESULT,%s,%s,%d,%.6f,%.6f,%.6f,%.3f,PASS\n", scenario,
                    v.name, samples, v.milliseconds.front(), median,
                    v.milliseconds.back(), gbps);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

static std::vector<Variant> load_variants(const uint32_t* data, uint32_t* sums,
                                         size_t count, int blocks, int passes,
                                         bool reuse, const std::vector<uint32_t>& expected) {
    std::vector<Variant> variants;
#define ADD_LOAD(op) variants.push_back({#op, [=](uint32_t) { \
    if (reuse) read_reuse<Load::op><<<blocks, 256>>>(data, sums, passes); \
    else read_stream<Load::op><<<blocks, 256>>>(data, sums, count); \
}, [=](uint32_t) { verify_sums(sums, expected); }, {}})
    ADD_LOAD(default_op); ADD_LOAD(ca); ADD_LOAD(cg);
    ADD_LOAD(cs); ADD_LOAD(lu); ADD_LOAD(cv);
#undef ADD_LOAD
    return variants;
}

static std::vector<Variant> store_variants(uint32_t* data, size_t count, int blocks,
                                          const std::function<void(uint32_t)>& verify) {
    std::vector<Variant> variants;
#define ADD_STORE(op) variants.push_back({#op, [=](uint32_t gen) { \
    write_stream<Store::op><<<blocks, 256>>>(data, count, gen); \
}, verify, {}})
    ADD_STORE(default_op); ADD_STORE(wb); ADD_STORE(cg); ADD_STORE(cs); ADD_STORE(wt);
#undef ADD_STORE
    return variants;
}

static int integer(const char* text, int upper) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (!*text || *end || value < 1 || value > upper)
        throw std::runtime_error("invalid positive integer argument");
    return static_cast<int>(value);
}

int main(int argc, char** argv) {
    try {
        if (argc > 4 || (argc > 1 && std::string(argv[1]) == "--help")) {
            std::printf("Usage: %s [stream_MiB=64 (1..1024)] [samples=9 (1..99)] "
                        "[reuse_passes=64 (1..4096)]\n", argv[0]);
            return argc > 4 ? 2 : 0;
        }
        const int mib = argc > 1 ? integer(argv[1], 1024) : 64;
        const int samples = argc > 2 ? integer(argv[2], 99) : 9;
        const int passes = argc > 3 ? integer(argv[3], 4096) : 64;
        // Set mapping flags before creating allocations or launching work.
        const auto init = cudaSetDeviceFlags(cudaDeviceMapHost);
        if (init != cudaSuccess) {
            std::printf("SKIP GPU initialization: %s\n", cudaGetErrorString(init));
            return 77;
        }
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        int driver = 0, runtime = 0;
        CUDA_CHECK(cudaDriverGetVersion(&driver));
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
        std::printf("GPU=%s cc=%d.%d SMs=%d L2_bytes=%d driver=%d runtime=%d "
                    "canMapHostMemory=%d integrated=%d\n", prop.name, prop.major, prop.minor,
                    prop.multiProcessorCount, prop.l2CacheSize, driver, runtime,
                    prop.canMapHostMemory, prop.integrated);
        const int stream_blocks = prop.multiProcessorCount * 4;
        const int reuse_blocks = prop.multiProcessorCount;
        // Deliberately non-multiple of warp/block size: exercise the tail path.
        const size_t count = static_cast<size_t>(mib) * 1024 * 1024 / 4 + 37;
        const size_t reuse_count = static_cast<size_t>(reuse_blocks) * 256 * 8;
        const size_t allocation_count = std::max(count, reuse_count);
        std::printf("CONFIG stream_bytes=%zu reuse_bytes=%zu tile_bytes=8192 "
                    "stream_blocks=%d reuse_blocks=%d threads=256 passes=%d samples=%d\n",
                    count * 4, reuse_count * 4, stream_blocks, reuse_blocks, passes, samples);
        std::puts("FORMAT RESULT,scenario,operator,samples,min_ms,median_ms,max_ms,logical_GBps,status");
        if (count * 4 <= static_cast<size_t>(prop.l2CacheSize))
            std::puts("NOTE streaming working set fits in L2; this is not a DRAM bandwidth test");
        uint32_t *data = nullptr, *sums = nullptr;
        unsigned* errors = nullptr;
        CUDA_CHECK(cudaMalloc(&data, allocation_count * 4));
        CUDA_CHECK(cudaMalloc(&sums, static_cast<size_t>(stream_blocks) * 256 * 4));
        CUDA_CHECK(cudaMalloc(&errors, sizeof(unsigned)));
        initialize_ring<<<reuse_blocks, 256>>>(data);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<uint32_t> reuse_expected(reuse_blocks * 256, 0);
        for (int b = 0; b < reuse_blocks; ++b)
            for (int t = 0; t < 256; ++t)
                for (int j = 0; j < 8; ++j)
                    reuse_expected[b * 256 + t] +=
                        static_cast<uint32_t>(b * 2048 + j * 256 + t) *
                        static_cast<uint32_t>(passes);
        auto reuse = load_variants(data, sums, reuse_count, reuse_blocks, passes,
                                   true, reuse_expected);
        benchmark("device_reuse", static_cast<double>(reuse_count) * 4 * passes, samples, reuse);
        write_stream<Store::default_op><<<stream_blocks, 256>>>(data, count, 0);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto stream_expected = expected_stream(count, stream_blocks * 256, 0);
        auto stream = load_variants(data, sums, count, stream_blocks, 1, false, stream_expected);
        benchmark("device_stream_load", static_cast<double>(count) * 4, samples, stream);
        const auto verify = [=](uint32_t gen) {
            CUDA_CHECK(cudaMemset(errors, 0, sizeof(unsigned)));
            verify_store<<<stream_blocks, 256>>>(data, count, gen, errors);
            CUDA_CHECK(cudaGetLastError());
            unsigned actual = 0;
            CUDA_CHECK(cudaMemcpy(&actual, errors, sizeof(actual), cudaMemcpyDeviceToHost));
            if (actual) throw std::runtime_error("store mismatch count=" + std::to_string(actual));
        };
        auto writes = store_variants(data, count, stream_blocks, verify);
        benchmark("device_stream_store", static_cast<double>(count) * 4, samples, writes);

        int skipped = 0;
        if (!prop.canMapHostMemory) {
            std::puts("SKIP mapped_host reason=canMapHostMemory_is_zero");
            skipped = 1;
        } else {
            constexpr size_t mapped_count = 65536 + 37;
            uint32_t *host = nullptr, *mapped = nullptr;
            CUDA_CHECK(cudaHostAlloc(&host, mapped_count * 4, cudaHostAllocMapped));
            CUDA_CHECK(cudaHostGetDevicePointer(&mapped, host, 0));
            // CPU writes only when GPU is idle. No concurrent CPU/GPU polling.
            for (uint32_t phase = 1; phase <= 2; ++phase) {
                for (size_t i = 0; i < mapped_count; ++i) host[i] = pattern(i, phase);
                auto expected = expected_stream(mapped_count, stream_blocks * 256, phase);
                auto loads = load_variants(mapped, sums, mapped_count, stream_blocks, 1, false, expected);
                for (auto& v : loads) {
                    v.launch(phase);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaDeviceSynchronize());
                    v.verify(phase);
                    std::printf("CHECK,mapped_load,%s,phase=%u,PASS\n", v.name, phase);
                }
            }
            auto mapped_writes = store_variants(mapped, mapped_count, stream_blocks,
                [=](uint32_t gen) {
                    for (size_t i = 0; i < mapped_count; ++i)
                        if (host[i] != pattern(i, gen))
                            throw std::runtime_error("mapped store mismatch at " + std::to_string(i));
                });
            uint32_t gen = 3;
            for (auto& v : mapped_writes) {
                v.launch(gen);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                v.verify(gen++);
                std::printf("CHECK,mapped_store,%s,PASS\n", v.name);
            }
            CUDA_CHECK(cudaFreeHost(host));
        }
        CUDA_CHECK(cudaFree(errors));
        CUDA_CHECK(cudaFree(sums));
        CUDA_CHECK(cudaFree(data));
        std::printf("SUMMARY failed=0 skipped=%d\n", skipped);
        return skipped ? 77 : 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}

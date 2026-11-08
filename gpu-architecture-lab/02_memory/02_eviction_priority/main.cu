#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cudaError_t e, const char* call, int line) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(call) + ":" +
        std::to_string(line) + ": " + cudaGetErrorString(e));
}
#define CUDA_CHECK(call) check((call), #call, __LINE__)
enum class Hint { normal, first, last, unchanged, no_allocate };

// Use the L1-qualified instruction family, without mixing .ca/.cg/.cs syntax.
template <Hint H, bool Generic = false>
__device__ __forceinline__ uint32_t read_hint(const uint32_t* p) {
    const auto address = Generic ? reinterpret_cast<unsigned long long>(p) :
                                  __cvta_generic_to_global(p);
    uint32_t value;
#define LOAD(h) do { \
    if constexpr (Generic) \
        asm volatile("ld.L1::" h ".u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory"); \
    else \
        asm volatile("ld.global.L1::" h ".u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory"); \
} while (0)
    if constexpr (H == Hint::normal) LOAD("evict_normal");
    if constexpr (H == Hint::first) LOAD("evict_first");
    if constexpr (H == Hint::last) LOAD("evict_last");
    if constexpr (H == Hint::unchanged) LOAD("evict_unchanged");
    if constexpr (H == Hint::no_allocate) LOAD("no_allocate");
#undef LOAD
    return value;
}

template <Hint H, bool Generic = false>
__device__ __forceinline__ void write_hint(uint32_t* p, uint32_t value) {
    const auto address = Generic ? reinterpret_cast<unsigned long long>(p) :
                                  __cvta_generic_to_global(p);
#define STORE(h) do { \
    if constexpr (Generic) \
        asm volatile("st.L1::" h ".u32 [%0], %1;" : : "l"(address), "r"(value) : "memory"); \
    else \
        asm volatile("st.global.L1::" h ".u32 [%0], %1;" : : "l"(address), "r"(value) : "memory"); \
} while (0)
    if constexpr (H == Hint::normal) STORE("evict_normal");
    if constexpr (H == Hint::first) STORE("evict_first");
    if constexpr (H == Hint::last) STORE("evict_last");
    if constexpr (H == Hint::unchanged) STORE("evict_unchanged");
    if constexpr (H == Hint::no_allocate) STORE("no_allocate");
#undef STORE
}

__host__ __device__ uint32_t pattern(uint32_t i) { return i * 1664525u + 1013904223u; }

template <Hint H, bool Generic>
__global__ void copy_hint(const uint32_t* input, uint32_t* output, uint32_t n, uint32_t tag) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        write_hint<H, Generic>(output + i, read_hint<H, Generic>(input + i) ^ tag);
}

struct Observation {
    unsigned long long cycles;
    uint32_t sum;
    uint32_t final_index;
    uint32_t sm;
};

template <Hint Prime, Hint Retouch, Hint Noise, bool WithNoise>
__global__ void hot_probe(const uint32_t* hot, const uint32_t* noise, const uint32_t* scrub,
                          uint32_t slots, uint32_t noise_words, uint32_t scrub_words,
                          Observation* output) {
    // Exactly one block of one warp: all stages use the same SM and its L1.
    const uint32_t lane = threadIdx.x;
    uint32_t sum = 0, index = lane;
    for (uint32_t i = lane; i < scrub_words; i += 32)
        sum += read_hint<Hint::normal>(scrub + i);
    __syncwarp();
    // A value-dependent ring prevents repeated loads being hoisted by ptxas.
#pragma unroll 1
    for (uint32_t i = 0; i < slots; ++i) {
        index = read_hint<Prime>(hot + index);
        sum += index;
    }
    __syncwarp();
#pragma unroll 1
    for (uint32_t i = 0; i < slots; ++i) {
        index = read_hint<Retouch>(hot + index);
        sum += index;
    }
    __syncwarp();
    if constexpr (WithNoise) {
        for (uint32_t i = lane; i < noise_words; i += 32)
            sum += read_hint<Noise>(noise + i);
    }
    __syncwarp();
    const unsigned long long start = clock64();
    // Keep the measurement policy constant and preserve priority on accesses.
#pragma unroll 1
    for (uint32_t i = 0; i < slots; ++i) {
        index = read_hint<Hint::unchanged>(hot + index);
        sum += index;
    }
    __syncwarp(); // Includes completion of the final load; cost is in the interval.
    const unsigned long long end = clock64();
    uint32_t sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    output[lane] = {end - start, sum, index, sm};
}

static int integer(const char* text, int upper) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (!*text || *end || value < 1 || value > upper)
        throw std::runtime_error("arguments must be positive integers within their documented limits");
    return static_cast<int>(value);
}

struct Case {
    const char* name;
    bool noise;
    std::function<void()> launch;
    std::vector<unsigned long long> cycles;
};

int main(int argc, char** argv) {
    try {
        if (argc > 4 || (argc > 1 && std::string(argv[1]) == "--help")) {
            std::printf("Usage: %s [samples=11 (1..101)] [noise_KiB=256 (1..4096)] "
                        "[hot_KiB=16 (1..128)]\n", argv[0]);
            return argc > 4 ? 2 : 0;
        }
        const int samples = argc > 1 ? integer(argv[1], 101) : 11;
        const uint32_t noise_words = (argc > 2 ? integer(argv[2], 4096) : 256) * 256;
        const uint32_t hot_words = (argc > 3 ? integer(argv[3], 128) : 16) * 256;
        constexpr uint32_t scrub_words = 512 * 256;
        const uint32_t slots = hot_words / 32;
        int count = 0;
        const auto status = cudaGetDeviceCount(&count);
        if (status != cudaSuccess || count == 0) {
            std::printf("SKIP GPU unavailable: %s\n", cudaGetErrorString(status));
            return 77;
        }
        CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        if (prop.major < 7) { std::puts("SKIP requires sm_70+"); return 77; }
        int driver = 0, runtime = 0;
        CUDA_CHECK(cudaDriverGetVersion(&driver));
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
        std::printf("GPU=%s cc=%d.%d SMs=%d L2_bytes=%d driver=%d runtime=%d\n",
                    prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                    prop.l2CacheSize, driver, runtime);
        std::printf("CONFIG samples=%d hot_bytes=%u noise_bytes=%u scrub_bytes=%u "
                    "slots=%u grid=1 block=32\n", samples, hot_words * 4,
                    noise_words * 4, scrub_words * 4, slots);

        // Functional test: every hint, both global and generic-global addresses.
        constexpr uint32_t n = 10037;
        uint32_t *input, *output;
        CUDA_CHECK(cudaMalloc(&input, n * 4));
        CUDA_CHECK(cudaMalloc(&output, n * 4));
        std::vector<uint32_t> host(n), actual(n);
        for (uint32_t i = 0; i < n; ++i) host[i] = pattern(i);
        CUDA_CHECK(cudaMemcpy(input, host.data(), n * 4, cudaMemcpyHostToDevice));
        uint32_t tag = 0;
#define COPY_CASE(h, generic) do { \
    ++tag; \
    copy_hint<Hint::h, generic><<<7, 128>>>(input, output, n, tag); \
    CUDA_CHECK(cudaGetLastError()); \
    CUDA_CHECK(cudaMemcpy(actual.data(), output, n * 4, cudaMemcpyDeviceToHost)); \
    for (uint32_t i = 0; i < n; ++i) \
        if (actual[i] != (host[i] ^ tag)) throw std::runtime_error("copy mismatch"); \
    std::printf("CHECK,%s,%s,10037_elements,PASS\n", #h, generic ? "generic_global" : "global"); \
} while (0)
#define BOTH(h) COPY_CASE(h, false); COPY_CASE(h, true)
        BOTH(normal); BOTH(first); BOTH(last); BOTH(unchanged); BOTH(no_allocate);
#undef BOTH
#undef COPY_CASE
        CUDA_CHECK(cudaFree(input));
        CUDA_CHECK(cudaFree(output));

        uint32_t *hot, *noise, *scrub;
        Observation* observations;
        CUDA_CHECK(cudaMalloc(&hot, hot_words * 4));
        CUDA_CHECK(cudaMalloc(&noise, noise_words * 4));
        CUDA_CHECK(cudaMalloc(&scrub, scrub_words * 4));
        CUDA_CHECK(cudaMalloc(&observations, 32 * sizeof(Observation)));
        host.resize(std::max({hot_words, noise_words, scrub_words}));
        for (uint32_t i = 0; i < hot_words; ++i) host[i] = (i + 32) % hot_words;
        CUDA_CHECK(cudaMemcpy(hot, host.data(), hot_words * 4, cudaMemcpyHostToDevice));
        for (uint32_t i = 0; i < host.size(); ++i) host[i] = pattern(i);
        CUDA_CHECK(cudaMemcpy(noise, host.data(), noise_words * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(scrub, host.data(), scrub_words * 4, cudaMemcpyHostToDevice));
        uint32_t expected[2][32]{};
        for (uint32_t t = 0; t < 32; ++t) {
            uint32_t sum = 0;
            for (uint32_t i = t; i < scrub_words; i += 32) sum += pattern(i);
            for (uint32_t i = t; i < hot_words; i += 32) sum += i * 3u;
            expected[0][t] = sum;
            for (uint32_t i = t; i < noise_words; i += 32) sum += pattern(i);
            expected[1][t] = sum;
        }
        std::vector<Case> cases;
#define ADD(name, prime, retouch, pressure, enabled) \
    cases.push_back({name, enabled, [=] { \
        hot_probe<Hint::prime, Hint::retouch, Hint::pressure, enabled><<<1, 32>>>( \
            hot, noise, scrub, slots, noise_words, scrub_words, observations); \
    }, {}})
        ADD("no_noise", normal, unchanged, normal, false);
        ADD("baseline", normal, unchanged, normal, true);
        ADD("hot_first", first, unchanged, normal, true);
        ADD("hot_last_unchanged", last, unchanged, normal, true);
        ADD("hot_last_normal", last, normal, normal, true);
        ADD("hot_first_normal", first, normal, normal, true);
        ADD("noise_first", normal, unchanged, first, true);
        ADD("noise_last", normal, unchanged, last, true);
        ADD("noise_unchanged", normal, unchanged, unchanged, true);
        ADD("noise_no_allocate", normal, unchanged, no_allocate, true);
        ADD("last_plus_noise_first", last, unchanged, first, true);
        ADD("last_plus_noise_no_allocate", last, unchanged, no_allocate, true);
#undef ADD
        auto run = [&](Case& c, int sample) {
            c.launch();
            CUDA_CHECK(cudaGetLastError());
            Observation values[32];
            CUDA_CHECK(cudaMemcpy(values, observations, sizeof(values), cudaMemcpyDeviceToHost));
            for (uint32_t t = 0; t < 32; ++t) {
                if (values[t].sum != expected[c.noise][t] || values[t].final_index != t ||
                    values[t].sm != values[0].sm || values[t].cycles == 0)
                    throw std::runtime_error(std::string(c.name) + ": observation mismatch");
            }
            if (sample >= 0) {
                c.cycles.push_back(values[0].cycles);
                std::printf("SAMPLE,%s,%d,sm=%u,probe_cycles=%llu\n", c.name, sample,
                            values[0].sm, values[0].cycles);
            }
        };
        for (auto& c : cases) run(c, -1);
        for (int sample = 0; sample < samples; ++sample)
            for (size_t offset = 0; offset < cases.size(); ++offset)
                run(cases[(sample + offset) % cases.size()], sample);
        std::puts("FORMAT RESULT,case,samples,min_cycles,median_cycles,max_cycles,status");
        for (auto& c : cases) {
            std::sort(c.cycles.begin(), c.cycles.end());
            const size_t mid = c.cycles.size() / 2;
            const double median = c.cycles.size() % 2 ? c.cycles[mid] :
                (c.cycles[mid - 1] + c.cycles[mid]) / 2.0;
            std::printf("RESULT,%s,%d,%llu,%.1f,%llu,PASS\n", c.name, samples,
                        c.cycles.front(), median, c.cycles.back());
        }
        CUDA_CHECK(cudaFree(observations));
        CUDA_CHECK(cudaFree(scrub));
        CUDA_CHECK(cudaFree(noise));
        CUDA_CHECK(cudaFree(hot));
        std::puts("SUMMARY copy_cases=10 probe_cases=12 failed=0 skipped=0");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}

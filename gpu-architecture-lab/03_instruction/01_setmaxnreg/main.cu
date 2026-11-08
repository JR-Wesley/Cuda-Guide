#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifndef LAB_DYNAMIC
#define LAB_DYNAMIC 1
#endif
#ifndef LAB_TARGET_CC
#define LAB_TARGET_CC 90
#endif

constexpr int kThreads = 256;
constexpr int kWarpgroup = 128;
constexpr int kAccumulators = 128;
constexpr int kInitial = 128;

static void check(cudaError_t error, const char* expression, int line) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA_ERROR line=%d call=%s code=%d name=%s message=%s\n",
                     line, expression, int(error), cudaGetErrorName(error),
                     cudaGetErrorString(error));
        std::exit(1);
    }
}
#define CUDA_CHECK(call) check((call), #call, __LINE__)

template<int Count>
__device__ __forceinline__ void decrease() {
    static_assert(Count >= 24 && Count <= 256 && Count % 8 == 0);
#if LAB_DYNAMIC
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(Count) : "memory");
#endif
}

template<int Count>
__device__ __forceinline__ void increase() {
    static_assert(Count >= 24 && Count <= 256 && Count % 8 == 0);
#if LAB_DYNAMIC
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(Count) : "memory");
#endif
}

// Unsigned arithmetic gives an exact, portable modulo-2^32 CPU reference.
__host__ __device__ inline uint32_t initial_value(uint32_t index, int lane, int round) {
    return (index + 1u) * 747796405u ^ (uint32_t(lane) + 1u) * 2891336453u
           ^ uint32_t(round) * 277803737u;
}
__host__ __device__ inline uint32_t advance(uint32_t value, int lane, int step) {
    return value * 1664525u + 1013904223u + uint32_t(lane) + uint32_t(step);
}

// mode=0: fixed-register control; mode=1: all threads 128->64->128;
// mode=2: WG0/WG1 128/128->64/192->128/128.
// The build specifies --maxrregcount=128. Inspect the emitted resource metadata;
// the command-line cap alone is not proof of an initial allocation of 128.
template<int Mode>
__global__ void experiment(uint32_t* output, int rounds, int steps) {
    const bool donor = threadIdx.x < kWarpgroup; // uniform within each warpgroup
    const uint32_t index = blockIdx.x * kThreads + threadIdx.x;
    for (int round = 0; round < rounds; ++round) {
        if constexpr (Mode == 1) decrease<64>();
        if constexpr (Mode == 2) { if (donor) decrease<64>(); }
        __syncthreads(); // release completed before requesting; all 256 participate
        if constexpr (Mode == 1) increase<128>();
        __syncthreads(); // explicit warpgroup synchronization after setmaxnreg

        if (donor) {
            uint32_t result = initial_value(index, 0, round);
            for (int step = 0; step < steps; ++step) result = advance(result, 0, step);
            output[size_t(round) * gridDim.x * kThreads + index] = result;
        } else {
            if constexpr (Mode == 2) increase<192>();
            // Barrier 1 belongs only to WG1; all its 128 threads participate.
            // Use the non-.aligned form: the other half of the CTA skips it.
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            // Initialize only AFTER inc. No accumulator is live across dec.
            uint32_t accum[kAccumulators];
            #pragma unroll
            for (int j = 0; j < kAccumulators; ++j)
                accum[j] = initial_value(index, j, round);
            #pragma unroll 1
            for (int step = 0; step < steps; ++step) {
                #pragma unroll
                for (int j = 0; j < kAccumulators; ++j)
                    accum[j] = advance(accum[j], j, step);
            }
            uint32_t result = 0;
            #pragma unroll
            for (int j = 0; j < kAccumulators; ++j)
                result ^= accum[j] * (2u * uint32_t(j) + 1u);
            // Retain every round and end accumulator lifetimes before shrinking.
            output[size_t(round) * gridDim.x * kThreads + index] = result;
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if constexpr (Mode == 2) decrease<128>();
        }
        __syncthreads(); // return borrowed registers before the donor requests them
        if constexpr (Mode == 2) { if (donor) increase<128>(); }
        __syncthreads(); // next iteration starts only after the full restore
    }
}

static int parse(const char* value, int maximum) {
    errno = 0;
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (errno || !*value || *end || parsed < 1 || parsed > maximum) {
        std::fprintf(stderr, "Invalid positive integer: %s (maximum=%d)\n", value, maximum);
        std::exit(2);
    }
    return int(parsed);
}

static std::vector<uint32_t> reference(int blocks, int rounds, int steps) {
    std::vector<uint32_t> expected(size_t(blocks) * kThreads * rounds);
    for (int r = 0; r < rounds; ++r) {
        for (int i = 0; i < blocks * kThreads; ++i) {
            uint32_t value = 0;
            const bool donor = (i % kThreads) < kWarpgroup;
            for (int j = 0; j < (donor ? 1 : kAccumulators); ++j) {
                uint32_t a = initial_value(uint32_t(i), j, r);
                for (int s = 0; s < steps; ++s) a = advance(a, j, s);
                value ^= a * (2u * uint32_t(j) + 1u);
            }
            expected[size_t(r) * blocks * kThreads + i] = value;
        }
    }
    return expected;
}

template<int Mode>
static void run_case(const char* name, uint32_t* device, int blocks, int rounds,
                     int steps, int repeats, const std::vector<uint32_t>& expected) {
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, experiment<Mode>));
    std::printf("CASE %s numRegs=%d localSizeBytes=%zu binaryVersion=%d\n",
                name, attributes.numRegs, attributes.localSizeBytes, attributes.binaryVersion);
    if constexpr (Mode != 0) {
        if (attributes.numRegs != kInitial) {
            std::fprintf(stderr, "FAIL initial register allocation must be 128; refusing dynamic launch\n");
            std::exit(1);
        }
    }
    const size_t bytes = expected.size() * sizeof(uint32_t);
    CUDA_CHECK(cudaMemset(device, 0xa5, bytes));
    experiment<Mode><<<blocks, kThreads>>>(device, rounds, steps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint32_t> actual(expected.size());
    CUDA_CHECK(cudaMemcpy(actual.data(), device, bytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::fprintf(stderr, "FAIL %s index=%zu expected=%u actual=%u\n",
                         name, i, expected[i], actual[i]);
            std::exit(1);
        }
    }
    std::printf("PASS %s checked=%zu\n", name, expected.size());
    if (repeats == 0) return; // sanitizer run: correctness only
    for (int i = 0; i < 3; ++i) experiment<Mode><<<blocks, kThreads>>>(device, rounds, steps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times;
    for (int sample = 0; sample < 5; ++sample) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < repeats; ++i)
            experiment<Mode><<<blocks, kThreads>>>(device, rounds, steps);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms * 1000.0f / repeats);
        std::printf("TIMING %s sample=%d batch_average_us=%.6f launches=%d\n",
                    name, sample, times.back(), repeats);
    }
    std::sort(times.begin(), times.end());
    std::printf("SUMMARY %s median_batch_average_us=%.6f min=%.6f max=%.6f\n",
                name, times[2], times.front(), times.back());
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main(int argc, char** argv) {
    if (argc != 1 && argc != 5) {
        std::fprintf(stderr, "Usage: %s [blocks rounds steps repeats(0=validation-only)]\n", argv[0]);
        return 2;
    }
    const int blocks = argc == 5 ? parse(argv[1], 4096) : 32;
    const int rounds = argc == 5 ? parse(argv[2], 1024) : 3;
    const int steps = argc == 5 ? parse(argv[3], 4096) : 8;
    const int repeats = argc == 5 ? (std::string(argv[4]) == "0" ? 0 : parse(argv[4], 10000)) : 20;
    if (uint64_t(blocks) * rounds > 262144 || uint64_t(blocks) * rounds * steps > 1048576) {
        std::fprintf(stderr, "Input exceeds this teaching experiment's memory/work budget\n");
        return 2;
    }
    int count = 0;
    cudaError_t availability = cudaGetDeviceCount(&count);
    if (availability != cudaSuccess || count == 0) {
        std::fprintf(stderr, "SKIPPED no usable CUDA device: code=%d message=%s count=%d\n",
                     int(availability), cudaGetErrorString(availability), count);
        return 77;
    }
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp property{};
    CUDA_CHECK(cudaGetDeviceProperties(&property, 0));
    int driver = 0, runtime = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    std::printf("DEVICE name=%s cc=%d.%d driver_api=%d runtime=%d SMs=%d\n",
                property.name, property.major, property.minor, driver, runtime, property.multiProcessorCount);
    std::printf("CONFIG target_cc=%d dynamic=%d blocks=%d threads=%d rounds=%d steps=%d repeats=%d\n",
                LAB_TARGET_CC, LAB_DYNAMIC, blocks, kThreads, rounds, steps, repeats);
    // This lab builds an architecture-specific native cubin, not a portable PTX JIT fallback.
    if (property.major * 10 + property.minor != LAB_TARGET_CC) {
        std::fprintf(stderr, "SKIPPED device/target mismatch; dynamic instruction was not executed\n");
        return 77;
    }
    const auto expected = reference(blocks, rounds, steps);
    uint32_t* device = nullptr;
    CUDA_CHECK(cudaMalloc(&device, expected.size() * sizeof(uint32_t)));
    run_case<0>("fixed128", device, blocks, rounds, steps, repeats, expected);
#if LAB_DYNAMIC
    run_case<1>("roundtrip", device, blocks, rounds, steps, repeats, expected);
    run_case<2>("redistribute", device, blocks, rounds, steps, repeats, expected);
#else
    std::puts("BASELINE_ONLY setmaxnreg not compiled; no dynamic-register result");
#endif
    CUDA_CHECK(cudaFree(device));
    return 0;
}

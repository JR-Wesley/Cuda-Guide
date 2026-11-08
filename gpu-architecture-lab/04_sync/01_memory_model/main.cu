#include <cuda/atomic>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace cg = cooperative_groups;

static void check(cudaError_t error, const char* expression, int line) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(expression) + " at line " +
                                 std::to_string(line) + ": " + cudaGetErrorString(error));
    }
}
#define CUDA_CHECK(call) check((call), #call, __LINE__)

enum class Mode { release_acquire, seq_cst, fence_atomic, atomic_fence,
                  fence_fence, relaxed_atomic_payload };

static const char* mode_name(Mode mode) {
    switch (mode) {
        case Mode::release_acquire: return "release_acquire";
        case Mode::seq_cst: return "seq_cst";
        case Mode::fence_atomic: return "fence_atomic";
        case Mode::atomic_fence: return "atomic_fence";
        case Mode::fence_fence: return "fence_fence";
        case Mode::relaxed_atomic_payload: return "relaxed_atomic_payload";
    }
    return "unknown";
}

// All atomic_ref instantiations must meet their alignment requirement.
constexpr size_t atomic_alignment =
    cuda::atomic_ref<int, cuda::thread_scope_system>::required_alignment;
static_assert(atomic_alignment >=
              cuda::atomic_ref<int, cuda::thread_scope_device>::required_alignment);
static_assert(atomic_alignment >=
              cuda::atomic_ref<int, cuda::thread_scope_block>::required_alignment);

struct Mailbox {
    alignas(atomic_alignment) int flag;
    alignas(atomic_alignment) int payload;
};
struct Result {
    int received;
    int stale;
    int unexpected;
    int timeout;
};

template <bool CrossBlock>
__device__ void round_barrier() {
    if constexpr (CrossBlock) cg::this_grid().sync();
    else __syncthreads();
}

template <cuda::thread_scope Scope, Mode M, bool CrossBlock>
__global__ void message_passing(Mailbox* box, Result* result, int rounds,
                                int spin_limit) {
    // Producer and consumer occupy different warps, including the block case.
    const int rank = blockIdx.x * blockDim.x + threadIdx.x;
    for (int round = 0; round < rounds; ++round) {
        if (rank == 0) {
            box->flag = 0;
            box->payload = 0;
        }
        // No atomic_ref from the previous round survives into reset.
        round_barrier<CrossBlock>();
        {
            cuda::atomic_ref<int, Scope> flag(box->flag);
            const int expected = round + 1;
            if (rank == 0) {
                if constexpr (M == Mode::relaxed_atomic_payload) {
                    cuda::atomic_ref<int, Scope> payload(box->payload);
                    payload.store(expected, cuda::memory_order_relaxed);
                } else {
                    box->payload = expected;
                }

                if constexpr (M == Mode::fence_atomic || M == Mode::fence_fence)
                    cuda::atomic_thread_fence(cuda::memory_order_release, Scope);

                if constexpr (M == Mode::seq_cst)
                    flag.store(1, cuda::memory_order_seq_cst);
                else if constexpr (M == Mode::release_acquire || M == Mode::atomic_fence)
                    flag.store(1, cuda::memory_order_release);
                else
                    flag.store(1, cuda::memory_order_relaxed);
            } else if (rank == 32) {
                bool ready = false;
                for (int spin = 0; spin < spin_limit; ++spin) {
                    int observed;
                    if constexpr (M == Mode::seq_cst)
                        observed = flag.load(cuda::memory_order_seq_cst);
                    else if constexpr (M == Mode::release_acquire || M == Mode::fence_atomic)
                        observed = flag.load(cuda::memory_order_acquire);
                    else
                        observed = flag.load(cuda::memory_order_relaxed);
                    if (observed == 1) { ready = true; break; }
                    __nanosleep(64);
                }
                if (ready) {
                    if constexpr (M == Mode::atomic_fence || M == Mode::fence_fence)
                        cuda::atomic_thread_fence(cuda::memory_order_acquire, Scope);

                    int value;
                    if constexpr (M == Mode::relaxed_atomic_payload) {
                        cuda::atomic_ref<int, Scope> payload(box->payload);
                        value = payload.load(cuda::memory_order_relaxed);
                    } else {
                        value = box->payload;
                    }
                    ++result->received;
                    if (value == 0) ++result->stale;
                    else if (value != expected) ++result->unexpected;
                } else {
                    ++result->timeout; // Never read non-atomic payload on timeout.
                }
            }
        }
        // This barrier is AFTER the observation, so it cannot publish the payload
        // for the read above. It only makes reuse in the next round safe.
        round_barrier<CrossBlock>();
    }
}

__global__ void relaxed_counter(int* value) {
    cuda::atomic_ref<int, cuda::thread_scope_device> counter(*value);
    for (int i = 0; i < 100; ++i)
        counter.fetch_add(1, cuda::memory_order_relaxed);
}

template <cuda::thread_scope Scope, Mode M, bool CrossBlock>
static bool run_case(const char* scope_name, Mailbox* box, Result* result,
                     int rounds, int spin_limit, const cudaDeviceProp& prop,
                     int& skipped) {
    CUDA_CHECK(cudaMemset(result, 0, sizeof(Result)));
    if constexpr (CrossBlock) {
        int blocks_per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, message_passing<Scope, M, true>, 32, 0));
        if (!prop.cooperativeLaunch || blocks_per_sm * prop.multiProcessorCount < 2) {
            std::printf("SKIP scope=%s mode=%s reason=cooperative_grid_unavailable\n",
                        scope_name, mode_name(M));
            ++skipped;
            return true;
        }
        void* args[] = {&box, &result, &rounds, &spin_limit};
        CUDA_CHECK(cudaLaunchCooperativeKernel(
            reinterpret_cast<const void*>(message_passing<Scope, M, true>),
            dim3(2), dim3(32), args));
    } else {
        message_passing<Scope, M, false><<<1, 64>>>(box, result, rounds, spin_limit);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    Result host{};
    CUDA_CHECK(cudaMemcpy(&host, result, sizeof(host), cudaMemcpyDeviceToHost));
    const bool valid = host.received == rounds && host.timeout == 0 &&
                       host.unexpected == 0 &&
                       (M == Mode::relaxed_atomic_payload || host.stale == 0);
    std::printf("%s scope=%s mode=%s rounds=%d received=%d stale=%d unexpected=%d timeout=%d\n",
                valid ? "PASS" : "FAIL", scope_name, mode_name(M), rounds,
                host.received, host.stale, host.unexpected, host.timeout);
    return valid;
}

template <cuda::thread_scope Scope, bool CrossBlock>
static int run_scope(const char* name, Mailbox* box, Result* result,
                     int rounds, int spin_limit, const cudaDeviceProp& prop,
                     int& skipped) {
    int failed = 0;
#define RUN_MODE(m) failed += !run_case<Scope, Mode::m, CrossBlock>( \
    name, box, result, rounds, spin_limit, prop, skipped)
    RUN_MODE(release_acquire);
    RUN_MODE(seq_cst);
    RUN_MODE(fence_atomic);
    RUN_MODE(atomic_fence);
    RUN_MODE(fence_fence);
    RUN_MODE(relaxed_atomic_payload);
#undef RUN_MODE
    return failed;
}

static int positive_int(const char* value) {
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (!*value || *end || parsed < 1 || parsed > 1000000)
        throw std::runtime_error("arguments must be integers in [1, 1000000]");
    return static_cast<int>(parsed);
}

int main(int argc, char** argv) {
    try {
        if (argc > 3 || (argc > 1 && std::strcmp(argv[1], "--help") == 0)) {
            std::printf("Usage: %s [rounds=1000] [spin_limit=100000]\n", argv[0]);
            return argc > 3 ? 2 : 0;
        }
        const int rounds = argc > 1 ? positive_int(argv[1]) : 1000;
        const int spin_limit = argc > 2 ? positive_int(argv[2]) : 100000;
        int device_count = 0;
        const auto status = cudaGetDeviceCount(&device_count);
        if (status != cudaSuccess || device_count == 0) {
            std::printf("SKIP GPU unavailable: %s, count=%d\n",
                        cudaGetErrorString(status), device_count);
            return 77;
        }
        CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        int driver = 0, runtime = 0;
        CUDA_CHECK(cudaDriverGetVersion(&driver));
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
        std::printf("GPU=%s cc=%d.%d SMs=%d driver=%d runtime=%d\n",
                    prop.name, prop.major, prop.minor, prop.multiProcessorCount, driver, runtime);
        std::printf("cooperativeLaunch=%d managedMemory=%d concurrentManagedAccess=%d "
                    "pageableMemoryAccess=%d hostNativeAtomicSupported=%d\n",
                    prop.cooperativeLaunch, prop.managedMemory, prop.concurrentManagedAccess,
                    prop.pageableMemoryAccess, prop.hostNativeAtomicSupported);
        if (prop.major < 7) {
            std::puts("SKIP this harness requires compute capability >= 7.0");
            return 77;
        }
        Mailbox* box = nullptr;
        Result* result = nullptr;
        int* count = nullptr;
        CUDA_CHECK(cudaMalloc(&box, sizeof(Mailbox)));
        CUDA_CHECK(cudaMalloc(&result, sizeof(Result)));
        CUDA_CHECK(cudaMalloc(&count, sizeof(int)));
        CUDA_CHECK(cudaMemset(count, 0, sizeof(int)));
        relaxed_counter<<<32, 128>>>(count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        int actual = 0;
        CUDA_CHECK(cudaMemcpy(&actual, count, sizeof(int), cudaMemcpyDeviceToHost));
        const int expected = 32 * 128 * 100;
        int failed = actual != expected;
        int skipped = 0;
        std::printf("%s relaxed_counter actual=%d expected=%d\n",
                    failed ? "FAIL" : "PASS", actual, expected);
        failed += run_scope<cuda::thread_scope_block, false>(
            "block", box, result, rounds, spin_limit, prop, skipped);
        failed += run_scope<cuda::thread_scope_device, true>(
            "device", box, result, rounds, spin_limit, prop, skipped);
        // Device allocation, concurrently accessed only by one GPU: valid system
        // scope case. CPU observes the results only after kernel completion.
        failed += run_scope<cuda::thread_scope_system, true>(
            "system_gpu_only", box, result, rounds, spin_limit, prop, skipped);
        CUDA_CHECK(cudaFree(count));
        CUDA_CHECK(cudaFree(result));
        CUDA_CHECK(cudaFree(box));
        std::printf("SUMMARY failed=%d skipped=%d\n", failed, skipped);
        return failed ? 1 : (skipped ? 77 : 0);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ERROR %s\n", error.what());
        return 1;
    }
}

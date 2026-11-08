#include <cuda_runtime.h>
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_scan.cuh>
#include <cub/version.cuh>
#if __has_include(<cuda/warp>)
#include <cuda/warp>
#define LAB_HAS_WARP 1
#else
#define LAB_HAS_WARP 0
#endif
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#error "This lab requires sm_80 or newer for native warp reductions"
#endif

constexpr unsigned FULL = 0xffffffffu;
void check_cuda(cudaError_t e, const char* call, int line) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string(call) + " line=" + std::to_string(line) +
                                 " " + cudaGetErrorName(e) + ": " + cudaGetErrorString(e));
    }
}
#define CUDA(call) check_cuda((call), #call, __LINE__)
void finish() { CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize()); }

template<class T> struct Buffer {
    T* p = nullptr;
    explicit Buffer(size_t n) { CUDA(cudaMalloc(&p, n * sizeof(T))); }
    ~Buffer() { if (p) cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void put(const std::vector<T>& h) { CUDA(cudaMemcpy(p, h.data(), h.size()*sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get(size_t n) {
        std::vector<T> h(n);
        CUDA(cudaMemcpy(h.data(), p, n*sizeof(T), cudaMemcpyDeviceToHost));
        return h;
    }
};

__host__ __device__ unsigned low_bits(int n) { return n == 32 ? FULL : ((1u << n) - 1); }
__device__ int linear_tid() {
    return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}

struct Collective {
    unsigned members, snapshot, votes, equal, same_mask, uniform_mask, zero_equal;
    unsigned bits_and, bits_or, bits_xor, sparse;
    int all, any, same, uniform, sum, minimum, maximum, sparse_sum, sparse_reference;
};

// Handles both a partial physical warp and a partial logical input tail.
__global__ void collectives_kernel(const int* input, Collective* output, int n) {
    int t = linear_tid(), lane = t % 32;
    int threads = blockDim.x * blockDim.y * blockDim.z;
    int i = blockIdx.x * threads + t;
    unsigned physical = low_bits(min(32, threads - (t / 32) * 32));
    unsigned members = __ballot_sync(physical, i < n); // BEFORE any early return
    if (i >= n) return;
    int value = input[i];
    int key = value & 3;
    Collective r{};
    r.members = members;
    r.snapshot = __activemask(); // observational only, never used as a collective mask
    r.votes = __ballot_sync(members, value > 0);
    r.all = __all_sync(members, value > 0);
    r.any = __any_sync(members, value > 0);
    r.equal = __match_any_sync(members, key);
    r.same_mask = __match_all_sync(members, key, &r.same);
    int first = __shfl_sync(members, value, __ffs(members) - 1);
    r.uniform_mask = __match_all_sync(members, first, &r.uniform);
    // Numeric equality (+0 == -0) differs from match's bitwise comparison.
    float signed_zero = __uint_as_float((lane & 1) ? 0x80000000u : 0u);
    r.zero_equal = __match_any_sync(members, signed_zero);
    r.sum = __reduce_add_sync(members, value);
    r.minimum = __reduce_min_sync(members, value);
    r.maximum = __reduce_max_sync(members, value);
    unsigned bits = static_cast<unsigned>(value) & 255u;
    r.bits_and = __reduce_and_sync(members, bits);
    r.bits_or = __reduce_or_sync(members, bits);
    r.bits_xor = __reduce_xor_sync(members, bits);
    // Deliberately irregular membership; every participant calls with the same mask.
    bool selected = (value & 1) != 0;
    r.sparse = __ballot_sync(members, selected);
    if (selected) {
        r.sparse_sum = __reduce_add_sync(r.sparse, value);
        // A safe teaching reference for arbitrary masks, not a fast reduction tree.
        for (unsigned left = r.sparse; left; left &= left - 1)
            r.sparse_reference += __shfl_sync(r.sparse, value, __ffs(left) - 1);
    }
    output[i] = r;
}

struct Shuffle { int idx, wrapped, up, down, butterfly, scan; };
__global__ void shuffle_kernel(const int* input, Shuffle* output, int width, int delta, int xor_mask) {
    int t = threadIdx.x, lane = t % 32, value = input[t];
    Shuffle r{};
    r.idx = __shfl_sync(FULL, value, 0, width);
    r.wrapped = __shfl_sync(FULL, value, width + 1, width);
    r.up = __shfl_up_sync(FULL, value, delta, width);
    r.down = __shfl_down_sync(FULL, value, delta, width);
    r.butterfly = __shfl_xor_sync(FULL, value, xor_mask, width);
    int sum = value;
    for (int offset = 1; offset < width; offset *= 2) {
        int previous = __shfl_up_sync(FULL, sum, offset, width);
        if (lane % width >= offset) sum += previous;
    }
    r.scan = sum;
    output[t] = r;
}

struct Library { int scan8, reduce32, broadcast, lib_match; };
__global__ void library_kernel(const int* input, Library* output) {
    using Scan = cub::WarpScan<int, 8>;
    using Reduce = cub::WarpReduce<int, 32>;
    // 64 threads: 8 logical scan warps, 2 physical reduction warps.
    __shared__ typename Scan::TempStorage scan_storage[8];
    __shared__ typename Reduce::TempStorage reduce_storage[2];
    int t = threadIdx.x, lane = t % 32, value = input[t];
    Library r{};
    Scan(scan_storage[t / 8]).InclusiveSum(value, r.scan8);
    int aggregate = Reduce(reduce_storage[t / 32]).Sum(value);
    // Only lane 0's CUB aggregate is meaningful; initialize other shuffle operands.
    int source = lane == 0 ? aggregate : 0;
    r.reduce32 = __shfl_sync(FULL, source, 0);
#if LAB_HAS_WARP
    r.broadcast = cuda::device::warp_shuffle_idx(value, 0);
#else
    r.broadcast = __shfl_sync(FULL, value, 0);
#endif
#if LAB_HAS_WARP && defined(CCCL_VERSION) && CCCL_VERSION >= 3001000
    r.lib_match = cuda::device::warp_match_all(r.broadcast);
#else
    r.lib_match = -1;
#endif
    output[t] = r;
}

// Full physical warps only. The shared storage is reused across rounds.
__global__ void shared_exchange_kernel(const int* input, int* output, int rounds, bool cross_half) {
    __shared__ int data[64];
    int t = threadIdx.x, lane = t % 32, base = t - lane;
    int group = lane / 16;
    unsigned mask = group == 0 ? 0xffffu : 0xffff0000u;
    int value = input[t];
    for (int round = 0; round < rounds; ++round) {
        data[t] = value;
        if (cross_half) {
            __syncwarp(FULL); // producers and consumers span both halves
            value = data[base + 31 - lane];
            __syncwarp(FULL); // protect reads before the next overwrite
        } else if (group == 0) {
            __syncwarp(mask);
            value = data[base + 15 - lane];
            __syncwarp(mask);
        } else {
            __syncwarp(mask);
            value = data[base + 47 - lane]; // 16..31 -> 31..16, stays in this half
            __syncwarp(mask);
        }
    }
    output[t] = value;
}

// Both calls are legal. The first is a logic bug if the intended groups have 16 lanes.
__global__ void width_debug_kernel(int* output) {
    int lane = threadIdx.x;
    output[lane] = __shfl_sync(FULL, lane, 16); // all 32 lanes receive lane 16
    output[32 + lane] = __shfl_sync(FULL, lane, 0, 16); // two broadcasts: lane 0 / 16
}

size_t checks = 0;
template<class A, class B> void expect(A actual, B expected, const char* field, int i) {
    ++checks;
    if (actual != expected)
        throw std::runtime_error(std::string(field) + " index=" + std::to_string(i) +
            " got=" + std::to_string(static_cast<long long>(actual)) +
            " expected=" + std::to_string(static_cast<long long>(expected)));
}
std::vector<int> values(int n, int seed) {
    std::vector<int> result(n);
    unsigned state = static_cast<unsigned>(seed);
    for (int& v : result) { state = state * 1664525u + 1013904223u; v = int((state >> 16) % 101) - 50; }
    return result;
}

void test_collectives(bool quick) {
    const std::vector<int> sizes = quick ? std::vector<int>{1, 17, 33, 65} :
        std::vector<int>{1, 2, 4, 17, 31, 32, 33, 63, 64, 65, 129, 259, 1025};
    int cases = 0;
    for (dim3 block : {dim3(1), dim3(17), dim3(32), dim3(33), dim3(64), dim3(8, 8)}) {
        int threads = block.x * block.y;
        for (int n : sizes) for (int seed : {7, 19}) {
            auto input = values(n, seed);
            Buffer<int> d_in(n); d_in.put(input);
            Buffer<Collective> d_out(n);
            collectives_kernel<<<(n + threads - 1) / threads, block>>>(d_in.p, d_out.p, n);
            finish();
            auto output = d_out.get(n);
            for (int i = 0; i < n; ++i) {
                int t = i % threads, lane = t % 32;
                int base = i - lane;
                int count = std::min({32, threads - (t / 32)*32, n - base});
                unsigned members = low_bits(count), votes = 0, equal = 0, zeros = 0, sparse = 0;
                unsigned band = FULL, bor = 0, bxor = 0;
                int sum = 0, minimum = input[base], maximum = input[base], sparse_sum = 0;
                for (int j = 0; j < count; ++j) {
                    int v = input[base+j];
                    if (v > 0) votes |= 1u << j;
                    if ((v & 3) == (input[i] & 3)) equal |= 1u << j;
                    if ((j & 1) == (lane & 1)) zeros |= 1u << j;
                    if (v & 1) { sparse |= 1u << j; sparse_sum += v; }
                    sum += v; minimum = std::min(minimum, v); maximum = std::max(maximum, v);
                    unsigned bits = static_cast<unsigned>(v) & 255u;
                    band &= bits; bor |= bits; bxor ^= bits;
                }
                const auto& r = output[i];
#define E(field, wanted) expect(r.field, (wanted), #field, i)
                E(members, members); E(votes, votes); E(all, votes == members); E(any, votes != 0);
                // Do not assert snapshot == members: convergence at a later call is not guaranteed.
                expect((r.snapshot & (1u << lane)) != 0, true, "snapshot_contains_self", i);
                E(equal, equal); E(same, equal == members); E(same_mask, equal == members ? members : 0u);
                E(uniform, 1); E(uniform_mask, members); E(zero_equal, zeros);
                E(sum, sum); E(minimum, minimum); E(maximum, maximum);
                E(bits_and, band); E(bits_or, bor); E(bits_xor, bxor); E(sparse, sparse);
                E(sparse_sum, (input[i] & 1) ? sparse_sum : 0);
                E(sparse_reference, (input[i] & 1) ? sparse_sum : 0);
#undef E
            }
            ++cases;
        }
    }
    std::printf("PASS collectives cases=%d (partial physical warps, input tails, sparse masks, 2D blocks)\n", cases);
}

void test_shuffles_and_libraries() {
    auto input = values(64, 29);
    Buffer<int> d_in(64); d_in.put(input);
    Buffer<Shuffle> d_out(64);
    int cases = 0;
    for (int width : {1, 2, 4, 8, 16, 32})
        for (int delta : {0, 1, 2, 8, 16}) for (int xor_mask : {1, 2, 8, 16}) {
            shuffle_kernel<<<1, 64>>>(d_in.p, d_out.p, width, delta, xor_mask);
            finish();
            auto output = d_out.get(64);
            for (int i = 0; i < 64; ++i) {
                int lane = i % 32, warp_base = i - lane;
                int base = i - lane % width, local = lane % width;
                // XOR may reach an earlier subsection; a later subsection returns self.
                int xor_lane = lane ^ xor_mask;
                if (xor_lane > (lane / width) * width + width - 1) xor_lane = lane;
                int scan = 0;
                for (int j = base; j <= i; ++j) scan += input[j];
                const auto& r = output[i];
                expect(r.idx, input[base], "idx", i);
                expect(r.wrapped, input[base + (width + 1) % width], "wrapped", i);
                expect(r.up, input[local >= delta ? i - delta : i], "up", i);
                expect(r.down, input[local + delta < width ? i + delta : i], "down", i);
                expect(r.butterfly, input[warp_base + xor_lane], "xor", i);
                expect(r.scan, scan, "scan", i);
            }
            ++cases;
        }
    std::printf("PASS shuffle cases=%d (all legal widths, up/down edges, XOR earlier groups, scan)\n", cases);
    Buffer<Library> d_lib(64);
    library_kernel<<<1, 64>>>(d_in.p, d_lib.p); finish();
    auto lib = d_lib.get(64);
    for (int i = 0; i < 64; ++i) {
        int scan = 0, sum = 0;
        for (int j = (i/8)*8; j <= i; ++j) scan += input[j];
        for (int j = (i/32)*32; j < (i/32+1)*32; ++j) sum += input[j];
        expect(lib[i].scan8, scan, "cub_scan8", i);
        expect(lib[i].reduce32, sum, "cub_reduce32", i);
        expect(lib[i].broadcast, input[(i/32)*32], "library_broadcast", i);
#if LAB_HAS_WARP && defined(CCCL_VERSION) && CCCL_VERSION >= 3001000
        expect(lib[i].lib_match, 1, "library_match", i);
#else
        expect(lib[i].lib_match, -1, "library_match_unavailable", i);
#endif
    }
    std::printf("PASS CUB scan8/reduce32; libcu++ shuffle=%s match=%s CUB_VERSION=%d\n",
        LAB_HAS_WARP ? "tested" : "SKIPPED (intrinsic fallback)",
        lib[0].lib_match == -1 ? "SKIPPED (unavailable)" : "tested", CUB_VERSION);
}

void test_shared() {
    auto input = values(64, 37);
    Buffer<int> d_in(64), d_out(64); d_in.put(input);
    for (bool cross : {false, true}) for (int rounds : {1, 2, 101}) {
        shared_exchange_kernel<<<1, 64>>>(d_in.p, d_out.p, rounds, cross); finish();
        auto output = d_out.get(64);
        int width = cross ? 32 : 16;
        for (int i = 0; i < 64; ++i) {
            int src = rounds % 2 ? (i/width)*width + width - 1 - i%width : i;
            expect(output[i], input[src], "shared_exchange", i);
        }
    }
    std::puts("PASS shared exchange: disjoint half masks / full masks, rounds=1,2,101");
}

void debug_width() {
    Buffer<int> d_out(64);
    width_debug_kernel<<<1, 32>>>(d_out.p); finish();
    auto output = d_out.get(64);
    int mismatches = 0;
    for (int lane = 0; lane < 32; ++lane) {
        int wanted = lane < 16 ? 0 : 16;
        expect(output[lane], 16, "legal_default_width", lane);
        if (output[lane] != wanted) ++mismatches;
        expect(output[32+lane], wanted, "fixed_width16", lane);
    }
    expect(mismatches, 16, "expected_logic_mismatches", 0);
    std::printf("PASS debug: default width produced %d expected mismatches; src=0,width=16 fixed all lanes\n", mismatches);
}

int main(int argc, char** argv) {
    try {
        std::string mode = argc > 1 ? argv[1] : "run";
        if (argc > 2 || (mode != "run" && mode != "quick" && mode != "debug")) {
            std::fprintf(stderr, "Usage: %s [run|quick|debug]\n", argv[0]); return 2;
        }
        int count = 0;
        cudaError_t query = cudaGetDeviceCount(&count);
        if (query != cudaSuccess) { check_cuda(query, "cudaGetDeviceCount", __LINE__); }
        if (!count) { std::puts("SKIPPED: no CUDA device"); return 77; }
        CUDA(cudaSetDevice(0));
        cudaDeviceProp prop{}; CUDA(cudaGetDeviceProperties(&prop, 0));
        int driver = 0, runtime = 0;
        CUDA(cudaDriverGetVersion(&driver)); CUDA(cudaRuntimeGetVersion(&runtime));
        std::printf("GPU=%s cc=%d.%d driver_api=%d runtime=%d CUDART=%d mode=%s\n",
            prop.name, prop.major, prop.minor, driver, runtime, CUDART_VERSION, mode.c_str());
        if (prop.major < 8) { std::puts("SKIPPED: this complete suite requires cc >= 8.0"); return 77; }
        if (mode == "debug") debug_width();
        else { test_collectives(mode == "quick"); test_shuffles_and_libraries(); test_shared(); }
        std::printf("PASS checked_fields=%zu\n", checks);
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL %s\n", e.what()); return 1; }
}

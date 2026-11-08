#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

// Each invocation launches only the selected faulty kernel; no baseline kernel masks its report.
__global__ void bad_global_write(int *p) {
    if (threadIdx.x == 0) p[1] = 42;  // allocation contains only one int
}

__global__ void shared_write_race(int *out) {
    __shared__ int slot;
    slot = static_cast<int>(threadIdx.x);  // two unordered writes to the same shared word
    __syncthreads();
    if (threadIdx.x == 0) out[0] = slot;
}

__global__ void uninitialized_global_read(const int *in, int *out) {
    out[0] = in[0];  // in was allocated but never initialized
}

__global__ void divergent_warp_barrier(int *out) {
    if (threadIdx.x < 17) __syncwarp(0xffffu);  // lane 16 is not in the mask
    if (threadIdx.x == 0) out[0] = 1;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s memcheck|racecheck|initcheck|synccheck\n", argv[0]);
        return 2;
    }
    int *a = nullptr;
    int *b = nullptr;
    cudaError_t rc = cudaMalloc(&a, sizeof(int));
    if (rc != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc(a): %s\n", cudaGetErrorString(rc));
        return 1;
    }
    if (std::strcmp(argv[1], "memcheck") == 0) {
        bad_global_write<<<1, 1>>>(a);
    } else if (std::strcmp(argv[1], "racecheck") == 0) {
        shared_write_race<<<1, 2>>>(a);
    } else if (std::strcmp(argv[1], "initcheck") == 0) {
        rc = cudaMalloc(&b, sizeof(int));
        if (rc != cudaSuccess) {
            std::fprintf(stderr, "cudaMalloc(b): %s\n", cudaGetErrorString(rc));
            return 1;
        }
        uninitialized_global_read<<<1, 1>>>(b, a);
    } else if (std::strcmp(argv[1], "synccheck") == 0) {
        divergent_warp_barrier<<<1, 32>>>(a);
    } else {
        std::fprintf(stderr, "unknown case: %s\n", argv[1]);
        cudaFree(a);
        return 2;
    }
    if (rc == cudaSuccess) rc = cudaGetLastError();
    if (rc == cudaSuccess) rc = cudaDeviceSynchronize();
    if (rc != cudaSuccess) std::fprintf(stderr, "CUDA execution: %s\n", cudaGetErrorString(rc));
    // A program's exit status alone cannot establish that the sanitizer detected the fault.
    if (rc != cudaSuccess) return 1;  // a sticky device error makes later cleanup calls misleading
    if (b != nullptr) {
        rc = cudaFree(b);
        if (rc != cudaSuccess) {
            std::fprintf(stderr, "cudaFree(b): %s\n", cudaGetErrorString(rc));
            return 1;
        }
    }
    rc = cudaFree(a);
    if (rc != cudaSuccess) {
        std::fprintf(stderr, "cudaFree(a): %s\n", cudaGetErrorString(rc));
        return 1;
    }
    return 0;
}

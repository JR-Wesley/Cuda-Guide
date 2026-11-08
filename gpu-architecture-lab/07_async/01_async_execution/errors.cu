#include "common.cuh"

__global__ void harmless() {}
__global__ void invalid_write() {
    // Deliberate fault, only in a separate process selected by the "fault" argument.
    asm volatile("st.global.u32 [%0], %1;" : : "l"(0ull), "r"(123u) : "memory");
}
int main(int argc,char** argv) {
    try {
        require(argc==2,"Usage: errors launch|fault");
        const std::string mode=argv[1];
        require(mode=="launch" || mode=="fault","unknown mode");
        if(!initialize()) return 77;
        if(mode=="fault") {
            invalid_write<<<1,1>>>();
            const auto launch=cudaGetLastError();
            const auto sync=cudaDeviceSynchronize();
            std::printf("EXPECTED_FAULT,launch_check=%s,synchronize=%s\n",cudaGetErrorName(launch),cudaGetErrorName(sync));
            require(launch==cudaErrorIllegalAddress || sync==cudaErrorIllegalAddress,
                    "expected illegal address was not reported (see sanitizer output if instrumented)");
            std::puts("PASS expected asynchronous execution fault; exiting process without reuse");
            return 0;
        }
        cudaDeviceProp p{};
        CUDA_CHECK(cudaGetDeviceProperties(&p,0));
        harmless<<<1,p.maxThreadsPerBlock+1>>>();
        const auto peek=cudaPeekAtLastError();
        const auto get=cudaGetLastError();
        const auto cleared=cudaGetLastError();
        std::printf("EXPECTED_LAUNCH_ERROR,peek=%s,get=%s,after_clear=%s\n",
                    cudaGetErrorName(peek),cudaGetErrorName(get),cudaGetErrorName(cleared));
        require((peek==cudaErrorInvalidConfiguration || peek==cudaErrorInvalidValue) &&
                get==peek && cleared==cudaSuccess,
                "unexpected launch error state");
        harmless<<<1,1>>>();
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        std::puts("PASS recovered_from_invalid_launch_configuration");
    } catch(const std::exception& e) { std::fprintf(stderr,"FAIL %s\n",e.what()); return 1; }
}

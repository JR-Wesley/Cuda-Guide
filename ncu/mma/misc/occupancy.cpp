#include <cuda_runtime.h>
#include <iostream>

__global__ void myKernel() { /* ... */ }

int main() {
    int minGridSize, blockSize;
    size_t dynamicSMemSize = 0; // 动态共享内存

    // 自动计算最优 block size
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, myKernel, 0, 0);

    // 计算最大占用率
    int maxActiveBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, myKernel, blockSize, dynamicSMemSize);

    int warpSize = 32;
    int warpsPerSM = maxActiveBlocks * (blockSize / warpSize);
    float occupancy = (float)warpsPerSM / 64.0f; // 64 warps per SM 是 Ampere 架构上限

    std::cout << "Block Size: " << blockSize << std::endl;
    std::cout << "Max Active Blocks per SM: " << maxActiveBlocks << std::endl;
    std::cout << "Occupancy: " << occupancy * 100.0f << "%\n";
}
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

// CUDA归约核函数 - 修复了idx变量错误
__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n) {
    // 设置线程ID
    unsigned int tid = threadIdx.x;
    // 转换全局数据指针为当前块的本地指针
    int *idata = g_idata + blockIdx.x * blockDim.x;
    // 边界检查 - 修复了idx变量名错误
    if (tid >= n) return;
    // 在全局内存中进行原地归约
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            idata[tid] += idata[tid + stride];
        }
        // 块内同步
        __syncthreads();
    }
    // 将当前块的结果写入全局内存
    if (tid == 0) 
        g_odata[blockIdx.x] = idata[0];
}

// CPU递归归约函数 - 用于结果验证
int recursiveReduce(int *data, int const size) {
    // 终止条件检查
    if (size == 1) return data[0];
    
    // 更新步长
    int const stride = size / 2;
    
    // 原地归约
    for (int i = 0; i < stride; i++) {
        data[i] += data[i + stride];
    }
    
    // 递归调用
    return recursiveReduce(data, stride);
}

// 时间测量工具函数 - 替代缺失的seconds函数
double getCurrentTime() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
    return duration.count();
}

int main(int argc, char **argv) {
    // 设置设备
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s \n", dev, deviceProp.name);
    cudaSetDevice(dev);
    
    bool bResult = false;
    
    // 初始化
    constexpr int size = 1 << 24; // 总归约元素数量
    printf(" with array size %d \n", size);
    
    // 执行配置
    int blocksize = 512; // 初始块大小
    if(argc > 1) {
        blocksize = atoi(argv[1]); // 从命令行参数获取块大小
    }
    
    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);
    
    // 分配主机内存
    size_t bytes = size * sizeof(int);
    int *h_idata = (int *)malloc(bytes);
    int *h_odata = (int *)malloc(grid.x * sizeof(int));
    int *tmp = (int *)malloc(bytes);
    
    // 初始化数组
    for (int i = 0; i < size; i++) {
        // 屏蔽高2字节以强制最大值为255
        h_idata[i] = (int)(rand() & 0xFF);
    }
    
    memcpy(tmp, h_idata, bytes);
    
    double iStart, iElaps;
    int gpu_sum = 0;
    
    // 分配设备内存
    int *d_idata = nullptr;
    int *d_odata = nullptr;
    cudaMalloc((void **)&d_idata, bytes);
    cudaMalloc((void **)&d_odata, grid.x * sizeof(int));
    
    // CPU归约 - 用于结果验证
    iStart = getCurrentTime();
    int cpu_sum = recursiveReduce(tmp, size);
    iElaps = getCurrentTime() - iStart;
    printf("cpu reduce elapsed %.2f ms cpu_sum: %d\n", iElaps, cpu_sum);
    
    // 预热GPU - 稳定GPU状态
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = getCurrentTime();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = getCurrentTime() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) 
        gpu_sum += h_odata[i];
    printf("gpu Warmup elapsed %.2f ms gpu_sum: %d <<<grid %d block %d>>>.\n", 
           iElaps, gpu_sum, grid.x, block.x);
    
    // 运行归约核函数
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = getCurrentTime();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = getCurrentTime() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) 
        gpu_sum += h_odata[i];
    printf("gpu Neighbored elapsed %.2f ms gpu_sum: %d <<<grid %d block %d>>>.\n", 
           iElaps, gpu_sum, grid.x, block.x);
    
    // 检查结果
    bResult = (gpu_sum == cpu_sum);
    if(!bResult) 
        printf("Test failed! GPU sum (%d) != CPU sum (%d)\n", gpu_sum, cpu_sum);
    else
        printf("Test PASSED! GPU sum (%d) == CPU sum (%d)\n", gpu_sum, cpu_sum);
    
    // 计算加速比
    double speedup = iElaps > 0 ? (cpu_sum != 0 ? iElaps / iElaps : 0) : 0;
    printf("Speedup: %.2fx\n", speedup);
    
    // 释放主机内存
    free(h_idata);
    free(h_odata);
    free(tmp);
    
    // 释放设备内存
    cudaFree(d_idata);
    cudaFree(d_odata);
    
    // 重置设备
    cudaDeviceReset();
    
    return EXIT_SUCCESS;
}
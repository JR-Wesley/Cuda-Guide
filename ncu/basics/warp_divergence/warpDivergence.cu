#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 导致warp divergence的核函数
__global__ void mathKernel1(float* c, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    
    float a, b;
    if (tid % 2 == 0) {
        a = 1.0f;
        b = 2.0f;
    } else {
        a = 3.0f;
        b = 4.0f;
    }
    
    // 增加计算复杂度，让分支的影响更明显
    float result = 0.0f;
    for (int i = 0; i < 100; i++) {
        result += sqrtf(a * i + b) * sinf(a * b);
    }
    
    c[tid] = result;
}

// 避免warp divergence的核函数
__global__ void mathKernel2(float* c, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    
    float a, b;
    // 同一个warp内的所有线程执行相同的分支
    if ((tid / warpSize) % 2 == 0) {
        a = 1.0f;
        b = 2.0f;
    } else {
        a = 3.0f;
        b = 4.0f;
    }
    
    // 相同的计算复杂度
    float result = 0.0f;
    for (int i = 0; i < 100; i++) {
        result += sqrtf(a * i + b) * sinf(a * b);
    }
    
    c[tid] = result;
}

// 验证结果正确性
bool validateResults(float* h_c1, float* h_c2, int size) {
    for (int i = 0; i < size; i++) {
        if (fabs(h_c1[i] - h_c2[i]) > 1e-6) {
            printf("结果验证失败：索引 %d, 值1: %f, 值2: %f\n", i, h_c1[i], h_c2[i]);
            return false;
        }
    }
    return true;
}

int main() {
    // 增加数据规模
    const int size = 16 * 1024 * 1024; // 16MB 数据
    const int threadsPerBlock = 512;
    const int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    // 分配内存
    float* h_c1 = (float*)malloc(size * sizeof(float));
    float* h_c2 = (float*)malloc(size * sizeof(float));
    float* d_c;
    cudaMalloc((void**)&d_c, size * sizeof(float));
    
    // 定义事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds1 = 0.0f, milliseconds2 = 0.0f;
    
    const int iterations = 10; // 多次迭代以获得更准确的结果
    
    // 测试 mathKernel1 (有divergence)
    for (int i = 0; i < iterations; i++) {
        cudaEventRecord(start);
        mathKernel1<<<blocksPerGrid, threadsPerBlock>>>(d_c, size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float iterTime;
        cudaEventElapsedTime(&iterTime, start, stop);
        milliseconds1 += iterTime;
    }
    milliseconds1 /= iterations;
    
    cudaMemcpy(h_c1, d_c, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 测试 mathKernel2 (无divergence)
    for (int i = 0; i < iterations; i++) {
        cudaEventRecord(start);
        mathKernel2<<<blocksPerGrid, threadsPerBlock>>>(d_c, size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float iterTime;
        cudaEventElapsedTime(&iterTime, start, stop);
        milliseconds2 += iterTime;
    }
    milliseconds2 /= iterations;
    
    cudaMemcpy(h_c2, d_c, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 验证结果
    bool resultsValid = validateResults(h_c1, h_c2, size);
    if (resultsValid) {
        printf("结果验证成功！\n");
    }
    
    // 输出性能比较
    printf("mathKernel1 (有warp divergence) 平均执行时间: %.3f ms\n", milliseconds1);
    printf("mathKernel2 (无warp divergence) 平均执行时间: %.3f ms\n", milliseconds2);
    float speedup = milliseconds1 / milliseconds2;
    printf("加速比: %.2f倍\n", speedup);
    
    // 清理资源
    free(h_c1);
    free(h_c2);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
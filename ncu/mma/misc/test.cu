#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include <vector>

// 这个实现比 2 根据尺寸不同会快几倍
__global__ void sgemm_naive_kernel(const int M, const int N, const int K,
                                   const float alpha, const float *A,
                                   const float *B, const float beta, float *C) {

  // compute position in C that this thread is responsible for
  const uint n = blockIdx.x * blockDim.x + threadIdx.x;
  const uint m = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (n < N && m < M) {
    float sum = 0.0f;
    for (int i = 0; i < K; ++i) {
      sum += A[m * K + i] * B[i * N + n];
    }

    // C = α*(A@B)+β*C
    C[m * N + n] = alpha * sum + beta * C[m * N + n];
  }
}

__global__ void sgemm_naive2_kernel(const int M, const int N, const int K,
                                   const float alpha, const float *A,
                                   const float *B, const float beta, float *C) {

  // compute position in C that this thread is responsible for
  const uint m = blockIdx.x * blockDim.x + threadIdx.x;
  const uint n = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (n < N && m < M) {
    float sum = 0.0f;
    for (int i = 0; i < K; ++i) {
      sum += A[m * K + i] * B[i * N + n];
    }

    // C = α*(A@B)+β*C
    C[m * N + n] = alpha * sum + beta * C[m * N + n];
  }
}

// 初始化矩阵
void initMatrix(float* matrix, int size, float value = 0.0f) {
    if (value == 0.0f) {
        // 随机初始化
        for (int i = 0; i < size; ++i) {
            matrix[i] = static_cast<float>(rand()) / RAND_MAX;
        }
    } else {
        // 初始化为指定值
        for (int i = 0; i < size; ++i) {
            matrix[i] = value;
        }
    }
}

// 验证结果正确性
bool verifyResult(const float* hostC, const float* hostA, const float* hostB, 
                  int M, int N, int K, float alpha, float beta, float tolerance = 1e-3) {
    std::vector<float> expected(M * N, 0.0f);
    
    // 计算期望结果: C = α*(A@B)+β*C
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += hostA[m * K + k] * hostB[k * N + n];
            }
            expected[m * N + n] = alpha * sum + beta * hostC[m * N + n];
        }
    }
    
    // 比较结果
    for (int i = 0; i < M * N; ++i) {
        if (std::abs(hostC[i] - expected[i]) > tolerance) {
            std::cout << "Mismatch at index " << i << ": " << hostC[i] 
                      << " vs " << expected[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    // 矩阵维度
    constexpr int M = 2048;
    constexpr int N = 2048;
    constexpr int K = 2048;
    
    // SGEMM参数
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // 计算矩阵大小
    const int sizeA = M * K;
    const int sizeB = K * N;
    const int sizeC = M * N;
    
    // 分配主机内存
    std::vector<float> hostA(sizeA);
    std::vector<float> hostB(sizeB);
    std::vector<float> hostC(sizeC, 0.0f);
    std::vector<float> hostC_copy(sizeC, 0.0f); // 用于验证结果正确性
    
    // 初始化矩阵
    initMatrix(hostA.data(), sizeA);
    initMatrix(hostB.data(), sizeB);
    
    // 分配设备内存
    float *devA, *devB, *devC;
    cudaMalloc((void**)&devA, sizeA * sizeof(float));
    cudaMalloc((void**)&devB, sizeB * sizeof(float));
    cudaMalloc((void**)&devC, sizeC * sizeof(float));
    
    // 复制数据到设备
    cudaMemcpy(devA, hostA.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, hostB.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice);
    
    // 定义网格和块大小
    const int BLOCK_SIZE = 32;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 测试参数
    const int numRuns = 10;
    float elapsedTime1 = 0.0f, elapsedTime2 = 0.0f;
    
    std::cout << "Testing SGEMM implementations with M=" << M 
              << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "Number of runs: " << numRuns << std::endl;
    
    // 测试第一种实现
    std::cout << "\nTesting sgemm_naive_kernel..." << std::endl;
    for (int i = 0; i < numRuns; ++i) {
        // 复制初始C矩阵到设备
        cudaMemcpy(devC, hostC.data(), sizeC * sizeof(float), cudaMemcpyHostToDevice);
        
        // 记录开始时间
        cudaEventRecord(start);
        
        // 启动内核
        sgemm_naive_kernel<<<gridDim, blockDim>>>(M, N, K, alpha, devA, devB, beta, devC);
        
        // 记录结束时间
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        // 计算耗时
        float msec;
        cudaEventElapsedTime(&msec, start, stop);
        elapsedTime1 += msec;
        
        // 检查错误
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error in sgemm_naive_kernel: " 
                      << cudaGetErrorString(error) << std::endl;
            return -1;
        }
    }
    
    // 将结果复制回主机以验证
    cudaMemcpy(hostC.data(), devC, sizeC * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 验证第一种实现的正确性
    if (verifyResult(hostC.data(), hostA.data(), hostB.data(), M, N, K, alpha, beta)) {
        std::cout << "sgemm_naive_kernel result verification: PASSED" << std::endl;
    } else {
        std::cout << "sgemm_naive_kernel result verification: FAILED" << std::endl;
    }
    
    // 计算平均时间
    elapsedTime1 /= numRuns;
    std::cout << "sgemm_naive_kernel average time: " << elapsedTime1 << " ms" << std::endl;
    
    // 测试第二种实现
    std::cout << "\nTesting sgemm_naive2_kernel..." << std::endl;
    for (int i = 0; i < numRuns; ++i) {
        // 复制初始C矩阵到设备
        cudaMemcpy(devC, hostC_copy.data(), sizeC * sizeof(float), cudaMemcpyHostToDevice);
        
        // 记录开始时间
        cudaEventRecord(start);
        
        // 启动内核
        sgemm_naive2_kernel<<<gridDim, blockDim>>>(M, N, K, alpha, devA, devB, beta, devC);
        
        // 记录结束时间
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        // 计算耗时
        float msec;
        cudaEventElapsedTime(&msec, start, stop);
        elapsedTime2 += msec;
        
        // 检查错误
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error in sgemm_naive2_kernel: " 
                      << cudaGetErrorString(error) << std::endl;
            return -1;
        }
    }
    
    // 将结果复制回主机以验证
    cudaMemcpy(hostC_copy.data(), devC, sizeC * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 验证第二种实现的正确性
    if (verifyResult(hostC_copy.data(), hostA.data(), hostB.data(), M, N, K, alpha, beta)) {
        std::cout << "sgemm_naive2_kernel result verification: PASSED" << std::endl;
    } else {
        std::cout << "sgemm_naive2_kernel result verification: FAILED" << std::endl;
    }
    
    // 计算平均时间
    elapsedTime2 /= numRuns;
    std::cout << "sgemm_naive2_kernel average time: " << elapsedTime2 << " ms" << std::endl;
    
    // 性能比较
    std::cout << "\nPerformance comparison:" << std::endl;
    std::cout << "sgemm_naive_kernel: " << elapsedTime1 << " ms" << std::endl;
    std::cout << "sgemm_naive2_kernel: " << elapsedTime2 << " ms" << std::endl;
    
    if (elapsedTime1 < elapsedTime2) {
        std::cout << "sgemm_naive_kernel is faster by " 
                  << (elapsedTime2 - elapsedTime1) << " ms" << std::endl;
    } else if (elapsedTime2 < elapsedTime1) {
        std::cout << "sgemm_naive2_kernel is faster by " 
                  << (elapsedTime1 - elapsedTime2) << " ms" << std::endl;
    } else {
        std::cout << "Both implementations have similar performance" << std::endl;
    }
    
    // 清理资源
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    
    std::cout << "\nTest completed successfully!" << std::endl;
    
    return 0;
}
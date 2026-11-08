#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <limits>

const int THREADS_PER_BLOCK = 256;   // 块内线程数（需是2的幂，适配分治）
const int BATCH_SIZE = 32;           // 批次大小（可自定义）
const int DATA_LEN_PER_BATCH = 1024; // 每个batch的数据长度（可自定义）

// 核函数：每个block处理1个batch，输出该batch的最大值
__global__ void batchReduceMax(const float *d_in, float *d_out, int batch_size,
                               int data_len) {
  __shared__ float s_data[THREADS_PER_BLOCK];
  int tid = threadIdx.x;
  int batch_idx = blockIdx.x; // 1个block对应1个batch

  // 初始化：读取当前batch的所有数据到共享内存（超出数据长度的线程设为最小值）
  int data_idx = tid;
  s_data[tid] = -FLT_MAX;
  while (data_idx < data_len) {
    int global_idx = batch_idx * data_len + data_idx;
    s_data[tid] = max(s_data[tid], d_in[global_idx]);
    data_idx += blockDim.x; // 线程跨步遍历batch内数据
  }
  __syncthreads();

  // 块内分治reduce求最大值
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_data[tid] = max(s_data[tid], s_data[tid + s]);
    }
    __syncthreads();
  }

  // 每个batch的最大值写入输出（blockIdx.x对应batch_idx）
  if (tid == 0) {
    d_out[batch_idx] = s_data[0];
  }
}

int main() {
  // 1. 主机端数据准备
  float *h_in = new float[BATCH_SIZE * DATA_LEN_PER_BATCH];
  float *h_out = new float[BATCH_SIZE];

  // 填充测试数据（可替换为实际数据）
  for (int b = 0; b < BATCH_SIZE; b++) {
    for (int d = 0; d < DATA_LEN_PER_BATCH; d++) {
      h_in[b * DATA_LEN_PER_BATCH + d] =
          static_cast<float>(rand() % 1000); // 0~999随机数
    }
  }

  // 2. 设备端内存分配
  float *d_in, *d_out;
  cudaMalloc(&d_in, BATCH_SIZE * DATA_LEN_PER_BATCH * sizeof(float));
  cudaMalloc(&d_out, BATCH_SIZE * sizeof(float));

  // 3. 数据拷贝到设备
  cudaMemcpy(d_in, h_in, BATCH_SIZE * DATA_LEN_PER_BATCH * sizeof(float),
             cudaMemcpyHostToDevice);

  // 4. 启动核函数：block数=batch_size，每个block处理1个batch
  dim3 blockDim(THREADS_PER_BLOCK);
  dim3 gridDim(BATCH_SIZE);
  batchReduceMax<<<gridDim, blockDim>>>(d_in, d_out, BATCH_SIZE,
                                        DATA_LEN_PER_BATCH);
  cudaDeviceSynchronize(); // 等待核函数执行完成

  // 5. 结果拷贝回主机并打印
  cudaMemcpy(h_out, d_out, BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << "每个batch的最大值：" << std::endl;
  for (int b = 0; b < BATCH_SIZE; b++) {
    std::cout << "batch " << b << ": " << h_out[b] << std::endl;
  }

  // 6. 释放内存
  delete[] h_in;
  delete[] h_out;
  cudaFree(d_in);
  cudaFree(d_out);

  return 0;
}

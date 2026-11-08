# CUTE SGEMM 代码解读

## 本地编译入口

原构建入口已迁移为 [commands/compile/run.sh](commands/compile/run.sh)，只编译、不运行或采集。显式指定本机支持的 `CUDA_ARCH`；例如适用时执行 `CUDA_ARCH=sm_89 bash commands/compile/run.sh`。日志位于该动作的 `output.log`，共享产物位于 `build/<CUDA_ARCH>/<Release或Debug>/`。参数、依赖与仅静态验证的边界见 [NCU 导航](../../README.md)。本文既有性能叙述不是新入口的实测结果。

## 概述

这个示例展示了如何使用 NVIDIA 的 CUTE (CUDA Template Engine) 库实现一个高效的 SGEMM (Single-precision General Matrix Multiplication)。CUTE 是一个用于构建 CUDA 内核的现代 C++ 模板库，它提供了一种更直观和灵活的方式来描述张量操作和内存布局。


## 代码结构分析

### 主要函数

1. `gemm_device`: CUDA 内核函数，执行实际的矩阵乘法计算
2. `gemm_nt`: 配置参数并启动 NT (Non-transpose, Transpose) 模式的 GEMM 计算
3. `gemm_tn`: 配置参数并启动 TN (Transpose, Non-transpose) 模式的 GEMM 计算
4. `main`: 主函数，设置矩阵大小、分配内存、调用 GEMM 函数并进行性能测试

### 模板参数

`gemm_device` 函数使用了大量的模板参数来配置计算:

- `ProblemShape`: 问题的形状 (M, N, K)
- `CtaTiler`: 线程块级别的分块大小 (BLK_M, BLK_N, BLK_K)
- `TA, TB, TC`: 矩阵 A, B, C 的数据类型
- `ASmemLayout, BSmemLayout, CSmemLayout`: 共享内存中的布局
- `AThreadLayout, BThreadLayout, CThreadLayout`: 线程块中线程的布局

### 内存层次结构

代码展示了 GPU 的三层内存层次结构:

1. **全局内存 (Global Memory)**: 通过 `make_gmem_ptr` 创建的张量
2. **共享内存 (Shared Memory)**: 通过 `make_smem_ptr` 创建的张量
3. **寄存器 (Register)**: 用于存储累加器的张量

### 数据布局配置

在 `gemm_nt` 和 `gemm_tn` 函数中，我们定义了不同的内存布局:

- `sA, sB, sC`: 共享内存布局
- `tA, tB, tC`: 线程布局

这些布局决定了数据在内存中的存储方式以及线程如何访问这些数据。

## 实现细节

从 `main` 函数执行开始解析细节：

### 基础设置

如选取设备、CUDA event 计时器、错误检查。

```cpp
    cute::device_init(0);
    GPU_Clock timer;
    CUTE_CHECK_LAST();
```

主要调用 `gemm` 函数，模板参数为 `class TA, TB, TC, Alpha, Beta`。因为 C++ 的模板参数推导机制。当调用这个函数时，编译器可以根据传入的实际参数自动推导出模板参数的类型。例如：

```cpp
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm(char transA, char transB, int m, int n, int k, Alpha alpha,
          TA const *A, int ldA, TB const *B, int ldB, Beta beta, TC *C, int ldC,
          cudaStream_t stream = 0)
```

当前会执行这个函数分支：

```cpp
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_nt(int m, int n, int k, Alpha alpha, TA const *A, int ldA,
             TB const *B, int ldB, Beta beta, TC *C, int ldC,
             cudaStream_t stream = 0)
```

通过这些参数初始化一些 shape/layout （见[Layout](../layout/layout.md），最后启动核函数。

```cpp
  dim3 dimBlock(size(tC));
  dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>(prob_shape, cta_tiler, A, dA,
                                                sA, tA, B, dB, sB, tB, C, dC,
                                                sC, tC, alpha, beta);
```

### 1. 张量创建和分块

```cpp
// 创建表示完整矩阵的张量
Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA); // (M,K)
Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB); // (N,K)
Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC); // (M,N)

// 获取当前线程块对应的分块
auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _); // (m,n,k)
Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // (BLK_M,BLK_K,k)
Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // (BLK_N,BLK_K,k)
Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BLK_M,BLK_N)
```

### 2. 共享内存分配和张量创建

```cpp
// 共享内存缓冲区
__shared__ TA smemA[cosize_v<ASmemLayout>];
__shared__ TB smemB[cosize_v<BSmemLayout>];
Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (BLK_M,BLK_K)
Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // (BLK_N,BLK_K)
```

### 3. 线程分区

```cpp
// 将全局内存和共享内存的分块分配给线程
Tensor tAgA = local_partition(gA, tA, threadIdx.x); // (THR_M,THR_K,k)
Tensor tAsA = local_partition(sA, tA, threadIdx.x); // (THR_M,THR_K)

Tensor tBgB = local_partition(gB, tB, threadIdx.x); // (THR_N,THR_K,k)
Tensor tBsB = local_partition(sB, tB, threadIdx.x); // (THR_N,THR_K)
```

### 4. 主计算循环

```cpp
// K方向分块循环
auto K_TILE_MAX = size<2>(tAgA);
for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
    // 将数据从全局内存复制到共享内存
    copy(tAgA(_, _, k_tile), tAsA);
    copy(tBgB(_, _, k_tile), tBsB);
    
    __syncthreads(); // 同步确保所有线程都完成了数据复制
    
    // 执行计算
    gemm(tCsA, tCsB, tCrC);
    
    __syncthreads(); // 等待所有线程完成读取共享内存
}
```

### 5. 后处理 (Epilogue)

```cpp
// 应用 alpha 和 beta 缩放因子
axpby(alpha, tCrC, beta, tCgC);
```

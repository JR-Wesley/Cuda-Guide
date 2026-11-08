# CUTLASS GEMM 示例解读

## 本地编译入口

原构建入口已迁移为 [commands/compile/run.sh](commands/compile/run.sh)，只编译、不运行。当前 CMake 仍写死 `sm_89`，因此仅在本机环境适用时执行 `CUDA_ARCH=sm_89 bash commands/compile/run.sh`。命令、输出与退出码写入 `commands/compile/output.log`，产物位于 `build/sm_89/Release/`（Debug 独立目录）。既有 CMake 的 Debug 表达式和编译参数问题未在本次修复；详见 [NCU 导航与限制](../../README.md)。本次未构建或实测，不把原有 build 当作新入口通过证据。

> 代码参考：`.cucutlass/examples/08_turing_tensorop_gemm/turing_tensorop_gemm`

## 代码解读

### 概述

代码演示了使用 Tensor Core 进行矩阵乘法运算，其中：
- 矩阵 A 和 B 使用 8 位有符号整数（int8_t）
- 累加器和输出矩阵 D 使用 32 位整数（int32_t）
- 使用 Turing 架构（SM75）的 Tensor Core 指令

CUTLASS 将内核划分为层次化的可组合部分：
1. **线程级（Thread-level）**：每个线程计算其自己的 tile 大小
2. **Warp级（Warp-level）**：多个线程 tile 组合成 warp tile
3. **线程块级（Threadblock-level）**：多个 warp tile 组合成 threadblock tile

### 模板参数配置

代码中定义了多个模板参数来配置 GEMM 内核：

数据类型配置

```cpp
using ElementAccumulator = int32_t;     // 累加器数据类型
using ElementComputeEpilogue = int32_t; // 后处理操作的数据类型
using ElementInputA = int8_t;           // 矩阵 A 的元素类型
using ElementInputB = int8_t;           // 矩阵 B 的元素类型
using ElementOutput = int32_t;          // 输出矩阵 D 的元素类型
```

矩阵布局配置
```cpp
using LayoutInputA = cutlass::layout::RowMajor;    // 矩阵 A 使用行主序
using LayoutInputB = cutlass::layout::ColumnMajor; // 矩阵 B 使用列主序
using LayoutOutput = cutlass::layout::RowMajor;    // 输出矩阵使用行主序
```

硬件架构和操作类配置
```cpp
using MMAOp = cutlass::arch::OpClassTensorOp; // 使用 Tensor Core 操作
using SmArch = cutlass::arch::Sm75;           // 针对 Turing 架构 (SM75)
```

Tile 大小配置
```cpp
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 64>; // 线程块 tile: 128x256x64
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 64>;          // Warp tile: 64x64x64
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 16>;              // MMA 操作 tile: 8x8x16
```

其他配置
```cpp
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // 线程块调度方式
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<...>; // 后处理操作
constexpr int NumStages = 2; // MMA 流水线阶段数
```

### 数据配置

输入数据类似这样声明，维护了 host device memory：

```cpp
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
      problem_size.mk()); // <- Create matrix A with dimensions M x K
```

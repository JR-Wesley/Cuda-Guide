# 只构建 PTX 虚拟指令

## 背景

PTX 是 CUDA 的虚拟指令表示，不是当前 GPU 直接执行的最终机器码。Driver 加载它时需要即时编译（JIT），所以“nvcc 能生成 PTX”不等于“当前驱动能读懂这个版本”。源码与 cubin 构建共用 `../module-loading/image.cu` 和 `kernel.cuh`，不复制。

## 1. 底层命令

工作目录：`cuda-sdk-lab/experiments/module-ptx-build/`。

```bash
mkdir -p ../../build/sm_89/module-loading
nvcc -std=c++17 -O2 -arch=compute_89 --ptx \
  ../module-loading/image.cu -o ../../build/sm_89/module-loading/image.ptx
```

`--ptx` 停在虚拟指令生成阶段；`compute_89` 是虚拟目标，而不是 `sm_89` 机器码。`-O2` 优化、`-std=c++17` 选择 C++ 标准、`-o` 指定文件。PTX 的 `.version` 由工具链生成，不能靠手改版本号保证兼容。

## 2. 独立入口

```bash
CUDA_ARCH=sm_89 bash run.sh
```

脚本把显式的 `CUDA_ARCH` 转为 `compute_89`，只构建这一个文件；可用 `NVCC` 指定编译器。不切目录、不运行、不自动构建 Driver。失败删除自身失败产物并返回真实退出码。下一步是 [Driver 主程序构建](../module-driver-build/README.md) 与 [Driver-PTX 运行](../module-driver-ptx/README.md)。

## 3. 本次观察与练习

2026-11-08，nvcc 13.4.92 构建退出 0：[状态](log/status.tsv)、[命令](log/build.sh)、[输出](log/build.log)。随后独立运行在加载阶段失败：驱动报告不支持 `.version 9.4`、当前只支持 `9.2`，详见运行目录原始证据。这不是 PTX 构建失败。

本动作 `log/` 重跑覆盖同名记录，不生成归档。练习：打开生成的 `image.ptx` 找到 `.version`、`.target`、`add_one`，解释为什么目标架构匹配仍可能无法 JIT；不要通过删错误检查或改 PTX 头部“修复”运行。

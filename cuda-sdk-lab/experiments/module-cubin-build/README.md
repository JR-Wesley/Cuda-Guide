# 只构建 cubin 原生镜像

## 背景

cubin 是含 GPU 机器码的设备镜像，不是能从 shell 直接执行的 CPU 程序。`../module-loading/image.cu` 只包含共享的 `kernel.cuh`；`extern "C"` 保持函数名 `add_one`，方便 Driver 按名字查找。先读 [64 元素例子](../module-loading/README.md)。

## 1. 底层命令

工作目录：`cuda-sdk-lab/experiments/module-cubin-build/`。

```bash
mkdir -p ../../build/sm_89/module-loading
nvcc -std=c++17 -O2 -arch=sm_89 --cubin \
  ../module-loading/image.cu -o ../../build/sm_89/module-loading/image.cubin
```

`--cubin` 只产生设备机器码镜像；`-arch=sm_89` 明确目标 GPU 架构；`-O2` 开启优化；`-std=c++17` 指定语言标准；`-o` 为输出文件。它不产生 `driver`。

## 2. 记录入口与下一步

```bash
CUDA_ARCH=sm_89 bash run.sh
```

可用 `NVCC` 指定编译器。脚本只编译，不切目录；删除自身旧输出，失败则移除不完整文件并保留真实退出码。再单独构建 [Driver 主程序](../module-driver-build/README.md)，最后进入 [Driver-cubin](../module-driver-cubin/README.md) 运行，不能把构建成功当作校验成功。

## 3. 实测与练习

2026-11-08，nvcc 13.4.92、`sm_89` 构建退出 0，见 [状态](log/status.tsv)、[命令](log/build.sh)、[输出](log/build.log)。本动作的 `log/` 就地保存记录，重跑覆盖同名文件；产物仍共享于 `build/<arch>/module-loading/`。

练习：保持源码和 `-O2` 不变，与 [PTX 构建](../module-ptx-build/README.md) 对比目标和输出选项。说明哪个文件已经是机器码，哪个仍需要驱动编译；不据文件大小推断性能。

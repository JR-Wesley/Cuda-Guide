# 只构建 Driver 主程序

## 背景

Driver API 允许 CPU 主程序在运行时选择 GPU 镜像。`../module-loading/driver.cpp` 读取文件、创建当前上下文（CUDA 操作所属的设备状态）、加载模块、找到 `add_one`、传递设备指针并校验 64 个整数。它本身不编译 kernel，故和镜像构建分离。

## 1. 底层构建

工作目录：`cuda-sdk-lab/experiments/module-driver-build/`。

```bash
mkdir -p ../../build/sm_89/module-loading
nvcc -std=c++17 -O2 -cudart=none ../module-loading/driver.cpp \
  -L/opt/cuda/lib64/stubs -lcuda -o ../../build/sm_89/module-loading/driver
```

`-std=c++17` 选择 C++ 标准，`-O2` 优化 CPU 代码；`-cudart=none` 不链接不需要的 CUDA Runtime。`-L` 是链接器找库的位置，`-lcuda` 链接 Driver API；stub 是仅用于链接的占位库，运行必须使用真实驱动，不能把 stub 目录加入 `LD_LIBRARY_PATH`。没有 `-arch` 是因为本文件只含 CPU 代码；`CUDA_ARCH` 在此用于选共享产物目录。

## 2. 独立入口

```bash
CUDA_ARCH=sm_89 bash run.sh
```

可指定 `NVCC`、`CUDA_STUB_DIR`，后者默认 `/opt/cuda/lib64/stubs`。脚本不切目录、不会构建或运行镜像；失败会清除自身失败输出并返回实际退出码。后续分别执行 [cubin 构建](../module-cubin-build/README.md) 和 [Driver-cubin 运行](../module-driver-cubin/README.md)，或选择 [PTX 构建](../module-ptx-build/README.md) 和 [Driver-PTX 运行](../module-driver-ptx/README.md)。

## 3. 真实证据与练习

2026-11-08，nvcc 13.4.92 构建退出 0：[状态](log/status.tsv)、[命令](log/build.sh)、[输出](log/build.log)。源码只做必要修复：JIT 日志长度选项按接口要求传整数转成的 `void*`，不再传长度变量的地址；保留错误检查和 JIT 原文。

日志位于本动作 `log/`，重跑覆盖同名文件。编译成功不保证镜像可加载。练习：解释为什么同一个 `driver` 无须重新编译，就能分别传入 cubin 和 PTX；再检查两次运行究竟在哪个 API 开始出现区别。

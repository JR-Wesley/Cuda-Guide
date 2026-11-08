# 只构建共享故障 debug 程序

## 背景

为了把工具诊断对应到具体源码，本例用带设备调试信息的构建；它不用于测性能。四个故障仍共用 [cases.cu](../sanitizer/cases.cu)，每次运行只选择一个 kernel，不能在四个检查器目录各编译一份。

## 1. 底层命令

工作目录：`cuda-sdk-lab/experiments/sanitizer-build/`。

```bash
mkdir -p ../../build/sm_89/sanitizer
nvcc -std=c++17 -G -cudart=shared \
  -gencode arch=compute_89,code=sm_89 ../sanitizer/cases.cu \
  -o ../../build/sm_89/sanitizer/cases
```

`-G` 生成 GPU 调试信息并关闭常规设备优化，本例不再叠加重复的 `-lineinfo`。`-cudart=shared` 使用共享 Runtime 库，避免大静态可执行文件；`-gencode` 只内嵌指定架构的原生机器码，不依赖 PTX JIT。`-std=c++17` 指定语言标准，`-o` 指定输出。

## 2. 入口

```bash
CUDA_ARCH=sm_89 bash run.sh
```

可用 `NVCC` 指定工具。脚本不切目录、只编译；构建前删除自身旧输出，失败清理不完整产物并返回实际退出码。修改源码或参数后手动重编译。

随后手动进入 [memcheck](../sanitizer-memcheck/README.md)、[racecheck](../sanitizer-racecheck/README.md)、[initcheck](../sanitizer-initcheck/README.md)、[synccheck](../sanitizer-synccheck/README.md) 中的一个；也可先到 [静态检查](../sanitizer-static/README.md) 读取机器指令。这里不会自动执行任何一个后续动作。

## 3. 实测与练习

2026-11-08，nvcc 13.4.92、`sm_89` 构建退出 0：[命令](log/build.sh)、[原始输出](log/build.log)、[状态](log/status.tsv)。编译无输出是正常的，是否检出故障必须另看动态报告。

本动作 `log/` 由共享 `record.sh` 记录；重跑覆盖同名文件，无归档。练习：解释为什么只需这一个 debug 二进制就能做四种检查，以及为什么 debug 指令不能拿来比较优化后性能。

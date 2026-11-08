# 3. 只生成带重定位信息的 cubin

## 背景与目标

cubin 是指定 GPU 架构的设备二进制，通常采用 ELF 文件格式，包含 SASS 机器指令。与 PTX 不同，它已针对真实架构完成机器码生成，但不是能直接在 CPU 上启动的程序。

## 底层命令与解释

工作目录：`cuda-sdk-lab/experiments/compile-cubin/`。

```bash
mkdir -p ../../build/sm_89/compile/artifacts
nvcc -std=c++17 -O3 -lineinfo -gencode arch=compute_89,code=sm_89 \
  -Xptxas=-v,--preserve-relocs --cubin ../compile/compiler_probe.cu \
  -o ../../build/sm_89/compile/artifacts/probe.cubin
```

`--cubin` 只输出设备二进制。`-gencode` 固定虚拟架构与机器目标。`-Xptxas` 把逗号分隔的选项传给 PTX 汇编器：`-v` 打印静态资源，`--preserve-relocs` 保留重定位信息（代码中地址引用与目标之间的关联），供后续 `nvdisasm` 分析使用。它不等于启用跨文件 RDC。

`-std=c++17 -O3 -lineinfo` 分别固定语言、优化与源码行信息。脚本不调用 `nvdisasm`，编译和检查是两件事。

## 独立脚本

```bash
CUDA_ARCH=sm_89 bash run.sh
```

实际命令在 `log/cubin.sh`，诊断在 `log/cubin.log`，原始状态在 `log/status.tsv`。`timeout 60s` 超时返回 124。重跑只覆盖自身日志和 `probe.cubin`；失败删除部分输出，不删除 PTX 或其他动作产物。

## 当前日志的实际观察

Toolkit 13.4.92：`cubin` 退出 **0**；[probe.cubin](../../build/sm_89/compile/artifacts/probe.cubin) 为 **5,416 字节**。[cubin.log](log/cubin.log) 报告 `saxpy_probe` 针对 `sm_89` 使用 **10 个寄存器**、**0 字节 stack frame**、**0 字节 spill stores/loads**、**376 字节 cmem[0]**。

这些是编译器的静态资源报告；日志里的 `Compile time` 是编译时间，不是 kernel 执行时间。这里没有反汇编或测量占用率。

## 边界与练习

先预测 PTX 中虚拟寄存器声明数量是否必须等于这里的 10，再解释寄存器分配为何可能复用物理寄存器。后续独立检查可以读取同一路径，使用 `nvdisasm -g ../../build/sm_89/compile/artifacts/probe.cubin`，但这不是本动作已执行的步骤。

保留重定位信息不保证任意版本反汇编器都能解析新架构；更不证明 GPU 运行正确。下一步：[打包 fatbin](../compile-fatbin/README.md)。

# 5. 只编译主机目标文件

## 背景与目标

CPU 编译器不能直接执行 `<<<...>>>` 启动语法。`nvcc` 先生成主机侧启动代码，再把 CPU 代码和内嵌设备镜像放进 `.o` 目标文件。目标文件仍需主机链接才能成为程序。

## 底层命令与解释

工作目录：`cuda-sdk-lab/experiments/compile-object/`。

```bash
mkdir -p ../../build/sm_89/compile/artifacts
nvcc -std=c++17 -O3 -lineinfo \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_89,code=compute_89 \
  -c ../compile/compiler_probe.cu -o ../../build/sm_89/compile/artifacts/probe.o
```

`-c` 表示编译但不做最终主机链接；它不代表只编译 CPU 部分。两条 `-gencode` 请求原生设备代码及 PTX。`-std=c++17`、`-O3`、`-lineinfo` 分别固定语言、优化与设备行信息。

这里 kernel 与调用者在一个源文件，未启用可重定位设备代码；后面的 RDC 专题才引入 `-dc`。

## 独立脚本

```bash
CUDA_ARCH=sm_89 bash run.sh
```

`log/object.sh` 是命令，`log/object.log` 是诊断，`log/status.tsv` 是状态。命令限时 60 秒，超时返回 124。只覆盖自身日志和 `probe.o`，失败删除部分输出；不调用链接动作。

## 当前日志的实际观察

Toolkit 13.4.92：`object` 退出 **0**，[object.log](log/object.log) 为空；[probe.o](../../build/sm_89/compile/artifacts/probe.o) 为 **14,520 字节**。这是已生成目标文件的证据，不是程序已运行的证据。

后续可单独用 `readelf -W -S` 查看 ELF 节、用 `nm -C` 查看主机符号（`-C` 将 C++ 修饰名还原为可读名字）；这些检查不塞进编译脚本。

## 边界与练习

解释为什么 `.o` 比独立 cubin 大，却仍不能直接启动。然后回答：若只修改 CPU 侧打印文字，需要重新做 PTX 动作吗？需要重建主机目标文件和程序吗？

下一步：[主机链接](../compile-host-link/README.md)。修改源码后必须重新执行本动作，再显式执行链接；脚本不会自动判定旧下游文件是否过期。

# 1. 只列出编译链：dryrun

## 背景与目标

`nvcc` 是编译驱动，不是一个独自完成所有工作的编译器。它把同一 `.cu` 中的 CPU 代码和 GPU 代码交给不同工具，再组合产物。先读 [compiler_probe.cu](../compile/compiler_probe.cu)：GPU 计算 `y[i] = 2*x[i]+y[i]`，CPU 分配内存、传输数据并检查结果。这里不运行它。

## 底层命令与解释

工作目录：`cuda-sdk-lab/experiments/compile-dryrun/`。

```bash
mkdir -p ../../build/sm_89/compile/artifacts
nvcc -std=c++17 -O3 -lineinfo \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_89,code=compute_89 \
  -cudart=shared --dryrun ../compile/compiler_probe.cu \
  -o ../../build/sm_89/compile/artifacts/probe_dryrun
```

- `-std=c++17` 选择 C++ 语言版本；`-O3` 请求优化；`-lineinfo` 保留设备指令与源码行的对应关系，不等于调试构建。
- `compute_89` 指虚拟架构，`sm_89` 指真实机器目标；两条 `-gencode` 请求原生机器码和 PTX 虚拟指令两个镜像。
- `-cudart=shared` 请求动态 CUDA Runtime；`-o` 指定拟输出文件。
- `--dryrun` 只打印拟执行的子命令，不执行它们。日志出现 `-E`、`.ii` 或 `/tmp/` 名称不代表生成了那些文件。不要逐行执行这份内部计划。

## 独立脚本

```bash
CUDA_ARCH=sm_89 bash run.sh
```

脚本不切换目录；`record` 来自 `../record.sh`。`timeout 60s` 给每条命令 60 秒上限，超时退出码为 124；通常返回编译器原始退出码。`log/dryrun.sh` 是可在本目录重放的外层命令，`log/dryrun.log` 是输出，`log/status.tsv` 是状态。重跑覆盖当前动作同名日志，不创建历史目录。

## 当前日志的实际观察

Toolkit **13.4.92**、目标 `sm_89`：`dryrun` 退出 **0**。在 [dryrun.log](log/dryrun.log) 中依次辨认：

1. `g++ -E`：预处理；`cudafe++`：转换主机侧 CUDA 语法和启动桩。
2. `cicc`：产生 PTX；`ptxas`：生成 `sm_89` 的 cubin。
3. `fatbinary`：打包 PTX 与机器码；`g++ -c`：产生主机目标文件。
4. `nvlink` 和最终 `g++` 链接；最终行使用 `-lcudart` 而不是 `-lcudart_static`。

本次没有生成 `probe_dryrun`，也没有 `keep/`。这份日志是**计划证据**，不是这些子工具已实际执行的证据。

## 边界与练习

先找出计划里两处 `fatbinary`，解释一个为什么与设备链接有关。再预测只保留 `code=sm_89` 时，哪个 PTX 打包参数会消失；手动修改 dryrun 命令验证，不改变其他编译动作。

`--keep` 保存内部中间文件，`-E` 输出预处理文本；它们常很大，本课程只介绍，不执行。下一步：[生成 PTX](../compile-ptx/README.md)。

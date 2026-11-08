# 只构建一个小型工具输入

## 为什么先单独编译

分析器处理的是可执行文件，而不是 `.cu` 文本。本动作只将共享 [vector_transpose.cu](../../examples/vector_transpose.cu) 编译一次，供运行、Sanitizer 和分析器复用；不启动 GPU，不采集性能。

源码保留向量加法与两个转置 kernel，但默认工具课程只用 257 个元素。不要为了学习命令建立大型应用或复制一套源码。

## 1. 看底层命令

当前目录：`cuda-sdk-lab/experiments/build/`。本机实测目标为 `sm_89`；其他设备需相应修改两种架构写法。

```bash
mkdir -p ../../build/sm_89
nvcc -std=c++17 -O3 -lineinfo -cudart=shared -Xptxas=-v \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_89,code=compute_89 \
  ../../examples/vector_transpose.cu -o ../../build/sm_89/vector_transpose
```

- `-std=c++17` 选择语言版本；`-O3` 指定主机优化等级，设备编译有自身优化控制。
- `-lineinfo` 保留设备源码行，不是 `-G` 调试构建。
- `-cudart=shared` 动态链接运行库，避免把完整静态 Runtime 填进每个教学程序；运行时需要可见的真实动态库。本机通过 `LD_LIBRARY_PATH=/opt/cuda/lib64` 找到它。
- 两条 `-gencode` 分别嵌入原生机器码与 PTX 后备镜像；PTX 后备存在不代表当前驱动一定能 JIT 它。
- `-Xptxas=-v` 输出设备资源统计，`-o` 指定唯一共享可执行文件。

## 2. 用脚本记录同一个构建动作

```bash
CUDA_ARCH=sm_89 bash run.sh
```

本动作只覆盖自己的日志和共享程序，失败时删除部分输出，之后不得继续使用旧文件。不会生成持久化预处理 `.ii`、`keep/` 或源码快照。

## 3. 本次观察

本机 Toolkit 13.4.92 编译成功：[命令](log/build.sh)、[原始输出](log/build.log)、[退出码](log/status.tsv)。程序大小 **44,984 字节**，不是大型基准框架。

`ptxas` 报告 `vector_add` 使用 12 个寄存器；`transpose_tiled` 使用 4,224 字节共享内存。后者可先手算 `32×33×sizeof(float)=4224`，再对照源码。静态资源数不证明性能，也不等于分析器所有运行时字段的口径。

下一步进入 [execution](../execution/README.md) 单独运行。练习：解释为什么编译成功不能证明结果正确，以及只改变输入长度为何无需重新构建。

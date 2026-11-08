# 2. 只生成 PTX

## 背景与目标

PTX 是 GPU 的虚拟指令文本，还需要转换成具体 GPU 的机器指令。这里从同一份 [compiler_probe.cu](../compile/compiler_probe.cu) 出发，停在 PTX 阶段；不依赖 dryrun 的任何产物。

先手算：`x[i]=1`、`y[i]=1`、`a=2` 时应得到 `3`。这个预期结果用来理解源码，不是本次 GPU 实测。

## 底层命令与解释

工作目录：`cuda-sdk-lab/experiments/compile-ptx/`。

```bash
mkdir -p ../../build/sm_89/compile/artifacts
nvcc -std=c++17 -O3 -lineinfo -arch=compute_89 --ptx \
  ../compile/compiler_probe.cu -o ../../build/sm_89/compile/artifacts/probe.ptx
```

`--ptx` 选择输出阶段；`-arch=compute_89` 指定虚拟架构，不请求可执行程序。`-std=c++17` 固定语言版本，`-O3` 请求优化，`-lineinfo` 让 PTX 的 `.loc` 对应源码行。PTX 中的寄存器名字是虚拟寄存器，不能直接当作硬件寄存器数量。

## 独立脚本

```bash
CUDA_ARCH=sm_89 bash run.sh
```

`CUDA_ARCH` 必须显式设置，脚本把 `sm_89` 转成 `compute_89`。本目录 `log/ptx.sh` 记录实际命令，`log/ptx.log` 保存输出，`log/status.tsv` 保存退出码。每条命令用 `timeout 60s` 限时，超时为 124。只覆盖本动作日志和 `probe.ptx`，失败删除部分输出，不运行程序。

## 当前日志的实际观察

Toolkit 13.4.92：`ptx` 退出 **0**，[ptx.log](log/ptx.log) 为空是成功时没有诊断，不是没有执行。产物 [probe.ptx](../../build/sm_89/compile/artifacts/probe.ptx) 为 **1,266 字节**。

实际文本包含 `.version 9.4`、`.target sm_89`、`.entry saxpy_probe`。`mad.lo.s32` 对应线程下标计算；`setp.ge.s32` 和条件跳转对应 `i < n` 的边界；`mul.wide.s32 ..., 4` 把 float 元素下标换成字节偏移；`fma.rn.f32` 对应乘加。

## 边界与练习

分别在源码和 PTX 找出 `x`、`y` 两个地址参数及元素偏移。为什么下标增加 1，地址增加 4？如果 `i >= n`，哪条跳转避免越界访问？

生成 PTX 不证明驱动能 JIT（即时编译）这个 PTX 版本；没有验证 GPU 加载或结果。下一步：[独立生成 cubin](../compile-cubin/README.md)，它重新读取源码，不读取这份 PTX。

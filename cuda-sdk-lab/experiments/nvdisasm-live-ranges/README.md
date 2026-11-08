# 寄存器活跃区间：一个值什么时候还不能丢？

## 背景与目标

寄存器保存计算的中间值。如果当前值以后还可能被读取，它就是“活跃”的；最后一次使用之后，该寄存器可以被别的值复用。活跃区间帮助理解寄存器压力，但不是执行耗时，也不是编译器分配的总寄存器数。

前置：先读 [SASS](../nvdisasm-sass/README.md)，知道指令会读/写寄存器；复用 [编译专题](../compile/README.md) 带重定位信息的 `probe.cubin`。这里不重编、不提取、不运行 GPU。

## 1. 底层命令

在 `cuda-sdk-lab/experiments/nvdisasm-live-ranges/` 中：

```bash
nvdisasm --print-code --print-life-ranges ../../build/sm_89/compile/artifacts/probe.cubin
```

`--print-code` 只保留代码节；`--print-life-ranges`（`-plr`）在指令旁追加寄存器活跃情况。使用默认宽表格即可，不先添加统计或解析框架。

本机帮助明确说明：活跃区间输出会忽略 `--print-line-info` 等行号选项，所以它是独立动作，不与 SASS/行号阅读强行合并。

## 2. 记录脚本

```bash
CUDA_ARCH=sm_89 bash run.sh
```

默认输入 `../../build/<CUDA_ARCH>/compile/artifacts/probe.cubin`；唯一可选参数是另一个独立 cubin。`CUDA_ARCH` 必填，`NVDISASM` 可指定工具路径。脚本按本动作目录解释路径，不切目录、不编译。

查看 `log/live_ranges.log` 及 `log/tool_version.log`；实际命令在同名 `.sh`，真实退出码在 `log/status.tsv`。失败即停，重跑覆盖同名输出。若数据流分析失败，先核对 cubin 的编译记录和重定位保留条件，不以 `--no-dataflow` 绕过后宣称得到了有效活跃区间。

## 3. 阅读表格

先读输出自带图例，再挑一个通用寄存器列：找到写入、后续读取、活跃延续及最终不再使用的位置。观察 `#` 活跃计数列如何变化，并分别看通用寄存器、谓词等不同寄存器类别，不把所有标记混为一个 `REG` 数字。

[静态资源](../binary-resources/README.md) 的 `REG` 描述每线程分配量；这里描述某个程序位置需要保留哪些值。两者不必相等，寄存器编号上界也不等于此处活跃数量。工具推导的是静态数据流，不是硬件采样，不能由此直接宣称溢出、occupancy 或性能瓶颈。

## 4. 练习

先选一个只在局部运算中使用的寄存器，预测它在哪次读取后失活，再沿表格检查。如果同一个寄存器随后又被写入，解释为什么这可以是新值的区间，而不是旧值一直活跃。不要通过修改源码让表格“符合预测”。

## 本轮验证

共享 cubin 生成后，已用 `CUDA_ARCH=sm_89 bash run.sh`、`nvdisasm` 13.4.92 离线检查，见 [活跃区间表](log/live_ranges.log) 与 [退出码 0](log/status.tsv)。实际图例中 `^` 是赋值、`v` 是使用、`x` 是使用并重新赋值、`:` 是仍在使用。输出分别列出 GPR（通用寄存器）、PRED（谓词）、UGPR（uniform 通用寄存器）；本次 GPR 的 `#` 最大为 5，而资源分配为 `REG:10`。例如 `FFMA` 行的 `R7` 标成 `x`，旧值作为加数被读取，新值又写回该寄存器。没有测量运行性能。

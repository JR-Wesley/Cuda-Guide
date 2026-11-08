# SASS 与源码行：把机器指令对应回一个表达式

## 背景与目标

看到机器指令后，还需要知道它大致来自哪行源码。`nvdisasm` 读取独立 cubin，可以把其中已有的源码行信息标在 SASS 旁。本节把 SASS 和行号放在同一个阅读动作，不做 CFG 或寄存器区间分析。

前置：知道 [PTX 与 SASS](../binary-sass/README.md) 不同；按 [编译专题](../compile/README.md) 显式准备 `probe.cubin`。编译需要 `-lineinfo`（保留行信息，不等于 `-G` 调试构建）；本动作不补编译、不提取镜像。

## 1. 底层命令

在 `cuda-sdk-lab/experiments/nvdisasm-sass/` 中：

```bash
nvdisasm --print-code --print-line-info ../../build/sm_89/compile/artifacts/probe.cubin
```

- `--print-code`（`-c`）：只打印代码节，避免大段无关数据。
- `--print-line-info`（`-g`）：读取 cubin 的 `.debug_line`，显示已有源码行映射。这里的 `-g` 是反汇编选项，不是重新启用编译调试模式。

输入必须是独立 cubin，不要把 `probe` 主机程序或 `probe.fatbin` 传给这个入口。一般工具定位见 [Toolkit 总导览](../../TOOLKIT_GUIDE.md)，这里仅讲本动作。

## 2. 记录脚本

```bash
CUDA_ARCH=sm_89 bash run.sh
```

默认读取 `../../build/<CUDA_ARCH>/compile/artifacts/probe.cubin`；唯一可选参数可替换为另一个已有 cubin。`CUDA_ARCH` 必填，仅选择目录；`NVDISASM` 可指定工具路径。脚本必须在本目录执行，不切目录、不编译、不运行 GPU。

`log/sass_lineinfo.log` 保存反汇编，`log/tool_version.log` 保存版本；同名 `.sh` 记录命令，`status.tsv` 记录真实退出码。重跑覆盖同名文件，失败即停；缺输入先回编译专题，旧日志不代表本次成功。

## 3. 阅读顺序

1. 找 `saxpy_probe` 的代码节，观察文件名/行号注释。
2. 对照 [compiler_probe.cu](../compile/compiler_probe.cu) 中线程索引计算和 `if (i < n)` 后的乘加表达式。
3. 区分地址偏移、助记符和源码行标记；一行源码可能对应多条指令，优化也可能重排或合并指令。

以上为预期观察点，不保证每条指令都有行号。无行号先检查 cubin 是否来自带 `-lineinfo` 的构建；命令返回零不证明行号确实存在。行映射不表示执行次数或耗时。

## 4. 练习

不改变输入，手动去掉 `--print-line-info` 再看终端输出。预测变化应主要来自注释，而非重新生成一套机器码；解释为何这里没有执行 CUDA 编译。

## 本轮验证

共享 cubin 生成后，已用 `CUDA_ARCH=sm_89 bash run.sh`、`nvdisasm` 13.4.92 离线检查，见 [SASS/行号](log/sass_lineinfo.log) 与 [退出码 0](log/status.tsv)。实际显示 `compiler_probe.cu` 的第 5、6、7、8 行：第 6 行关联 `S2R` / `IMAD` 索引计算，第 7 行关联边界判断、访存和 `FFMA`，第 8 行关联退出。行号确实存在，但不等于执行跟踪；未进行 GPU 运行。

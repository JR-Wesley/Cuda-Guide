# 镜像清单：文件里装了什么设备代码？

## 背景与目标

一个 fatbin（设备代码容器）可以同时装原生机器码 cubin 和虚拟指令 PTX。先列清单，再决定读什么，不要一开始就导出整个程序。ELF 是组织二进制节与符号的文件格式；这里的 ELF 清单指容器里的设备 ELF 镜像，不是主机程序的节清单。

前置：知道“编译生成文件”与“执行程序”不同；按 [编译专题](../compile/README.md) 显式生成 `build/<arch>/compile/artifacts/probe.fatbin`。本动作不编译、不启动 GPU。

## 1. 先看底层命令

以下命令均在 `cuda-sdk-lab/experiments/binary-list/` 内执行：

```bash
cuobjdump --list-elf ../../build/sm_89/compile/artifacts/probe.fatbin
cuobjdump --list-ptx ../../build/sm_89/compile/artifacts/probe.fatbin
```

`--list-elf` 列原生镜像，`--list-ptx` 列 PTX 镜像。它们回答同一个清单问题，但必须分别调用：不要把两个 list 选项拼进一个命令，选项会互相覆盖。`sm_89` 是本轮输入目录，命令不会重新指定或转换文件内架构。

## 2. 用独立脚本记录同样的动作

```bash
CUDA_ARCH=sm_89 bash run.sh
# 可选：只替换一个输入，仍然只列清单
CUDA_ARCH=sm_89 bash run.sh ../../build/sm_89/compile/artifacts/probe
```

无参数默认读 `probe.fatbin`；唯一可选参数是文件，不是目录。路径按当前目录解释，脚本不切目录。`CUDA_ARCH` 必填，`CUOBJDUMP` 可指定工具路径。第二次调用会覆盖本动作同名日志，不自动保留第一次对照。

`log/tool_version.log` 记录工具版本，`log/list_elf.log`、`log/list_ptx.log` 保存输出；同名 `.sh` 是实际命令，`log/status.tsv` 是真实退出码。失败即停止；文件缺失时提示先编译，不伪装成空清单。旧日志是否属于本次以状态表为准。

## 3. 怎么读输出

预期（待产物生成后核对）：同时包含原生和 PTX 的 fatbin 会在两份清单中各有条目。记录镜像名，再用 [PTX](../binary-ptx/README.md) 或 [SASS](../binary-sass/README.md) 动作读内容。镜像数不等于 kernel 数，退出码为零也不等于清单非空；空 ELF 清单还需要检查 PTX。

清单不能证明驱动选择了哪份镜像，更不能证明 JIT（运行时编译）成功。架构与 PTX 版本还要读 dump 中的头部。

## 4. 练习

先预测 `probe.o` 与 `probe.fatbin` 是否都能列出设备镜像，再只改变脚本的文件参数为 `../../build/sm_89/compile/artifacts/probe.o`。如果结果不符，先确认该目标文件的实际编译参数，而不是运行程序。

## 本轮验证

共享产物生成后，已用 `CUDA_ARCH=sm_89 bash run.sh` 离线验证，`cuobjdump` 13.4.92 的两个清单命令均返回 0。实际各有一份镜像：`compiler_probe.sm_89.cubin` 与 `compiler_probe.sm_89.ptx`，见 [ELF 清单](log/list_elf.log)、[PTX 清单](log/list_ptx.log) 和 [退出码](log/status.tsv)。没有执行主机程序或 GPU kernel。统一记录约定见 [二进制导航](../binary/README.md)。

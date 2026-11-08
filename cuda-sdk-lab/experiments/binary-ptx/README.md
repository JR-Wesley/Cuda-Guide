# PTX dump：阅读容器中的虚拟指令

## 背景与目标

PTX 是 NVIDIA 的虚拟指令表示，不是 GPU 直接执行的最终机器码。本节只回答“文件里嵌入了什么 PTX”，不做提取、不验证驱动 JIT。先完成 [镜像清单](../binary-list/README.md)，确认输入包含 PTX。

输入复用 [编译专题](../compile/README.md) 的 `probe.fatbin`，不另编示例；其 kernel `saxpy_probe` 对有效索引计算 `y[i] = a * x[i] + y[i]`。

## 1. 底层命令

在 `cuda-sdk-lab/experiments/binary-ptx/` 中：

```bash
cuobjdump --dump-ptx ../../build/sm_89/compile/artifacts/probe.fatbin
```

`--dump-ptx` 把内嵌 PTX 打印到标准输出。独立的 `probe.ptx` 已经是文本，应直接打开它，而不是要求工具从中再提取 PTX。dump 带有容器头部，不要把整个日志作为独立 PTX 传给编译器。

## 2. 记录脚本

```bash
CUDA_ARCH=sm_89 bash run.sh
# 可选：读取已有的小型可执行文件，不执行它
CUDA_ARCH=sm_89 bash run.sh ../../build/sm_89/compile/artifacts/probe
```

默认输入是 `../../build/<CUDA_ARCH>/compile/artifacts/probe.fatbin`。唯一可选参数是输入文件；路径原样使用。必须从本目录启动，脚本不切目录、不隐式编译。`CUDA_ARCH` 必填，`CUOBJDUMP` 可覆盖工具路径。

输出为 `log/ptx.log`，版本为 `log/tool_version.log`；`.sh` 保存命令，`status.tsv` 保存真实退出码。重跑覆盖同名日志，任一步失败就停止；先看本次状态表，再读输出。

## 3. 阅读顺序

1. 找 `.version`（PTX 语言版本）、`.target`（目标要求）、`.address_size`（地址位数）。PTX 版本不是 Toolkit 版本。
2. 找 `.entry saxpy_probe` 与 `.param`，对应 kernel 入口和传入参数。
3. 找线程索引计算、边界判断及浮点乘加/加载/存储。PTX 寄存器是虚拟名字，数量不等于硬件寄存器分配量。

这些是预期观察点；实际形式以编译输出为准。dump 为空先回到清单检查有无 PTX，不要把“命令成功”理解为“存在 PTX”。

## 4. 练习

打开共享的 `../../build/sm_89/compile/artifacts/probe.ptx`，找到同名入口，与 `log/ptx.log` 比较头部。解释哪些行属于 PTX，哪些是 `cuobjdump` 的容器说明；不要求两个独立编译产物逐字节相同。

## 本轮验证

共享输入生成后，已用 `CUDA_ARCH=sm_89 bash run.sh`、`cuobjdump` 13.4.92 完成离线检查，退出码为 0，见 [实际 PTX](log/ptx.log) 与 [状态](log/status.tsv)。本次读到 `.version 9.4`、`.target sm_89`、`.address_size 64` 和 `.entry saxpy_probe`；边界判断为 `setp.ge.s32` 配合条件跳转，运算含 `fma.rn.f32`。这不是 GPU 执行证据。后续对照 [SASS dump](../binary-sass/README.md)，不能由 PTX 行数推断运行速度。

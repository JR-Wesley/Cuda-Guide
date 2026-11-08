# ELF 与符号：代码放在哪里，函数叫什么？

## 背景与目标

ELF 把二进制内容分成节（section）；符号表把函数或对象名称关联到这些内容。先找 kernel 符号，再读其机器指令，比盲目搜索十六进制更容易。本节把节与符号作为一个阅读任务，默认只看小型 `probe.cubin`。

前置：按 [编译专题](../compile/README.md) 准备独立 cubin。cubin 是设备 ELF；`probe.o` 则是可能携带设备镜像的主机目标文件。两层 ELF 不应混淆。

## 1. 底层命令

在 `cuda-sdk-lab/experiments/binary-elf/` 中：

```bash
cuobjdump --dump-elf ../../build/sm_89/compile/artifacts/probe.cubin
cuobjdump --dump-elf-symbols ../../build/sm_89/compile/artifacts/probe.cubin
```

`--dump-elf` 展示设备 ELF 节内容；`--dump-elf-symbols` 单独列符号名称。前者可能很长，所以只用小 probe，不默认 dump 大型转置程序，也不建立输出解析器。

## 2. 记录脚本

```bash
CUDA_ARCH=sm_89 bash run.sh
```

默认输入 `../../build/<CUDA_ARCH>/compile/artifacts/probe.cubin`；唯一可选参数是一个输入文件。必须在本目录调用，脚本不切目录。`CUDA_ARCH` 必填；`CUOBJDUMP` 可替换工具路径；缺文件时提示显式编译，不隐式生成输入。

输出在 `log/elf.log`、`log/symbols.log`；工具版本在 `log/tool_version.log`，同名 `.sh` 保存实际命令，`log/status.tsv` 保存真实退出码。失败即停止，重跑覆盖同名文件；旧日志不代表本次成功。

## 3. 阅读节与符号

预期寻找 `.text.saxpy_probe`（函数机器码）、相关 `.nv.info`（CUDA 元数据）以及符号 `saxpy_probe`。真实节集合以工具输出为准。该示例使用 `extern "C"` 避免函数名被 C++ 改写；其他源码的实际符号名未必等于代码里写的名字。

符号表中“存在函数”不能证明执行过它；重定位信息用于描述链接/分析时需要解释的地址关系，不是 GPU 实测。独立 cubin 供后续 `nvdisasm` 使用，无需从主机文件提取。

## 4. 练习：区分主机层与设备层

若 `probe.o` 已生成，手动执行以下小范围阅读命令（不是脚本的隐藏步骤）：

```bash
readelf -W -S ../../build/sm_89/compile/artifacts/probe.o
nm -C ../../build/sm_89/compile/artifacts/probe.o
```

`readelf -S` 列节头而非整段内容，`-W` 避免长行折断；`nm` 列主机符号，`-C` 将 C++ 改写名称还原成人类可读形式。找设备容器相关的节，并解释这里为什么还会出现主机启动相关符号。系统没有这些 GNU 工具时先停在设备 ELF 阅读，不安装依赖。

## 本轮验证

输入生成后，已用 `CUDA_ARCH=sm_89 bash run.sh`、`cuobjdump` 13.4.92 离线检查，两条命令退出码均为 0，见 [状态](log/status.tsv)。[ELF 输出](log/elf.log) 为 64 位 `ET_EXEC` 设备 ELF，`sm=89`，含 `.text.saxpy_probe`、`.debug_line` 及调试相关重定位节；[符号输出](log/symbols.log) 中 `saxpy_probe` 为 `STT_FUNC STB_GLOBAL STO_ENTRY`。这里的设备 `ET_EXEC` 不意味着能直接当主机程序执行。主机层练习尚未执行。

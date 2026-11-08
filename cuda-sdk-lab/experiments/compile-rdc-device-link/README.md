# 8. 只做 RDC 设备链接

## 背景与目标

分别编译得到的两个 `.o` 还保留跨文件设备引用。设备链接器 `nvlink` 把 `rdc_main.o` 中的 `rdc_scale` 调用连接到 `rdc_device.o` 中的定义。输出 `rdc_dlink.o` 用于随后主机链接，不是可执行文件。

## 底层命令与解释

工作目录：`cuda-sdk-lab/experiments/compile-rdc-device-link/`；先完成 [RDC 编译](../compile-rdc-compile/README.md)。

```bash
nvcc -std=c++17 -O3 -lineinfo -gencode arch=compute_89,code=sm_89 \
  --device-link ../../build/sm_89/compile/artifacts/rdc_device.o \
  ../../build/sm_89/compile/artifacts/rdc_main.o \
  -o ../../build/sm_89/compile/artifacts/rdc_dlink.o
```

`--device-link`（简称 `-dlink`）只进行设备链接并生成承载链接结果的主机目标文件。它不替代原始 `.o` 中的 CPU 代码。`-gencode` 指定 `sm_89`；其余选项与编译阶段保持一致。本阶段不启用 LTO。

## 独立脚本

```bash
CUDA_ARCH=sm_89 bash run.sh
```

两个输入都经 `require_file` 检查，缺失返回 1，不自动编译。命令、输出、状态为 `log/device_link.sh`、`log/device_link.log`、`log/status.tsv`。`timeout 60s` 限时，超时为 124；只覆盖本动作日志与 `rdc_dlink.o`，失败删除部分输出。

## 当前日志的实际观察

Toolkit 13.4.92：`device_link` 退出 **0**，[device_link.log](log/device_link.log) 为空；`../../build/sm_89/compile/artifacts/rdc_dlink.o` 为 **11,832 字节**。这证明本组输入成功完成设备链接，不证明 kernel 已运行。

编译日志不包含 SASS；本动作不判断 `CALL` 是否保留。需要查看指令时，交给后续独立静态检查。

## 边界与练习

预测只传 `rdc_main.o` 为什么可能报未定义设备符号，再回源码定位缺少哪个定义。无需真的破坏正常产物来验证。

改动任一输入后，必须显式重做本动作及 [主机链接](../compile-rdc-host-link/README.md)。存在的 `rdc_app` 不会自动更新，也不能当作新输入的结果。

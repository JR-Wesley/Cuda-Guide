# 12. 只做 LTO 程序的主机链接

## 背景与目标

LTO 的设备优化已经在上一步完成。现在只把原始目标文件、设备链接结果和动态运行库连成 CPU 程序；这里不再做 LTO，也不重新编译或运行。

## 底层命令与解释

工作目录：`cuda-sdk-lab/experiments/compile-lto-host-link/`；先完成 [LTO 编译](../compile-lto-compile/README.md)、[LTO 设备链接](../compile-lto-device-link/README.md)。

```bash
nvcc -arch=sm_89 -cudart=shared --no-device-link \
  ../../build/sm_89/compile/artifacts/rdc_device_lto.o \
  ../../build/sm_89/compile/artifacts/rdc_main_lto.o \
  ../../build/sm_89/compile/artifacts/rdc_lto_dlink.o \
  -o ../../build/sm_89/compile/artifacts/rdc_lto_app
```

`--no-device-link` 明确复用已链接设备镜像；不必再传 `-dlto`。`-cudart=shared` 使用共享 CUDA Runtime。其结构与普通 RDC 主机链接一致，区别是使用 LTO 的三份输入和独立输出名。

## 独立脚本

```bash
CUDA_ARCH=sm_89 bash run.sh
```

三个输入都必须存在，缺失返回 1，不自动编译。日志在 `log/host_link.sh`、`log/host_link.log`、`log/status.tsv`；每条命令 `timeout 60s` 限时，超时为 124。只覆盖本动作日志与 `rdc_lto_app`，失败删除部分程序。

## 当前日志的实际观察

Toolkit 13.4.92：`host_link` 退出 **0**，[host_link.log](log/host_link.log) 为空；`../../build/sm_89/compile/artifacts/rdc_lto_app` 为 **28,680 字节**，恰与普通 `rdc_app` 大小相同。额外的 `readelf -d` 静态核对显示依赖 `libcudart.so.13`；不把这个检查归入链接日志。

相同总大小不代表设备指令相同：ELF 节布局、对齐及主机部分都会影响总大小。没有执行任一程序，也没有测量性能。

## 边界与练习

后续独立静态检查读取同目录下的 `rdc_app` 与 `rdc_lto_app`，用 `cuobjdump --dump-sass` 分别查看 `rdc_kernel`、`rdc_scale` 及调用指令。先预测，再用指令核对；当前编译日志不提供这个答案。

如果未看到调用，是否就能声称更快？不能：还需正确性、计时范围、预热与重复测量。本轮只完成离线构建，回到 [编译路线](../compile/README.md) 汇总证据。

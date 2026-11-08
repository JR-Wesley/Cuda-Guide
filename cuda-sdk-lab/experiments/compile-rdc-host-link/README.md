# 9. 只做 RDC 主机链接

## 背景与目标

设备代码已经链接好，CPU 程序仍需组合 `main`、CUDA 启动和注册代码以及运行库。主机链接同时需要两个原始目标文件与设备链接结果；不能只拿 `rdc_dlink.o` 代替全部输入。

## 底层命令与解释

工作目录：`cuda-sdk-lab/experiments/compile-rdc-host-link/`；依次完成 [RDC 编译](../compile-rdc-compile/README.md)、[设备链接](../compile-rdc-device-link/README.md)。

```bash
nvcc -arch=sm_89 -cudart=shared --no-device-link \
  ../../build/sm_89/compile/artifacts/rdc_device.o \
  ../../build/sm_89/compile/artifacts/rdc_main.o \
  ../../build/sm_89/compile/artifacts/rdc_dlink.o \
  -o ../../build/sm_89/compile/artifacts/rdc_app
```

`--no-device-link` 明确复用前一步的结果，不偷偷再做一次设备链接。`-cudart=shared` 动态链接 CUDA Runtime，控制程序体积；`-arch=sm_89` 明确本组配置。

## 独立脚本

```bash
CUDA_ARCH=sm_89 bash run.sh
```

缺任一输入即返回 1，不编译、不运行。`log/host_link.sh`、`log/host_link.log`、`log/status.tsv` 分别保存命令、诊断、状态。命令限时 `timeout 60s`，超时为 124。只覆盖本动作日志与 `rdc_app`，失败删除部分程序，保留输入。

## 当前日志的实际观察

Toolkit 13.4.92：`host_link` 退出 **0**，[host_link.log](log/host_link.log) 为空；`../../build/sm_89/compile/artifacts/rdc_app` 为 **28,680 字节**。本次额外的 `readelf -d` 静态检查确认依赖 `libcudart.so.13`，该检查未混入编译脚本。

没有执行 `rdc_app`，因此源码中返回值是否确实对应 `6.0f` 尚未做 GPU 验证。

## 边界与练习

画出三个输入各自提供什么：CPU 的 `main`、设备函数定义、已链接的设备镜像。为什么只重新编译 `rdc_device.cu` 后，旧的 `rdc_dlink.o` 不能继续用于可信实验？

后续静态检查读取 `rdc_app`；运行时还需共享库可见。下一步：[LTO 编译对照](../compile-lto-compile/README.md)。

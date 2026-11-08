# 11. 只做 LTO 设备链接

## 背景与目标

前一步已经把高层中间表示保存在两份 `.o` 中。这一步让设备链接器同时看到它们，执行跨文件优化并生成目标机器代码。对于 `rdc_scale(3.0f)`，可以提出“调用可能被内联并折叠”的假设，但必须另读指令才能验证。

## 底层命令与解释

工作目录：`cuda-sdk-lab/experiments/compile-lto-device-link/`；先完成 [LTO 编译](../compile-lto-compile/README.md)。

```bash
nvcc -std=c++17 -O3 -lineinfo -arch=sm_89 -dlto --device-link \
  ../../build/sm_89/compile/artifacts/rdc_device_lto.o \
  ../../build/sm_89/compile/artifacts/rdc_main_lto.o \
  -o ../../build/sm_89/compile/artifacts/rdc_lto_dlink.o
```

`-dlto` 在设备链接阶段启用 LTO；`-arch=sm_89` 指定最终机器目标；`--device-link` 表示停在设备链接结果，不生成 CPU 可执行程序。编译阶段的 `code=lto_89` 与这里的 `-dlto` 配套，不能只给普通 RDC 对象随意加链接选项就声称完成同一对照。

## 独立脚本

```bash
CUDA_ARCH=sm_89 bash run.sh
```

缺输入时返回 1，不隐式构建。日志在 `log/device_link.sh`、`log/device_link.log`、`log/status.tsv`。命令限时 `timeout 60s`，超时为 124。只覆盖本动作日志与 `rdc_lto_dlink.o`，失败删除部分输出；不删除普通 RDC 链接结果。

## 当前日志的实际观察

Toolkit 13.4.92：`device_link` 退出 **0**，[device_link.log](log/device_link.log) 为空；`../../build/sm_89/compile/artifacts/rdc_lto_dlink.o` 为 **9,280 字节**。普通 RDC 对应文件为 **11,832 字节**。

观察仅说明两条构建路径产生了不同大小的目标文件；这还不能证明哪个函数被内联，更不能证明运行速度改变。编译日志没有 SASS，对照检查留在独立分析动作。

## 边界与练习

固定源码、目标架构和优化等级，只比较是否使用 LTO。先写下预期：`rdc_kernel` 里是否还需要调用 `rdc_scale`？在完成 [LTO 主机链接](../compile-lto-host-link/README.md) 后，分别读取两份程序的 SASS 才能检验预测。

若修改任何 LTO 编译输入，必须重做本阶段和主机链接；文件存在不代表新鲜。

# 7. 只编译 RDC 目标文件

## 背景与目标

[rdc_main.cu](../compile/rdc_main.cu) 中 `rdc_kernel` 调用 `rdc_scale(3.0f)`，定义却在 [rdc_device.cu](../compile/rdc_device.cu)。分别编译时还不知道另一文件的完整设备代码，需要把待解决的引用留给设备链接器。

RDC（Relocatable Device Code，可重定位设备代码）允许这种跨源文件设备调用。先手算 `2*3=6`；这是源码预期，非 GPU 实测。

## 底层命令与解释

工作目录：`cuda-sdk-lab/experiments/compile-rdc-compile/`。

```bash
mkdir -p ../../build/sm_89/compile/artifacts
nvcc -std=c++17 -O3 -lineinfo -gencode arch=compute_89,code=sm_89 \
  -dc ../compile/rdc_device.cu -o ../../build/sm_89/compile/artifacts/rdc_device.o
nvcc -std=c++17 -O3 -lineinfo -gencode arch=compute_89,code=sm_89 \
  -dc ../compile/rdc_main.cu -o ../../build/sm_89/compile/artifacts/rdc_main.o
```

`-dc` 等价于启用 RDC 后只编译目标文件（`-rdc=true -c`）。两条命令属于同一个“分别编译”阶段，不包含设备或主机链接。`-gencode` 只请求 `sm_89`，与后面的 LTO 最终机器目标一致；语言、优化和行信息由 `-std=c++17 -O3 -lineinfo` 固定。

## 独立脚本

```bash
CUDA_ARCH=sm_89 bash run.sh
```

每条命令限时 `timeout 60s`；超时返回 124。命令与诊断分别是 `log/compile_device.{sh,log}`、`log/compile_main.{sh,log}`，状态在 `log/status.tsv`。只覆盖自己的两份输出与日志；任一步失败，两份本动作目标文件都删除，避免混用新旧输入。其他动作输出不删除。

## 当前日志的实际观察

Toolkit 13.4.92：`compile_device=0`、`compile_main=0`，两份诊断均为空。产物位于 `../../build/sm_89/compile/artifacts/`：

| 产物 | 字节 |
| --- | ---: |
| `rdc_device.o` | 4,872 |
| `rdc_main.o` | 9,640 |

[状态](log/status.tsv) 表示分别编译完成，不代表 `rdc_scale` 的跨文件调用已经解析。

## 边界与练习

在源码中分别找出声明、调用和定义；解释为什么只有 `rdc_main.o` 还不够。若修改 `rdc_scale` 的实现，必须显式重做编译、设备链接、主机链接，不能只重做最后一步。

下一步：[设备链接](../compile-rdc-device-link/README.md)。

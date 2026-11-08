# 10. 只编译 LTO 中间表示目标文件

## 背景与目标

普通 RDC 在分别编译时看不到另一个文件的函数实现。LTO（Link-Time Optimization，链接时优化）保留更高层的中间表示 IR，让设备链接时有机会跨文件优化。我们复用相同两份源码，不复制它们，不把 LTO 误解为一定更快。

## 底层命令与解释

工作目录：`cuda-sdk-lab/experiments/compile-lto-compile/`。

```bash
mkdir -p ../../build/sm_89/compile/artifacts
nvcc -std=c++17 -O3 -lineinfo -gencode arch=compute_89,code=lto_89 \
  -dc ../compile/rdc_device.cu -o ../../build/sm_89/compile/artifacts/rdc_device_lto.o
nvcc -std=c++17 -O3 -lineinfo -gencode arch=compute_89,code=lto_89 \
  -dc ../compile/rdc_main.cu -o ../../build/sm_89/compile/artifacts/rdc_main_lto.o
```

与 [普通 RDC 编译](../compile-rdc-compile/README.md) 相比，关键变化是 `code=sm_89` 改成 `code=lto_89`：保存链接时优化所需的 IR，而不是此时输出最终设备机器码。`-dc` 仍表示分别编译，不链接；语言、优化和行信息保持相同。后续设备链接才使用 `-dlto`。

## 独立脚本

```bash
CUDA_ARCH=sm_89 bash run.sh
```

每条命令 `timeout 60s` 限时，超时为 124。日志是 `log/compile_device.{sh,log}`、`log/compile_main.{sh,log}` 与 `log/status.tsv`。只覆盖自己的两份输出；失败时两份均删除，不触碰普通 RDC 产物。没有独立 `.ltoir` 导出动作，因为对照只需要 `.o` 中的 IR。

## 当前日志的实际观察

Toolkit **13.4.92** 下先遇到一次真实失败：在编译阶段同时使用 `-dlto` 和 `-gencode ...code=sm_89`，`nvcc` 返回 **1**，提示：

```text
'-dlto' conflicts with '-gencode' or '-code' to control what is generated; use 'code=lto_<arch>' instead of '-dlto' to request lto intermediate
```

原始失败命令、输出、状态被明确命名为 [rejected_flags.sh](log/rejected_flags.sh)、[rejected_flags.log](log/rejected_flags.log)、[rejected_flags-status.tsv](log/rejected_flags-status.tsv)。这是本次修正的反例证据，不是脚本自动历史归档；重跑正常动作不覆盖它，也不要把它当推荐命令重放。

改成 `code=lto_89` 后，当前 [status.tsv](log/status.tsv) 为 `compile_device=0`、`compile_main=0`，两份成功诊断均为空。产物在 `../../build/sm_89/compile/artifacts/`：

| 产物 | 字节 |
| --- | ---: |
| `rdc_device_lto.o` | 5,400 |
| `rdc_main_lto.o` | 10,096 |

## 边界与练习

为什么普通 RDC 与 LTO 的 `.o` 大小不能直接表示最终指令数？解释“保留优化输入”与“执行优化”是两个阶段。下一步：[LTO 设备链接](../compile-lto-device-link/README.md)。本动作尚未证明内联、正确性或加速。

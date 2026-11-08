# 6. 只做普通程序的主机链接

## 背景与目标

[目标文件动作](../compile-object/README.md) 已生成 `probe.o`。链接器现在把其中对库函数的引用连接到运行库，并产生 CPU 可启动的程序。本动作不重新编译源码，也不执行程序。

## 底层命令与解释

工作目录：`cuda-sdk-lab/experiments/compile-host-link/`；先完成 `compile-object`。

```bash
nvcc -arch=sm_89 -cudart=shared --no-device-link \
  ../../build/sm_89/compile/artifacts/probe.o \
  -o ../../build/sm_89/compile/artifacts/probe
```

`--no-device-link` 表示不要在此隐式执行设备链接；普通单文件目标已携带完整设备代码。`-arch=sm_89` 明确目标配置。`-cudart=shared` 动态链接 CUDA Runtime，避免把静态运行库复制到每个教学程序。

## 独立脚本

```bash
CUDA_ARCH=sm_89 bash run.sh
```

缺 `probe.o` 时 `require_file` 报错并返回 1，不隐式编译。实际命令、诊断、状态分别在 `log/host_link.sh`、`log/host_link.log`、`log/status.tsv`。`timeout 60s` 超时返回 124。只覆盖本动作日志与 `probe`；失败删除部分程序，不动输入 `.o`。

## 当前日志的实际观察

Toolkit 13.4.92：`host_link` 退出 **0**，[host_link.log](log/host_link.log) 为空；[probe](../../build/sm_89/compile/artifacts/probe) 为 **22,528 字节**。

本次另外用 `readelf -d` 静态核对动态依赖，看到 `libcudart.so.13`；该检查未加入编译脚本或冒充 `host_link.log` 内容。本轮检查到 `LD_LIBRARY_PATH` 包含 `/opt/cuda/lib64`；构建成功仍不保证另一个 shell 的动态库搜索路径正确。

## 边界与练习

说明为什么动态链接让文件较小，却增加了运行时对共享库可见性的要求。若链接失败，先检查 `host_link.log`；若以后启动时报共享库缺失，检查库路径，而不是立即重编 kernel。

本次没有启动程序，因此没有 `correct=yes` 实测。下一步可学习 [RDC 编译](../compile-rdc-compile/README.md)，它解决跨文件设备调用，不是动态库加载问题。

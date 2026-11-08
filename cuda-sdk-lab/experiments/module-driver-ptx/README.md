# 只运行 Driver-PTX：保留 JIT 失败证据

## 背景

与 [Driver-cubin](../module-driver-cubin/README.md) 使用同一个 `driver`，本动作只把镜像换成 PTX。即时编译（JIT）指驱动在加载时将虚拟指令转为本机 GPU 机器码。架构号匹配和 PTX 语言版本兼容是两个条件；旧驱动可能读不懂新 Toolkit 生成的 PTX。

## 1. 前置构建

分别在 [module-driver-build](../module-driver-build/README.md) 和 [module-ptx-build](../module-ptx-build/README.md) 执行 `CUDA_ARCH=sm_89 bash run.sh`。共享输入为 `build/sm_89/module-loading/{driver,image.ptx}`，本动作不编译、不依赖 cubin。

## 2. 先看实际运行命令

工作目录：`cuda-sdk-lab/experiments/module-driver-ptx/`。

```bash
LD_LIBRARY_PATH=/opt/cuda/lib64 ../../build/sm_89/module-loading/driver \
  ../../build/sm_89/module-loading/image.ptx
```

镜像路径是唯一程序参数；`LD_LIBRARY_PATH` 设置动态库搜索目录，不应指向 stub。记录入口：

```bash
CUDA_ARCH=sm_89 LD_LIBRARY_PATH=/opt/cuda/lib64 bash run.sh
```

脚本不切目录。加载成功时预期校验 64 个元素并输出 `api=Driver correct=yes n=64`；这只是成功路径的预期，**不是本机此次结果**。

## 3. 本次真实失败

2026-11-08，RTX 4060 Laptop、CC 8.9、驱动查询值 596.21（CUDA UMD 13.2）、nvcc 13.4.92。构建成功，但 [原始运行输出](log/driver_ptx.log) 为：

```text
JIT error: ptxas application ptx input, line 9; fatal   : Unsupported .version 9.4; current version is '9.2'
cuModuleLoadDataEx: CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222): the provided PTX was compiled with an unsupported toolchain.
```

[状态表](log/status.tsv) 保留 `driver_ptx 1`；222 是 Driver API 错误值，1 是程序和脚本退出码，不能混淆。尚未到 kernel 启动或结果校验，不能声称 PTX 路径运行通过。不改驱动、不改 PTX 版本头、不吞错，也不拿 cubin 的通过记录替代。

[实际命令](log/driver_ptx.sh) 与输出保存在本动作 `log/`，重跑覆盖同名记录。驱动 JIT 缓存沿用默认或调用者设置，不做缓存命中或性能结论。

## 4. 练习

在 `driver.cpp` 找到 `check(load, "cuModuleLoadDataEx")`，解释为何错误后不会执行 `cuLaunchKernel`。再对比 cubin 日志：两次只换输入镜像，失败阶段不同；不要把“同一加一算法”误解成“同一加载条件”。

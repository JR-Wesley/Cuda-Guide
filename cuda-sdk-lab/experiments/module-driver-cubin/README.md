# 只运行 Driver-cubin

## 背景与代码路径

本动作验证 Driver API 显式加载原生 GPU 镜像。共享的 `../module-loading/driver.cpp` 依次执行 `cuModuleLoadDataEx`（加载）、`cuModuleGetFunction`（按 `add_one` 名称查找）、`cuLaunchKernel`（启动）、同步和逐元素核对。输入为 0–63，输出应为 1–64，详见 [源码讲解](../module-loading/README.md)。

## 1. 先构建两个独立输入

先分别在 [module-driver-build](../module-driver-build/README.md) 和 [module-cubin-build](../module-cubin-build/README.md) 执行 `CUDA_ARCH=sm_89 bash run.sh`。本动作只读取已有 `driver` 与 `image.cubin`，不依赖 Runtime 或 PTX，也不隐式编译。

## 2. 底层命令和记录入口

工作目录：`cuda-sdk-lab/experiments/module-driver-cubin/`。

```bash
LD_LIBRARY_PATH=/opt/cuda/lib64 ../../build/sm_89/module-loading/driver \
  ../../build/sm_89/module-loading/image.cubin
```

唯一参数就是镜像文件。加载器应找到系统真实 Driver 库；不要把 `/opt/cuda/lib64/stubs` 当作运行库目录。独立记录入口：

```bash
CUDA_ARCH=sm_89 LD_LIBRARY_PATH=/opt/cuda/lib64 bash run.sh
```

脚本不切目录；缺输入返回非零并提示显式构建，API 失败原样保留。`CUDA_ARCH` 必须与构建目录一致。

## 3. 真实证据与理解检查

2026-11-08，在 RTX 4060 Laptop、CC 8.9、驱动 596.21、nvcc 13.4.92 产物上，退出 0，输出 `api=Driver correct=yes n=64`，JIT info/error 缓冲为空。见 [原始输出](log/driver_cubin.log)、[命令](log/driver_cubin.sh)、[状态](log/status.tsv)。空日志缓冲不证明内部加载耗时为零。

本动作记录就地写入 `log/`，重跑覆盖同名文件，无自动归档。保持 `driver` 不变，下一步只换镜像，到 [Driver-PTX](../module-driver-ptx/README.md) 运行。练习：预测如果加载已经失败，还会不会执行核函数？由输出中的 API 名称核对，不把 cubin 成功推论成 PTX 一定成功。

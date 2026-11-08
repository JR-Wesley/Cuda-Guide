# Runtime 加载与运行：先看 64 个整数

## 背景与目标

CPU 上的程序需要把设备代码交给 GPU 才能启动 kernel（GPU 函数）。本专题比较两条路径：Runtime API 替程序管理内嵌镜像；Driver API 由程序明确加载镜像。先分清编译、加载、执行，不比较性能。前置只需理解数组、函数调用和“指针保存地址”；设备指针指向 GPU 内存，不能当作 CPU 数组直接读。

本目录保留唯一一套源码：`kernel.cuh`、`runtime.cu`、`driver.cpp`、`image.cu`。**本目录的 `run.sh` 现在只运行 Runtime**，不再连续运行 Driver。旧多构建脚本已被单 Runtime 构建替代，旧多运行逻辑不再保留。

## 1. 手算和读代码

`input[i]=i`，一个 block（线程组）包含 64 个线程。`kernel.cuh` 中每个线程用 `threadIdx.x` 作为下标，写 `output[i]=input[i]+1`：

| 线程/下标 | 输入 | 预期输出 |
| --- | --- | --- |
| 0 | 0 | 1 |
| 1 | 1 | 2 |
| 63 | 63 | 64 |

这个 kernel 没有长度参数或边界判断，要求分配的数组覆盖所有启动线程；不要只扩大线程数。沿 `runtime.cu` 的 `main` 阅读：分配设备内存 → 拷贝输入 → `cudaLaunchKernel` 启动 → 同步等候执行完成 → 拷回并逐元素校验 → 释放。`args` 保存的是两个指针变量的地址，不是数组内容。

## 2. 单独构建，再运行

先在 [module-build](../module-build/README.md) 按底层 `nvcc` 命令构建。再手动进入本目录 `cuda-sdk-lab/experiments/module-loading/`，底层运行命令是：

```bash
LD_LIBRARY_PATH=/opt/cuda/lib64 ../../build/sm_89/module-loading/runtime
```

`LD_LIBRARY_PATH` 告诉 Linux 动态加载器在哪里找共享 Runtime 库；不要把 Driver stub 目录加进去。`sm_89` 是本次 GPU 的机器代码架构，不是通用默认值。记录同一运行动作的入口：

```bash
CUDA_ARCH=sm_89 LD_LIBRARY_PATH=/opt/cuda/lib64 bash run.sh
```

预期 `api=Runtime correct=yes n=64`，失败先查 CUDA API 名称和错误，不把编译通过当作运行通过。脚本不切目录、不编译，不要求 Driver 或独立镜像存在。

## 3. 单变量对照路线

| 动作目录 | 单一功能 | 共享输出或输入 |
| --- | --- | --- |
| [module-build](../module-build/README.md) | Runtime 可执行文件构建 | `runtime`，含原生设备代码 |
| [module-driver-build](../module-driver-build/README.md) | Driver CPU 主程序构建 | `driver`，不内嵌 kernel |
| [module-cubin-build](../module-cubin-build/README.md) | 原生设备镜像构建 | `image.cubin` |
| [module-ptx-build](../module-ptx-build/README.md) | PTX 虚拟指令构建 | `image.ptx` |
| 本目录 | Runtime 运行 | 只读 `runtime` |
| [module-driver-cubin](../module-driver-cubin/README.md) | Driver 加载 cubin 并校验 | 只读 `driver` 和 `image.cubin` |
| [module-driver-ptx](../module-driver-ptx/README.md) | Driver 加载 PTX 并校验 | 只读 `driver` 和 `image.ptx` |

所有产物仍在 `../../build/<arch>/module-loading/`。Driver 两次运行使用同一 `driver`，仅替换镜像文件；PTX 需要驱动即时编译（JIT），cubin 是目标架构机器码。Runtime 镜像和独立 cubin 不是同一个文件，不能宣称两条 API 路径使用完全相同的机器码。

## 4. 本次真实证据与边界

2026-11-08 实测环境：RTX 4060 Laptop GPU，compute capability 8.9，驱动查询值 596.21，nvcc 13.4.92；`nvidia-smi` 同时显示 CUDA UMD 13.2。运行使用 `LD_LIBRARY_PATH=/opt/cuda/lib64`。

- 四个独立构建动作均退出 0，各自有 `log/build.sh`、`build.log` 和 `status.tsv`。
- 本动作 [原始输出](log/runtime.log) 为 `api=Runtime correct=yes n=64`，[状态](log/status.tsv) 为 `runtime 0`。
- Driver-cubin 校验通过；Driver-PTX **失败**，详见其动作 README 与原始日志。cubin 可运行不能证明 PTX JIT 兼容。

本实验不测加载时间、缓存命中或性能，也不能由 `correct=yes` 推断内部加载时机。驱动 JIT 缓存沿用默认或调用者设置，不修改系统环境。

每次运行 `source ../record.sh` 只初始化本动作 `log/status.tsv`，`record` 保存实际命令为 `<label>.sh`、原始合并输出为 `<label>.log` 和真实退出码。空构建日志表示编译器无输出，是否成功看状态表。重跑覆盖同名记录；未列入当次状态表的旧文件不能视为本次执行。可在动作目录（不是 `log/` 内）用 `bash log/runtime.sh` 重放命令，但须自行提供同样的动态库环境。缺文件提示先显式构建；源码或选项变化后手动重编译，无自动缓存或归档。

## 5. 解释与练习

先预测最后一个元素，再看逐元素校验。然后保持 `driver` 不变，仅从 cubin 换成 PTX：若失败发生在 `cuModuleLoadDataEx`，说明尚未启动 kernel，而不是加一算法算错。练习说明为什么 Runtime 成功和 PTX 加载失败可以同时成立。

参考：[Driver module management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html)、[Runtime execution control](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html)。

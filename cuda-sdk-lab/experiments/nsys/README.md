# nsys 时间线采集：先看操作先后关系

## 背景

Nsight Systems 用时间线把 CPU 的 CUDA API 调用、数据拷贝和 GPU kernel 联系起来。它首先回答“什么时候做了什么”，不是收集单 kernel 的全部硬件指标。

先完成 [build](../build/README.md) 和 [execution](../execution/README.md)。本动作只采集共享程序的 vector/257/1，不再遍历两个转置实现，也不自动运行 stats。

## 1. 底层命令

当前目录：`cuda-sdk-lab/experiments/nsys/`。

```bash
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --force-overwrite=true --output=./log/timeline \
  ../../build/sm_89/vector_transpose --case vector --n 257 --repeat 1
```

- `profile` 执行一次应用采集；`--trace=cuda` 只关注 CUDA 路径。
- `--sample=none` 不做 CPU 周期采样，`--cpuctxsw=none` 不采集 CPU 上下文切换，避免无关系统级采集及权限依赖。
- `--output` 指定报告基名；`--force-overwrite=true` 允许覆盖本动作同名报告。

记录入口：

```bash
CUDA_ARCH=sm_89 bash run.sh
```

脚本创建本动作 `log/`，检查共享输入，不隐式构建。需要保留旧报告时由使用者另命名，不自动创建历史目录。

## 2. 本次真实结果

Nsight Systems 2026.3.2，RTX 4060 Laptop、驱动 596.21：[命令](log/profile.sh)、[输出](log/profile.log)、[状态](log/status.tsv)。退出 0，程序 `correct=yes, errors=0`，checksum `3508.375000`，生成 [timeline.nsys-rep](log/timeline.nsys-rep)，本次约 82 KiB。

程序的 `batch_us_per_launch=4.512` 微秒属于分析器运行中的 event 区间，不代替普通运行基线。源码有预热和一次计时启动，时间线中预期看到两次 `vector_add`；下一步使用独立 [nsys-stats](../nsys-stats/README.md) 核对实例数。

工具输出包含 `/tmp/nsys-mikasa/` 的内部临时路径；这是工具自身行为，不是本项目创建的归档。项目只保留固定报告，没有大型 trace 或持久化 `.qdstrm` 副本。

## 3. 练习

沿源码标出分配 → H2D 拷贝 → 预热 → 同步 → event → kernel → event → D2H 拷贝。为什么 CPU 上 `cudaLaunchKernel` 的持续时间不等于 GPU kernel 的持续时间？API 提交与设备执行是不同区间。

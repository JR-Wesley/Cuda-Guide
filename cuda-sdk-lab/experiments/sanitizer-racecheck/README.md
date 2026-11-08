# 只检查 shared 写竞争：racecheck

## 1. 具体冲突

共享 [cases.cu](../sanitizer/cases.cu) 的 `shared_write_race` 启动 2 个线程，均向同一个 shared `slot` 写线程编号。两次写没有规定顺序，称为 WAW（write after write，写与写之间）竞争。后面的 `__syncthreads()` 只等待到达屏障，不会追溯地规定之前两次写的顺序。racecheck 主要分析共享内存竞争，不是通用全局内存竞争证明器。

## 2. 底层命令与入口

先在 [sanitizer-build](../sanitizer-build/README.md) 显式构建。工作目录：`cuda-sdk-lab/experiments/sanitizer-racecheck/`。

```bash
LD_LIBRARY_PATH=/opt/cuda/lib64 compute-sanitizer --tool racecheck \
  --error-exitcode 86 ../../build/sm_89/sanitizer/cases racecheck
```

`--tool` 选择检查器，程序末尾参数选择 kernel；`--error-exitcode 86` 保留可识别的工具报错退出值。`LD_LIBRARY_PATH` 用于共享 Runtime。记录同一动作：

```bash
CUDA_ARCH=sm_89 LD_LIBRARY_PATH=/opt/cuda/lib64 bash run.sh
```

可设置 `COMPUTE_SANITIZER`。脚本不切目录、不编译、只读共享 debug 产物。

## 3. 实测证据与判定

2026-11-08，RTX 4060 Laptop/CC 8.9、驱动 596.21、nvcc 13.4.92、Compute Sanitizer 2026.3.0.0。原始检查器退出 **86**，报告 `Race reported between Write access`，对应 `cases.cu:13`；汇总为 **1 hazard displayed (1 error, 0 warnings)**。见 [原始输出](log/racecheck.log)、[命令](log/racecheck.sh)、[状态](log/status.tsv)。

脚本分别确认退出码为 86、出现 `Race reported` 或 `WAW hazard`、`RACECHECK SUMMARY` 中 hazard 和 error 均非零。三项断言实测均为 0；脚本成功表示故障被发现，不是程序正确。工具缺失或只有其他错误不会通过。

`log/` 只属于本动作，重跑覆盖同名记录，原始 86 不改成 0，不自动归档。

## 4. 单变量练习

预测若只允许线程 0 写 `slot`，其他条件不动，报告会怎样。修改后显式重编译、仅运行本动作：预期负例断言失败，因为竞争消失。最后恢复故障并重新采集；不把改前的报告作为改后证据。

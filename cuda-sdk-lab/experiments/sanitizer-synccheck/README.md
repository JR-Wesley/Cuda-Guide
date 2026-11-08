# 只检查 warp 屏障误用：synccheck

## 1. 画出参与线程

共享 [cases.cu](../sanitizer/cases.cu) 的 `divergent_warp_barrier` 启动一个 32 线程 warp。`if (threadIdx.x < 17)` 让 lane 0–16 调用 `__syncwarp(0xffffu)`；掩码 `0xffff` 只有低 16 位为 1，列出的却只有 lane 0–15。lane 16 调用了屏障但自己的位不在掩码中，违反使用要求。这是同步原语参数错误，不是“任何分支都不能同步”。

## 2. 底层命令与入口

先在 [sanitizer-build](../sanitizer-build/README.md) 显式构建。工作目录：`cuda-sdk-lab/experiments/sanitizer-synccheck/`。

```bash
LD_LIBRARY_PATH=/opt/cuda/lib64 compute-sanitizer --tool synccheck \
  --error-exitcode 86 ../../build/sm_89/sanitizer/cases synccheck
```

`--tool` 选检查器，最后参数选程序故障分支；`--error-exitcode 86` 设置错误退出值；`LD_LIBRARY_PATH` 用于动态 Runtime 库。记录入口：

```bash
CUDA_ARCH=sm_89 LD_LIBRARY_PATH=/opt/cuda/lib64 bash run.sh
```

可指定 `COMPUTE_SANITIZER`。脚本不切目录、不编译，仅运行此项。

## 3. 本次真实证据

2026-11-08，RTX 4060 Laptop/CC 8.9、驱动 596.21、nvcc 13.4.92、Compute Sanitizer 2026.3.0.0。检查器原始退出 **86**，报告 `Barrier error detected. Invalid arguments.`，设备调用栈含 `divergent_warp_barrier` 的 `cases.cu:23`；汇总 **17 errors**。见 [原始日志](log/synccheck.log)、[命令](log/synccheck.sh)、[状态](log/status.tsv)。工具对多个参与线程给出报告，不能把 17 解读成 17 处独立源码错误。

退出码断言、特定诊断匹配、非零错误汇总实测均为 0，脚本才返回成功；日志中原始 86 不改变。任意非零返回并不足以通过。就地 `log/` 重跑覆盖同名记录、不归档。

## 4. 练习与边界

先预测只把条件 `< 17` 改成 `< 16`，调用线程集合是否与掩码一致；再显式重编译、只运行此检查。预期负例脚本因没有该诊断而失败。练习后恢复故障、重新采集并更新说明。即便 synccheck 没有报告，也不能证明整个程序的内存顺序和线程协作都正确。

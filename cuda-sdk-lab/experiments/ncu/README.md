# ncu 基本采集：只观察一次 kernel

## 背景

Nsight Compute 用硬件计数器等信息分析选定 kernel。先知道启动了多少块、用了多少线程，再读吞吐；不要一上来对整个程序收集所有指标。本动作只采集 `LaunchStats`（启动配置）和 `SpeedOfLight`（吞吐概览），内存表格在独立的 [ncu-memory](../ncu-memory/README.md)。

前置：显式执行 [build](../build/README.md)，再完成 [execution](../execution/README.md) 的结果核对。当前输入固定 vector、257 个元素、`repeat=1`，重点是工具用法而非峰值性能。

## 1. 先理解命令

当前目录：`cuda-sdk-lab/experiments/ncu/`。

```bash
ncu --export ./log/basic --force-overwrite \
  --kernel-name vector_add --launch-skip 1 --launch-count 1 \
  --section LaunchStats --section SpeedOfLight \
  ../../build/sm_89/vector_transpose --case vector --n 257 --repeat 1
```

- `--kernel-name` 只选目标函数；`--launch-skip 1` 跳过匹配函数的一次预热；`--launch-count 1` 只记录一次匹配启动，不等于物理上只执行一遍。
- `--section` 选择一组相关指标；工具可能多次重放 kernel 取得这些指标。
- `--export` 保存可供界面或 CLI 导入的报告；当前 **ncu 2026.3 默认生成 `.ncu-repz`**。不要照搬旧版 `.ncu-rep` 文件名。
- `--force-overwrite` 明确允许覆盖本动作同名报告。需要保留的对照应由使用者另命名，脚本不自动归档。

脚本会先创建日志目录并检查已有输入：

```bash
CUDA_ARCH=sm_89 bash run.sh
```

它只采集，不再自动运行第二个 case 或执行 `--import`。工具可由 `NCU` 指定，架构必须显式设置，不修改性能计数器权限。

## 2. 本次实际记录

环境：RTX 4060 Laptop，CC 8.9，驱动 596.21，nvcc 13.4.92，ncu 2026.3.1.0。[命令](log/profile.sh)、[采集输出](log/profile.log)、[状态](log/status.tsv)、[二进制报告](log/basic.ncu-repz)。

本次采集成功，退出 0，程序 `correct=yes`，校验和 `3508.375000`；工具显示 **8 passes**。这证明本次目标采集可用，不只是 `ncu --version` 可用。

报告的可读指标由独立 [ncu-report](../ncu-report/README.md) 导入解释，不在本脚本里隐式完成。采集下程序打印的 `batch_us_per_launch=514201.172` 微秒明显受重放/插桩扰动，**不是普通运行耗时**；不得与 execution 的 4.096 微秒直接作性能比值。

### 保留本轮遇到的接口差异

首次采集其实成功，但旧后缀检查找不到 `.ncu-rep`，脚本判为失败；现已按实际 `.ncu-repz` 修正并复跑。首次 [命令](log/report_suffix_attempt.sh)、[输出](log/report_suffix_attempt.log)、[状态](log/report_suffix_attempt-status.tsv) 和 [缺文件说明](log/report_suffix_attempt-input.log) 明确命名保留，不混进当前 `status.tsv`，也不是自动历史归档。

## 3. 理解检查

为什么 `--launch-count 1` 日志却显示 8 passes？它限制的是采集对象，不是计数器重放次数。若计数器报权限错误，保留实际错误并停止，不通过 sudo 或修改系统权限绕过。

参考：[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。

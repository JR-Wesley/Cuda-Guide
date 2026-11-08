# 正常程序只做 racecheck

## 背景

结果恰好正确仍可能存在竞争。racecheck 主要检测同一 block 中的 shared 内存冲突；它不能替代 [正常 memcheck](../validation/README.md) 的越界检查，也不能证明任意全局内存同步都正确。

共享 `../../examples/vector_transpose.cu` 的 `transpose_tiled` 先把数据写入 shared tile，再经过 `__syncthreads()` 供其他线程读取。本动作使用相同的小输入：65×65 矩阵、`repeat=1`；保留边界块，方便对照，不新增源码或复制构建产物。

## 1. 前置构建与底层命令

先按 [主构建脚本](../build/run.sh) 在 `experiments/build/` 显式构建 `build/sm_89/vector_transpose`。工作目录：`cuda-sdk-lab/experiments/validation-racecheck/`。

```bash
LD_LIBRARY_PATH=/opt/cuda/lib64 compute-sanitizer --tool racecheck \
  --error-exitcode 86 ../../build/sm_89/vector_transpose \
  --case tiled --n 65 --repeat 1
```

`--tool racecheck` 选共享内存检查；`--error-exitcode 86` 使检出的错误反映为非零返回。`--case tiled` 选分块转置，`--n` 指方阵宽度，`--repeat 1` 只缩短计时循环，不移除程序已有预热和校验。`LD_LIBRARY_PATH` 指共享 Runtime 库目录。

## 2. 单独入口

```bash
CUDA_ARCH=sm_89 LD_LIBRARY_PATH=/opt/cuda/lib64 bash run.sh
```

可设置 `COMPUTE_SANITIZER`。脚本不切目录、只运行 racecheck，不运行 memcheck、不编译；缺文件要求先构建。修改主程序或编译配置后须手动重编译。

## 3. 预期与本次真实观察

预期原始命令退出 0、`correct=yes, errors=0,` 且 `RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)`。脚本同时匹配结果正确和完全干净的汇总，非零错误、警告、其他失败不能冒充通过。若工具版本改变格式，保留日志先调查，不盲目放宽匹配。

**2026-11-08 实测通过**：主代理完成共享二进制构建后，等待 `validation` 的 memcheck 结束，再单独运行本动作；未重新编译。环境为 RTX 4060 Laptop GPU、compute capability 8.9、驱动 596.21、nvcc 13.4.92、Compute Sanitizer 2026.3.0.0；入口使用 `CUDA_ARCH=sm_89 LD_LIBRARY_PATH=/opt/cuda/lib64 bash run.sh`。

[原始输出](log/racecheck.log) 确认 `case=tiled, n=65, repeat=1, elements=4225`，结果为 `correct=yes, errors=0, checksum=25204.875000`，工具汇总为 `RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)`。[状态表](log/status.tsv) 中 `racecheck`、`correctness`、`summary` 均退出 0，动作脚本也退出 0；[实际命令](log/racecheck.sh) 可供核对。

这说明当前二进制在该输入下未被 racecheck 检出 shared 内存冲突，不证明所有输入或任意全局内存同步都正确。这里的真实正常程序记录与故障例子的预期检出记录分开保存，不互相替代。

`log/status.tsv` 保留检查器实际退出码；`racecheck.sh` / `racecheck.log` 保存命令及输出；`correctness`、`summary` 为额外断言记录。日志在本动作就地保存，重跑覆盖同名文件，不归档。检查器下的时间不与普通运行性能比较。

## 4. 理解检查

在源码中找到 shared 写入、屏障和 shared 读取三个阶段。先解释为什么不同线程读取前需要屏障，再看报告是否支持当前输入下没有检测到冲突；不要由一次零报告推广到所有输入和所有内存顺序。

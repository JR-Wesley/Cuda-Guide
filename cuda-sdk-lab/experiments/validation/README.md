# 正常程序只做 memcheck

## 背景与目标

本目录 **只保留正常程序的 memcheck**，原来同时运行 racecheck 的逻辑已删除；共享内存竞争检查在 [validation-racecheck](../validation-racecheck/README.md)。与故意出错的 [Sanitizer 专题](../sanitizer/README.md) 不同，这里要求程序结果正确且工具无错误。

读取共享主示例 `../../examples/vector_transpose.cu` 的 `transpose_tiled`：矩阵宽度 65，不是 32 的整数倍，因此会覆盖不完整边界块。输入为 65×65=4225 个元素，`repeat=1` 使测量循环只重复一次；程序本身仍有预热和结果核对。这不是性能实验，不使用检查器下的耗时作性能结论。

## 1. 先显式构建

先按 [主构建脚本](../build/run.sh) 在 `experiments/build/` 执行 `CUDA_ARCH=sm_89 bash run.sh`，得到 `build/sm_89/vector_transpose`。本动作不编译、不修改主示例；源文件或选项变化后由使用者重新构建。

## 2. 底层命令与入口

工作目录：`cuda-sdk-lab/experiments/validation/`。

```bash
LD_LIBRARY_PATH=/opt/cuda/lib64 compute-sanitizer --tool memcheck \
  --error-exitcode 86 ../../build/sm_89/vector_transpose \
  --case tiled --n 65 --repeat 1
```

`--tool memcheck` 查非法设备内存访问；`--error-exitcode 86` 防止工具发现错误但命令仍返回零。程序的 `--case tiled` 选分块转置，`--n` 是方阵宽度而非总元素数，`--repeat` 是计时循环重复数。记录入口：

```bash
CUDA_ARCH=sm_89 LD_LIBRARY_PATH=/opt/cuda/lib64 bash run.sh
```

可用 `COMPUTE_SANITIZER` 指定工具。脚本不切目录、缺文件就失败，不会调用构建或其他检查器。

## 3. 如何读结果：本次真实验证通过

预期原始命令退出 0、输出 `correct=yes, errors=0,`，工具汇总 `ERROR SUMMARY: 0 errors`；脚本分别记录正确性和零错误汇总的匹配结果，不能只凭退出 0 判断通过。

**2026-11-08 实测通过**：主代理完成共享二进制构建后，本动作先于 `validation-racecheck` 单独运行，未重新编译。环境为 RTX 4060 Laptop GPU、compute capability 8.9、驱动 596.21、nvcc 13.4.92、Compute Sanitizer 2026.3.0.0；入口使用 `CUDA_ARCH=sm_89 LD_LIBRARY_PATH=/opt/cuda/lib64 bash run.sh`。

[原始输出](log/memcheck.log) 确认 `case=tiled, n=65, repeat=1, elements=4225`，结果为 `correct=yes, errors=0, checksum=25204.875000`，工具汇总为 `ERROR SUMMARY: 0 errors`。[状态表](log/status.tsv) 中 `memcheck`、`correctness`、`summary` 均退出 0，动作脚本也退出 0；[实际命令](log/memcheck.sh) 可供核对。

这是当前二进制和该边界输入下未检测到非法访问的证据，不证明所有输入都安全，也不证明没有 shared 内存竞争。日志中的耗时保留原样，但受检查器插桩影响，不用于性能比较。

本动作 `log/status.tsv` 保存真实退出码，`memcheck.sh` / `memcheck.log` 保存命令和原始输出，另有 `correctness`、`summary` 的断言记录。重跑只覆盖当前动作同名文件，不归档；失败或未执行不能标成通过。

## 4. 练习

先画出宽度 65 时最后一个 32×32 边界块的有效范围，再对照源码中的 `x < width`、`y + j < width`。解释为什么这里无越界不能证明 shared 内存访问无竞争，然后单独去 racecheck 动作观察；两个检查器不能互相代替。

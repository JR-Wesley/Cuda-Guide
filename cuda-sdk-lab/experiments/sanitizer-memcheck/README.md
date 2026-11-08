# 只检查故意越界写：memcheck

## 1. 先手算地址

共享 [cases.cu](../sanitizer/cases.cu) 的 `bad_global_write` 只有 1 个线程。主机只分配 1 个 `int`（4 字节），合法元素是 `p[0]`，却写 `p[1]`；后者从分配区结束后的第一个字节开始。memcheck 用来检查这种非法设备内存访问，不是性能计时器。

## 2. 编译与运行分开

先在 [sanitizer-build](../sanitizer-build/README.md) 执行 `CUDA_ARCH=sm_89 bash run.sh`。再手动进入本目录 `cuda-sdk-lab/experiments/sanitizer-memcheck/`。底层命令：

```bash
LD_LIBRARY_PATH=/opt/cuda/lib64 compute-sanitizer --tool memcheck \
  --error-exitcode 86 ../../build/sm_89/sanitizer/cases memcheck
```

`--tool memcheck` 选择检查器；最后的 `memcheck` 是传给 `cases` 的故障选择参数，两者作用不同。`--error-exitcode 86` 指定工具检出错误的非零退出码。共享 Runtime 用 `LD_LIBRARY_PATH` 找库。

对应记录入口：

```bash
CUDA_ARCH=sm_89 LD_LIBRARY_PATH=/opt/cuda/lib64 bash run.sh
```

可用 `COMPUTE_SANITIZER` 指定工具。脚本只运行这个检查器，不切目录、不编译，缺输入提示先构建。

## 3. 本次真实结果

2026-11-08，RTX 4060 Laptop/CC 8.9、驱动 596.21、nvcc 13.4.92、Compute Sanitizer 2026.3.0.0。见 [原始日志](log/memcheck.log)、[命令](log/memcheck.sh)、[状态](log/status.tsv)：原始退出码 **86**，报告 `Invalid __global__ write of size 4 bytes`，对应 `cases.cu:8`，汇总 **2 errors**。第二项是 `cudaDeviceSynchronize` 返回 error 999；保留了 `CUDA execution: unknown error` 与应用失败信息，没有隐藏后续 API 错误。

`expected_exit`、`diagnostic`、`summary` 三个断言均为 0，动作脚本退出 0 仅表示预期故障检出。不能仅凭非零或 “unknown error” 判定成功。三项任一不匹配，脚本失败。

记录就地放在 `log/`，重跑覆盖同名文件，不归档；只以本次 `status.tsv` 为准。原始检查器退出码不会被断言结果替换。

## 4. 练习

先预测把 `p[1]` 改成 `p[0]` 的结果，再显式重编译并只执行本动作。应不再有该越界诊断，负例脚本应失败；练习后恢复故障源码并重编译、重新采集，再更新实测说明。完整边界见 [专题说明](../sanitizer/README.md)。

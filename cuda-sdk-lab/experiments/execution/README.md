# 只运行向量加法：先有正确性基线

## 背景与小输入

工具分析之前，先知道程序应该算什么。共享 [源码](../../examples/vector_transpose.cu) 的 `vector_add` 每线程计算一个 `a[i]+b[i]`，用 `i<count` 防止越界。`n=257` 时启动两块、每块 256 个线程，第二块仅第一个线程写回。

前置：在 [build](../build/README.md) 显式构建。本脚本不编译，只运行 **vector 一个 case**，不再循环执行两个转置实现。

## 1. 底层运行与参数

当前目录：`cuda-sdk-lab/experiments/execution/`。

```bash
../../build/sm_89/vector_transpose --case vector --n 257 --repeat 1
```

`--case` 选择计算，`--n` 是元素数，`--repeat` 是 event 计时区间内的启动次数。源码另外执行一次预热，因此 `repeat=1` 总共启动两次 kernel。先分配、拷贝、预热和同步，再计时，最后拷回逐元素校验。

记录入口：

```bash
CUDA_ARCH=sm_89 N=257 REPEAT=1 bash run.sh
```

`N=257`、`REPEAT=1` 是当前默认。`NVCC`、`VECTOR_N`、`MATRIX_N` 不是本入口参数。架构用于选择已有共享产物；运行时依赖动态 `cudart`，本机 `LD_LIBRARY_PATH` 已包含 `/opt/cuda/lib64`。缺产物先返回 build，不隐式构建。

## 2. 本次结果与计时边界

[实际命令](log/vector.sh)、[原始输出](log/vector.log)、[状态表](log/status.tsv) 对应 RTX 4060 Laptop、sm_89、驱动 596.21：

```text
correct=yes, errors=0, checksum=3508.375000
batch_us_per_launch=4.096, effective_GBps=0.753
```

这是本次小输入观察，不是稳定性能结论：只有一次计时启动，event 区间可能含主机提交间隙，并且排除分配及数据传输。`effective_GBps` 来自算法字节数估算，不是 DRAM 流量计数器。不要把 ncu 重放中的程序 event 时间拿来比较快慢。

`correctness` 记录是对输出的额外断言；核心逐元素结果校验仍由程序完成。重跑只覆盖本动作同名记录。

## 3. 练习

先手算 `N=4` 的输出 `0, 0.375, 0.75, 1.125`，checksum 应为 `2.25`，再手工运行检查；这组是练习预期，不是上面的实测输入。保持二进制不变改回 257，解释为什么有 255 个线程不写回。

后续分别进入 [正常 memcheck](../validation/README.md)、[ncu 基本采集](../ncu/README.md) 或 [nsys 时间线](../nsys/README.md)，不由本动作串联执行。

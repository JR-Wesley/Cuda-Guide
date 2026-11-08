# nsys stats：只从已有时间线生成汇总

## 背景

采集与统计分开，才能在不再次启动程序的情况下阅读已有证据。本动作不编译、不 profile，只将 [nsys](../nsys/README.md) 的报告转换为可查询数据并打印 kernel/API 汇总。

## 1. 底层命令与记录入口

当前目录：`cuda-sdk-lab/experiments/nsys-stats/`。

```bash
nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum \
  --sqlite ./log/timeline.sqlite ../nsys/log/timeline.nsys-rep
```

`--report` 选择 GPU kernel 汇总与 CPU API 汇总；`--sqlite` 将导出数据库放在**本统计动作**，不是采集目录。SQLite 是查询报告数据的文件格式，不是另一份 CUDA 程序。

```bash
bash run.sh
```

脚本检查输入并创建日志目录，只替换本动作的同名数据库与输出，保证重跑不误读旧 SQLite。缺少原报告时提示先完成 nsys 采集，不隐式重采集。无须设置 CUDA_ARCH，因为读取的报告已确定。

## 2. 本次汇总

[命令](log/stats.sh)、[完整输出](log/stats.log)、[状态](log/status.tsv) 退出 0，数据库 [timeline.sqlite](log/timeline.sqlite) 为 420 KiB。

- `vector_add` 实例数 **2**：一次预热加一次计时启动，与源码一致。
- GPU kernel 合计 **2336 ns**，均值 **1168 ns**，最小/最大 **1024/1312 ns**。
- `cudaLaunchKernel` 有两次调用；`cudaMalloc` 的三次 API 调用合计约 **384.293 ms**，第一次调用可能包含初始化等一次性开销，但仅凭此汇总不能将内部原因定死。

kernel 的均值不等于程序 event 批平均（4.512 us）；后者还可能包括启动间隙。CPU API 时间和 GPU 执行时间可能重叠，不能直接相加推导总耗时。这里只为学会读列名，不拿两次小 kernel 的平均值评价设备性能。

## 3. 理解检查

解释 `Instances` 和 `Num Calls` 的对象为何不同。若重新采集产生新的 `.nsys-rep`，为何不能继续引用旧 `stats.log`？统计必须与当次输入报告配套更新。

# Stage 2：Global Memory Optimization

## 当前独立动作约定

以下命令从项目根目录执行。先单独构建要使用的目标，再选择运行、校验或分析；后续动作缺产物只报错，不自动编译。源码或编译配置变化后必须重新执行构建。

`CUDA_ARCH=sm_89 python3 commands/build/run.py global_copy`

日志固定为 commands/<action>/log/，重复同动作覆盖该目录；先阅读或手动保留需要对照的结果再重跑。历史 results/ 不写入。本页历史分析链接不代表新入口已通过 GPU 复验。


## Learning Objectives

主教材：[Best Practices Guide 13.1 §10.2.1](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)。补充：[Programming Guide 的 Size and Alignment Requirement](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-c-programming-guide/index.html#size-and-alignment-requirement)。目标是从 lane 地址推导 sector 覆盖，区分 coalescing、向量宽度、缓存复用和实际 DRAM 流量。

完成后应能解释连续、strided、misaligned Copy 的地址图；说明为什么 float4 并不自动减少每字节的 sector，也不能仅凭耗时测出 transaction 或命中率。

## 先做一个最小对照

在项目根目录先运行 `CUDA_ARCH=sm_89 python3 commands/verify/run.py global_copy` 校验当前程序，再分别运行 `CUDA_ARCH=sm_89 python3 commands/run/run.py global_copy --variant scalar --direction gather --n 33 --stride 1 --offset 0 --warmup 1 --repeats 2 --iterations 3` 和同一命令仅将 `--stride 1` 改成 `--stride 2`。目的不是立刻排性能名次：先画第一个 warp 的 33 个读取下标，预测两个输入地址跨度，核对两次 `samples.csv` 的 `validation`、`allocation_bytes` 和 `requested_bytes`。**预期**校验均通过、stride=2 需要更大的地址跨度；实际时间不保证按比例变化。若失败先读本次 `commands/<action>/log/` 的 `status.txt`、`commands.jsonl` 和对应日志，再检查架构与工作集限制。掌握地址到逻辑元素的映射后再读下表的向量化和 sector 模型。

## Theory：同一个逻辑 Copy，改变地址布局

所有配置复制同一 seed=20261023 生成的 N 个有限 FP32 数，CPU 参考定义不变。stride/offset 的单位都是 **float 元素**。

| direction | 输入逻辑 i 的地址 | 输出逻辑 i 的地址 | 目的 |
| --- | --- | --- | --- |
| gather（默认） | `in[offset+i*stride]` | `out[i]` | 只改变读取布局 |
| scatter | `in[i]` | `out[offset+i*stride]` | 只改变写入布局 |
| both | `in[offset+i*stride]` | `out[offset+i*stride]` | 同时改变两侧，单独分析 |

`scalar` 每线程复制一个元素，支持任意合法 stride/offset。`float2`/`float4` 每线程复制一个完整向量，尾部用独立 scalar 路径；要求 stride=1，offset 分别是 2/4 的倍数。cudaMalloc 基址加合法偏移满足向量类型对齐；非法配置直接拒绝，不执行未对齐宽访问。

向量宽度是处理因素：N 固定，线程数随每线程工作量调整。它不是相同线程数/寄存器使用下仅换一条指令的对照；grid、资源和尾部均记录。连续 scalar 已经合并良好，向量化可能减少指令/调度，也可能增加寄存器和尾部开销。

## Questions

1. 32 个 lane 请求哪些字节，哪些请求覆盖同一个 sector？
2. stride=8 与 32 的首个 warp sector 数相同，性能为何仍可能不同？
3. offset=1 的额外边缘 sector 是否必须从 DRAM 重读？
4. float4 每 warp 16 sectors、scalar 只有 4，为什么不能说 float4 更差？
5. 固定 N 扫 stride 还改变了什么，怎样排除这些影响？
6. 向量化没有变快时应否保留结果？哪些边界让结论失效？

## Performance Model

每次逻辑读取 4N B、写入 4N B，requested bytes=`8N`。CSV 的 `working_set_bytes` 指**访问过的唯一有效字节**，不含洞；`allocation_bytes` 是实际分配量；`input/output_address_span_bytes` 是第一个逻辑元素到最后一个末尾的跨度，不含 offset 前缀。实际 cache sector footprint 不等于这些量。

一条 warp 访存指令的模型是 `S=|{floor(byte_address/32)}|`，包含每个 lane 的整个访问宽度。32 B sector 是地址覆盖单位，不是跨所有缓存层通用的 DRAM transaction。读写分别计算，部分 warp 只计活跃 lane。

| scalar stride，offset=0 | lane 0/1/2/31 的 byte offset | 读取 sectors | 有效读字节 / sector 字节 |
| ---: | --- | ---: | ---: |
| 1 | 0 / 4 / 8 / 124 | 4 | 100% |
| 2 | 0 / 8 / 16 / 248 | 8 | 50% |
| 4 | 0 / 16 / 32 / 496 | 16 | 25% |
| 8 | 0 / 32 / 64 / 992 | 32 | 12.5% |
| 32 | 0 / 128 / 256 / 3968 | 32 | 12.5% |

gather 写入仍为 4 sectors。stride=1、offset=1 时读取 byte 4..131，覆盖 5 sectors；相邻 warp 可能复用边缘数据，不能据此宣称 DRAM 流量或总耗时增加 25%。

完整 float2/float4 warp 覆盖 8/16 sectors，同时处理 64/128 floats；每个有效字节的理想利用率与 scalar 相同。`model_*_sectors_first_instruction` 只计首个完整向量指令的活跃 lane；独立 scalar 尾部列为 `tail_elements`。N 小于向量宽度时前者为**模型算出的 0**，不是将未测计数器填零。

带宽=`8N/(event_us×1000)` GB/s。只有实际流量 Q 与该层带宽 B 已知，才能用 Q/B 推导下界；本实验不把 sector 模型冒充 Q_DRAM。

## Benchmark Implementation

从项目根目录执行。build/verify/static/sanitizer 支持指定实验，每次保存独立证据。

```bash
CUDA_ARCH=sm_89 python3 commands/build/run.py global_copy
CUDA_ARCH=sm_89 python3 commands/verify/run.py global_copy
# 连续 scalar/float2/float4，等量逻辑 Copy。
CUDA_ARCH=sm_89 python3 commands/run/run.py global_copy --n 4194304 --stride 1 --offset 0
# 只变读取 stride，依次用 1/2/4/8。
CUDA_ARCH=sm_89 python3 commands/run/run.py global_copy --variant scalar --direction gather --n 4194304 --stride 4
# 同一 N、stride=1，只变读取基址 offset。
CUDA_ARCH=sm_89 python3 commands/run/run.py global_copy --variant scalar --n 4194304 --stride 1 --offset 1
# 写入布局另行对照。
CUDA_ARCH=sm_89 python3 commands/run/run.py global_copy --variant scalar --direction scatter --n 1048576 --stride 8
```

通用参数沿用 S1：`--n`、`--block`、`--warmup`、`--repeats`、`--iterations`、`--max-mib`、`--device`、`--csv`。默认 warmup=5、repeats=21、iterations=100、block=256。新增 stride=1（1..1024）、offset=0（0..1024）、direction=gather（gather/scatter/both）、variant=all（scalar/float2/float4）。stride>1 或任意未对齐 offset 使用 scalar；all 不会静默跳过不合法的向量版本。

默认总设备分配上限 256 MiB，且不超过可用显存一半。输入跨度=`4*((N-1)*read_stride+1)`，分配再加 `4*read_offset` 前缀；输出同理。分配前检查溢出和上限。host 参考、初始输出、回读另占内存。

## Correctness Verification

全量逐位比较整个输出分配，包括 offset 前缀和 stride 的洞；这些位置必须保持 sentinel=-777.25。输入洞为不同 sentinel，读错也会失败。每个样本后校验，失败不发布 CSV。

verify 有 27 个 GPU CTest 用例，覆盖 N=1..5、warp/block 邻域、三种 direction、stride=1/2/3/8/32；额外 CSV 检查覆盖 block=1/31/1024、向量合法偏移、N=7 的 float4 尾部、非法参数和拒绝覆盖。CPU host 测试检查 sector 模型已知值及溢出。

```bash
CUDA_ARCH=sm_89 python3 commands/sanitizer/run.py global_copy
```

四类 sanitizer 分别检查连续/向量路径和三种 direction 的 stride=3、offset=7、N=4099。当前无共享内存协作，racecheck/synccheck 通过不证明未来同步实验正确。

## Timing / Profiling

Event 与 host batch 范围沿用 [方法](../../docs/methodology.md)。E2E 包含**整个已分配输入 H2D**、一次 kernel、**整个已分配输出 D2H**及同步，排除输出 sentinel 复位。因此 stride 扫描的 E2E 传输量不恒定，不能把 E2E 变化全部归因为 coalescing。

```bash
CUDA_ARCH=sm_89 python3 commands/static/run.py global_copy
CUDA_ARCH=sm_89 python3 commands/nsys/run.py global_copy --variant scalar --stride 4 --n 1048576
CUDA_ARCH=sm_89 python3 commands/ncu/run.py global_copy --variant scalar --stride 4 --n 1048576
```

static 要求真实 SASS 有 64/128-bit load/store，不能只凭 float4 源码认定宽访问。NCU 先查询当前 sections/metrics，再选择 global request/sector、L1/L2、DRAM bytes/throughput、duration，并记录 replay/cache/clock 配置。缺少工具则 Not Tested，不硬编码不稳定名称，也不把插桩时间混入基线。

## Analysis、Exercises 与验收

实测见 [analysis.md](analysis.md)。完成 [练习](exercises.md) 后再看 [答案](answers.md)。验收要求：地址模型可复算；有效工作量一致且跨度单列；尾部/对齐验证；真实宽指令；原始统计；模型与事实分开；未测指标与推广边界明确。

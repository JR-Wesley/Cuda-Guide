# Stage 1 实验分析

日期：2026-10-23（Asia/Hong_Kong）。这是本次真实采集，未来结果另建目录，不改写这里链接的原始证据。以下性能数字都是特定环境中的观测，不是设备峰值或通用优化收益。

## 环境、实现和正确性

| 项目 | 本次记录 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU；CC 8.9；24 SM |
| 显存 | nvidia-smi 报告 8188 MiB |
| L2 | CUDA Runtime 报告 33,554,432 B（32 MiB）；未独立测量有效缓存容量 |
| 驱动 | 发布号 596.21；driver API 数字 13020 |
| Toolkit / Runtime | nvcc 13.4.92 / Runtime 数字 13040；教材固定 13.1 |
| Host 编译器 | 本次 GNU 16.2.1 |
| 编译 | C++17，Release，sm_89 + compute_89 PTX，`-O3 -lineinfo --ptxas-options=-v` |
| 默认配置 | block=256；grid_multiplier=4；warmup=5；repeats=21；iterations=100 |
| 输入与内存 | seed=20261023，有限 FP32；不别名；pageable host；设备缓冲区重复使用 |
| 时钟/功率 | 未锁频；仅记录入口 pstate/温度/功耗；后台干扰未隔离 |

环境依据在各次运行的 `001_nvidia-smi.log`、程序日志和 CSV；构建参数见每次 `compile_commands.json`。例如 [大规模 Add 的完整记录](../../results/20261023T141659Z_run_07rvbd0a/)。

已通过 [18 个 GPU shape 测试、额外 8 个 block 配置及参数/CSV 检查](../../results/20261023T140811Z_verify_7xd76rnk/verification.txt)，[CTest 原始输出](../../results/20261023T140811Z_verify_7xd76rnk/014_ctest.log) 保留每个用例。Copy 全量逐位一致；Add 按 `atol=rtol=1e-6` 全量验证。每个正式采样后都验证输出。

[Compute Sanitizer 记录](../../results/20261023T141621Z_sanitizer_h19d9rnu/) 的 memcheck/initcheck/racecheck/synccheck × 两个程序共 8 次运行均退出 0，无错误或 hazard 报告；输入 N=4099，覆盖全部版本。这不是未测输入的证明；本阶段无 shared 同步，racecheck/synccheck 的检查范围有限。插桩下程序打印的耗时不进入下表。

## 理论预测与数据规模

Add 的 requested bytes=`12N`，Copy=`8N`；二者都是较低算术强度。N=2^20 时工作集是 12/8 MiB，小于 API 报告的 L2；N=2^24 时是 192/128 MiB，大于报告的 L2。预测是：小规模可能由缓存复用与固定成本决定；大规模更可能受到较慢存储层吞吐约束。没有 NCU 指标，不能把这些预测直接确认为缓存/DRAM 事实。

grid-stride 在本机最多用 24×4=96 blocks；基线在两种 N 下分别用 4096 / 65536 blocks。减少 block 数可能减少调度工作，但增加循环控制与每线程工作，故不预设优化一定成功。Runtime D2D 是 API 参考，不把差异全部归因于一个因素。

带宽均以 `requested_bytes/(Event span per op ×1000)` 换算，GB/s 为十进制。未测量理论硬件带宽上限、真实 DRAM bytes、cache hit、sector 数或 achieved occupancy；这些项不能填零。

## 独立基线结果

每个时间单位为 **us/op**；E2E 是一次含传输的操作。Event 的 Median/Min/P95 对批量平均样本统计，不是单 kernel 活跃时长分布。21 样本的原始行在链接 CSV 内。

| 实验 / N | 版本 | Event Median | Min | P95 | 有效 GB/s | Host batch Median | Host E2E Median | 证据 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Add / 2^20 | scalar | 13.814 | 12.974 | 16.527 | 910.85 | 14.072 | 1734.902 | [CSV A](../../results/20261023T141657Z_run_2oxr279c/samples.csv) |
| Add / 2^20 | grid_stride | 10.231 | 9.623 | 13.790 | 1229.88 | 10.511 | 1718.771 | [CSV A](../../results/20261023T141657Z_run_2oxr279c/samples.csv) |
| Add / 2^24 | scalar | 889.682 | 886.354 | 892.414 | 226.29 | 890.629 | 23928.480 | [CSV B](../../results/20261023T141659Z_run_07rvbd0a/samples.csv) |
| Add / 2^24 | grid_stride | 869.056 | 866.847 | 873.052 | 231.66 | 869.759 | 22416.279 | [CSV B](../../results/20261023T141659Z_run_07rvbd0a/samples.csv) |
| Copy / 2^20 | scalar | 12.216 | 9.910 | 18.074 | 686.67 | 12.499 | 1223.841 | [CSV C](../../results/20261023T141708Z_run_dlehoff7/samples.csv) |
| Copy / 2^20 | grid_stride | 10.679 | 8.264 | 13.814 | 785.52 | 10.918 | 1086.728 | [CSV C](../../results/20261023T141708Z_run_dlehoff7/samples.csv) |
| Copy / 2^20 | runtime | 8.571 | 7.537 | 14.356 | 978.73 | 9.384 | 1126.760 | [CSV C](../../results/20261023T141708Z_run_dlehoff7/samples.csv) |
| Copy / 2^24 | scalar | 594.422 | 589.496 | 597.420 | 225.80 | 595.504 | 14849.114 | [CSV D](../../results/20261023T141709Z_run_8wzhoxek/samples.csv) |
| Copy / 2^24 | grid_stride | 583.414 | 581.558 | 586.998 | 230.06 | 584.510 | 14649.319 | [CSV D](../../results/20261023T141709Z_run_8wzhoxek/samples.csv) |
| Copy / 2^24 | runtime | 594.034 | 591.329 | 596.695 | 225.94 | 594.868 | 14713.520 | [CSV D](../../results/20261023T141709Z_run_8wzhoxek/samples.csv) |

复算示例：大规模 scalar Add 请求 201,326,592 B，`201326592/(889.681930542×1000)=226.2905 GB/s`。大规模 scalar Copy 的 Event 时间约为 Add 的 2/3，与请求字节比例相近；这是共同访存限制假设的支持线索，不能替代 DRAM 指标。

小工作集的有效带宽明显高于大工作集，与缓存复用假设相容；然而 N 同时改变 grid 和固定成本占比，因此尚未隔离缓存因素。小 Copy 的 P95 与 Min 差距较大，不能将一次顺序运行中的 variant 排名推广为稳定排序。

大规模 grid-stride 的观测 Median 小于 scalar：Add 约 2.3%，Copy 约 1.9%。这里只描述这次差值：尚未做跨进程 A/B/B/A、锁频或计数器归因，不能确认每种设备和时刻都有相同收益。寄存器更多但观测更快，说明单看 register 数不足以决定性能；具体原因仍需调度/内存指标验证。

E2E 比设备批量平均长得多，主要时间区间还包含 pageable H2D/D2H 与 host/API 等待。这里只能得出“设备驻留吞吐与当前 host 端到端不是同一目标”；尚未分离 PCIe、staging 和 host 调度的各自比例。

## 短 kernel：批量与单次 Event

固定 N=4096、scalar、block=256、warmup=5、21 样本，仅改变 K。

| K | Event Median / Min / P95 (us/op) | Host batch Median (us/op) | Host E2E Median (us) | 证据 |
| ---: | --- | ---: | ---: | --- |
| 1 | 7.200 / 5.120 / 26.624 | 30.929 | 60.566 | [K=1](../../results/20261023T141718Z_run_3pbgo5fv/samples.csv) |
| 100 | 6.953 / 5.366 / 12.104 | 7.379 | 66.411 | [K=100](../../results/20261023T141719Z_run__iirax8f/samples.csv) |

预测：批量摊薄 Event/host 等固定影响。观测：host batch 每次明显减少，Event Median 只小幅变化，P95 下降；E2E 不随 K 摊薄，因为始终测一次含传输操作。不能将这种变化解释为内核算法加速，也没有足够 K 点声称已经收敛。

## Nsight Systems：设备活跃时长与 span 的差别

[原始时间线](../../results/20261023T141752Z_nsys_o8egx3xz/timeline.nsys-rep) 使用相同 N=4096、scalar、block=256、K=100、warmup=5，但 repeats=3，单独插桩采集。共观察到 324 次内核，包含校验、预热和 E2E 内核；其数量与程序流程 `1+5+3×(5+100+1)=324` 一致。

[NSYS kernel 统计](../../results/20261023T141908Z_nsys-analysis__f0za1ak/summary_cuda_gpu_kern_sum.csv) 显示活跃时长 Median **0.992 us**、Min 0.960 us、Max 1.184 us。该次 [插桩 CSV](../../results/20261023T141752Z_nsys_o8egx3xz/instrumented_samples.csv) 的 Event span/op Median 为 **7.058 us**。两个值不是同一范围：批量 span 含 kernel 之间的空隙，单次 kernel 活跃只有约 1 us。API 统计还显示提交与同步开销，但不据此给未插桩运行精确分摊每一部分。

这直接支持测量框架使用 `event_span_us_per_op` 命名的必要性。独立 K=100 基线仍采用 6.953 us，不用 profiler 内的 7.058 us 替换。统计导出命令及退出码保存在 [导出记录](../../results/20261023T141908Z_nsys-analysis__f0za1ak/commands.jsonl)。

## PTX/SASS 静态观察

[本次静态检查](../../results/20261023T141321Z_static_8dc7bvhl/) 确认 PTX global load/store 存在；Add 的 SASS 含 `LDG`、`FADD`、`STG`，grid-stride 有实际循环回跳。源程序中的一条 C++ 操作还需索引、边界与地址计算，不能只数加法指令预测总时间。

| Kernel | registers/thread | static shared | stack / local | 依据 |
| --- | ---: | ---: | --- | --- |
| Add scalar | 12 | 0 B | 0 / 0 B | [Add resources](../../results/20261023T141321Z_static_8dc7bvhl/vector_add.resource-usage.txt) |
| Add grid_stride | 14 | 0 B | 0 / 0 B | 同上 |
| Copy scalar | 8 | 0 B | 0 / 0 B | [Copy resources](../../results/20261023T141321Z_static_8dc7bvhl/memory_copy.resource-usage.txt) |
| Copy grid_stride | 10 | 0 B | 0 / 0 B | 同上 |

[ptxas 编译日志](../../results/20261023T140745Z_build_e8_x9zhd/012_cmake.log) 还报告四个内核均为 0 spill loads/stores。这里的 0 来自实际静态报告；缓存命中率等未测指标不填 0。资源报告不能证明 achieved occupancy 或运行中的缓存访问行为。

## 已发现的限制与失败

1. **Toolkit 与驱动 PTX JIT 不兼容。** 原生 sm_89 cubin 运行通过，但设置 `CUDA_FORCE_PTX_JIT=1` 后失败于 `cudaFuncGetAttributes`，原始错误为 `cudaErrorUnsupportedPtxVersion: the provided PTX was compiled with an unsupported toolchain.` [失败日志](../../results/20261023T141840Z_run_2lh0m_0c/014_vector_add.log)、[环境变量与命令入口](../../results/20261023T141840Z_run_2lh0m_0c/invocation.json) 均保留，没有产生 `samples.csv`。静态 PTX 版本为 9.4。当前结论限定为原生 SM89 路径，不声称 PTX 可移植性已通过。
2. **Nsight Compute 不可用。** 当前 PATH 未找到 `ncu`，[状态为 Not Tested，退出 77](../../results/20261023T141753Z_ncu_rhoaug_y/status.txt)。L1/L2 hit、sector/transaction、DRAM bytes/throughput、achieved occupancy/stall 均未测量。没有为此修改权限或安装工具。
3. **其他架构未测。** SM7x/其他 SM8x、精确 Toolkit 13.1 构建都 Not Tested；当前 13.4 的目标列表也不包含 SM70。不能从 SM89 成功推出旧架构编译通过。
4. 最初普通沙箱中的 NVML 访问失败被保留；获准设备访问后的运行成功。静态/host-only 记录中的探测失败不等于那些 CPU 检查失败。[host-only 模式](../../results/20261023T141730Z_host-test_d87_o8wy/status.txt) 已实际通过。

## 证据复盘：从代码到下一轮实验

本节在已有数据之后补充，属于事后技术复盘；原始预测和采样记录以上文链接为准。保留的是可重新计算、可以被反驳的分析过程。

**第一轮：从读写量判断值得优先调查的方向。** scalar Add 每元素读 A/B、写 C，所以是 `4+4+4=12 B` 和一次加法，算术强度 `1/12 FLOP/B`；Copy 是 `8 B`、无核心算术。若二者在大规模上受到相近的有效访存吞吐限制，那么相同 N 应近似 `tAdd/tCopy=12/8=1.5`。实测 `889.682/594.422≈1.497`，支持继续调查访存路径，而不是先优化一条 FADD。该比值不唯一证明 DRAM 饱和：请求字节可能经过缓存，且没有测量真实流量。下一步 S2 固定 Copy 的逻辑字节，改变 lane 地址映射，观察“相同字节是否仍有相同时间”。这一轮已实施，见 S2。

**第二轮：从源码循环到资源代价。** grid-stride 将大规模 grid 从65536减至96 blocks，但每线程承担更多元素，并执行索引更新、比较和回跳。SASS 确认循环仍存在；资源从 Add 的12增至14 registers、Copy 的8增至10，未出现 spill。因此“减少 blocks”等于“减少全部工作”不成立。实测只有约2%的差异，远小于 grid 数量变化，说明不能按 block 数等比例预测收益。当前决定是保留两版本作为调度对照，而不把 grid-stride 固定为所有后续实验的赢家；稳定收益仍需顺序交换。S2 后来的漂移进一步提醒：不同时间采集的小差值不能拼成稳定排序。

**第三轮：从异常时间尺度反查测量对象。** N=4096 时 requested=49152 B；沿用大规模226 GB/s只是经验参照，得到约 `49152/(226×1000)=0.217 us` 的纯字节时间。实际 Event 约7 us，差距足以提示“仅按大规模带宽预测短任务”的模型不适用，但不能直接把差值全叫 launch latency。K=1→100 后，Host batch 每次30.929→7.379 us，Event 7.200→6.953 us：host固定成本被摊薄，设备 span 仍有其他组成。因此选择 NSYS 解决“kernel 本身长，还是间隙长”的歧义。

时间线给出约0.992 us kernel 活跃 Median，同次 Event span/op为7.058 us，确认二者不是同一计时对象。两种统计的样本集合不同，不能机械相减得到精确的每次间隙；但足以修正报告命名和解释。结果促使后续所有阶段沿用 `event_span_us_per_op`，短 kernel 不再把该列称作纯硬件执行时长。进一步用 K=10/1000 检验批量收敛仍待测。

**第四轮：性能目标回到端到端。** 大规模 scalar Add 的 E2E 23928 us，而驻留设备的 batch/op约890 us，二者相差约27倍；E2E包含pageable上传、回读和同步，因此不能通过加速FADD解决全部端到端时间。下一轮 Stage 6 将固定总数据和计算，分别改变pageable/pinned、stream与chunk，并用时间线判断是否重叠。这是后续实验动机，尚不是“已证明 PCIe 或 staging 占某个比例”的结论。

## 阶段交接

Stage 1 的正确性、计时、原始样本、静态检查与时间线验收已完成。最直接的学习结论是：请求带宽、批量设备 span、单 kernel 活跃时间、host E2E 各自回答不同问题；必须先确定范围再解释数字。

S2 从当前 scalar Copy 出发，先画 lane→sector 地址映射，再仅改变 stride；保持逻辑输出与有效工作量可比，并单独记录地址跨度带来的工作集变化。在可以使用 NCU 的环境补 sector/cache/DRAM 证据之前，相关机制归因继续保持“假设”。短 kernel 的 K=10/1000、跨进程顺序交换与更广架构复现作为后续练习，不伪装成本次已测内容。

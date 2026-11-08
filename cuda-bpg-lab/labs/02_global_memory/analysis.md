# Stage 2 分析：模型、指令与时间分别验证

日期：2026-10-23，Asia/Hong_Kong。设备在本轮重新查询：RTX 4060 Laptop GPU、SM89、24 SM、驱动 596.21、nvcc 13.4.92；Runtime 报告 L2=32 MiB。教材仍为 13.1。所有性能结果是未插桩原生 SM89 运行；旧的 PTX JIT 兼容性失败未通过改动驱动或删除检查规避。

## 实现与验收证据

- [27 个 GPU CTest 用例](../../results/20261023T145655Z_verify_n_qo85lc/014_ctest.log) 和 8 组额外配置全部通过。每次逻辑数据和所有输出洞/前缀逐位比较，覆盖 N 小于向量宽度、非整 warp/block、block=1、gather/scatter/both、合法偏移及非法对齐。
- [16 次 sanitizer 运行](../../results/20261023T145755Z_sanitizer___y1_d_0/) 覆盖四种工具和四组路径，无错误/hazard 报告。其计时不纳入基线。
- [静态检查](../../results/20261023T150441Z_static_npsfrq7e/) 确认 scalar、float2、float4 的实际 load/store，且 SASS 访问宽度为 32/64/128 bit；向量版本另保留 scalar 尾部。PTX 分别有 scalar、v2、v4 访问。对应寄存器为 10/10/14，每线程 local 和 static shared 均报告 0 B。
- [补充审计](../../results/20261023T150442Z_stage2-audit_fve49yp8/status.txt) 检查了禁止零预热、拒绝覆盖以及首轮全部 CSV 的字节数、sector 模型、带宽单位和原始样本统计复算。模型验证不是硬件计数器验证。

## 共同测量条件与模型

block=256、warmup=5、repeats=21、iterations=100；gather；同一个确定性逻辑输入序列；只读输入、独占输出；设备缓冲区重复使用，不 flush cache；pageable host。编译参数和源码/二进制快照保存在每次记录中。表中延迟为 Event span/op，Median/Min/P95 单位 us，有效带宽是逻辑 `8N` 字节除该 Median，并非 DRAM 实测带宽。

向量化假设：连续 scalar 已经合并良好，宽访问可能减少指令和线程调度；较大工作集若受访存吞吐约束，收益可能消失。stride 假设：读取 sectors 增加，sector 利用率下降；同时地址跨度和 cache footprint 变大，必须承认混杂。offset 假设：未对齐增加首 warp 的边缘 sector，但缓存复用和噪声可能掩盖差异。

## 连续与向量化对照

| N / 有效工作集 | 版本 | Median / Min / P95 us | 有效 GB/s | 首个完整 warp 的模型读 sectors | 证据 |
| --- | --- | --- | ---: | ---: | --- |
| 2^20 / 8 MiB | scalar | 18.913 / 12.206 / 45.196 | 443.53 | 4 | [小规模 CSV](../../results/20261023T150325Z_run_p69uzi8l/samples.csv) |
| 同上 | float2 | 12.646 / 8.929 / 16.241 | 663.32 | 8 | 同上 |
| 同上 | float4 | 8.929 / 7.608 / 14.490 | 939.45 | 16 | 同上 |
| 2^24 / 128 MiB | scalar | 595.948 / 587.919 / 601.208 | 225.22 | 4 | [大规模 CSV](../../results/20261023T150329Z_run_67jo82y8/samples.csv) |
| 同上 | float2 | 591.674 / 589.384 / 596.326 | 226.84 | 8 | 同上 |
| 同上 | float4 | 593.920 / 591.084 / 596.419 | 225.99 | 16 | 同上 |

不同宽度的每个 warp 处理 32/64/128 个 float，理想 sector 利用率相同。大规模三个版本约 225–227 GB/s，未观察到接近向量宽度倍数的加速；这与“相同有效字节且吞吐限制相近”假设相容，但尚无硬件指标确认瓶颈。小规模 float4 的观测 Median 更低，然而 scalar 的 P95 波动很大，不能给稳定倍数或把收益全归因于宽指令。

## 固定 N 的 stride 扫描

N=2^22，逻辑有效字节始终 32 MiB，只改变读取 stride，offset=0。以下分配量已考虑最后一个元素后不分配无用洞。

| stride | 模型读/写 sectors（首 warp） | allocation bytes | Median / Min / P95 us | 有效 GB/s | 证据 |
| ---: | --- | ---: | --- | ---: | --- |
| 1 | 4 / 4 | 33,554,432 | 41.216 / 38.728 / 43.407 | 814.11 | [s1](../../results/20261023T150340Z_run_pxpfe9z3/samples.csv) |
| 2 | 8 / 4 | 50,331,644 | 209.674 / 207.327 / 210.258 | 160.03 | [s2](../../results/20261023T150342Z_run_h5zy0kin/samples.csv) |
| 4 | 16 / 4 | 83,886,068 | 344.627 / 341.965 / 346.857 | 97.36 | [s4](../../results/20261023T150344Z_run_x_w13h6o/samples.csv) |
| 8 | 32 / 4 | 150,994,916 | 619.581 / 616.828 / 623.165 | 54.16 | [s8](../../results/20261023T150347Z_run_tcrcedy9/samples.csv) |

结果方向符合“稀疏读取更慢”的预测；幅度不能只由首 warp 的 sector 比例解释。stride=1 的分配/有效工作集位于 API 报告的 L2 容量附近，而更大 stride 扩大地址跨度和实际 sector footprint；cache 层次变化也是合理解释。因为没有 NCU，缓存 hit、实际 sectors 和 DRAM bytes 均 **Not Tested**。不将有效带宽下降直接等同于 DRAM 浪费比例。

E2E 扫描包含整个分配区域的传输，随 stride 增大而增加，CSV 明确记录 `e2e_transfer_bytes`。E2E 不构成等量传输对照，不能用它单独判断 kernel coalescing。

## Misalignment 与反例

固定 N=2^22、stride=1，offset 以 float 计。

| offset | 首 warp 模型读 sectors | Median / Min / P95 us | 证据 |
| ---: | ---: | --- | --- |
| 0 | 4 | 41.216 / 38.728 / 43.407 | [o0](../../results/20261023T150340Z_run_pxpfe9z3/samples.csv) |
| 1 | 5 | 45.444 / 43.264 / 47.493 | [o1](../../results/20261023T150351Z_run_jeerak9q/samples.csv) |
| 8 | 4 | 50.330 / 48.753 / 54.077 | [o8](../../results/20261023T150354Z_run_9nz8n35u/samples.csv) |

offset=8 已重新对齐 sector，首轮却比 offset=1 更慢。因此“只由首 warp sector 数决定总耗时”的模型被本次数据反驳；不能选取符合预期的前两行而删除第三行。

为调查顺序漂移，又做 A/B/B/A（A=8、B=0）独立进程复测：

| 次序 | offset | Median / Min / P95 us | 证据 |
| ---: | ---: | --- | --- |
| A1 | 8 | 55.110 / 50.135 / 92.364 | [A1](../../results/20261023T150552Z_run_o75vygxz/samples.csv) |
| B1 | 0 | 53.903 / 40.141 / 85.289 | [B1](../../results/20261023T150555Z_run_b1i20xlg/samples.csv) |
| B2 | 0 | 96.809 / 90.378 / 102.431 | [B2](../../results/20261023T150557Z_run_n4lw9rm3/samples.csv) |
| A2 | 8 | 101.489 / 96.963 / 106.179 | [A2](../../results/20261023T150601Z_run_78opmiok/samples.csv) |

同一 B 的跨进程变化远大于首轮 offset 差异。这说明当前会话不足以确认小幅 alignment 性能收益；具体漂移来源（频率、功耗、后台负载或其他因素）未隔离。没有锁频、持续监测或硬件计数器，不能断言是其中某一项。模型、正确性、宽指令的证据仍有效，但性能结论降为本次观测，不能推广为稳定排序。

## 证据复盘：为什么改变了后续实验设计

本节为事后复盘，不冒充预先记录的预测。目标是让读者沿源码地址式、指令和数据复核每一次结论调整。

**从 lane 地址到向量化的有限预期。** 对齐连续 scalar 的 lane l 访问 `base+4l`；32 lanes覆盖128 B，即4个32 B sector。float4 的lane l访问16 B，warp覆盖512 B，即16 sectors；每warp搬运的数据也增加4倍，因此“sector变多”不代表合并更差。固定 N，线程和warp数降为约1/4，但逻辑字节仍为8N。静态检查看到 LDG/STG.128，排除了“编译器根本没生成宽访问”这一解释。大规模时间仍约592–596 us，于是结论收窄为：宽指令确实生成，却没有带来4倍吞吐；具体是访存还是其他吞吐限制仍需NCU。下一步不盲目增加向量宽度，而调查数据复用与工作集。

**从 stride 数据反查遗漏变量。** gather s=1→2 时，模型每warp读/写sector从4/4到8/4，总数比为12/8=1.5。实测时间比 `209.674/41.216≈5.09`。这否定了“时间仅等于这个sector总数乘同一常数”的简化预测。检查地址式 `(N-1)s+1` 发现读取跨度也变大；连同输出，分配从约32 MiB到48 MiB，且稀疏sector footprint增加。缓存状态是合理竞争解释，但仅凭容量附近的拐点不能确定L2行为。为隔离这个因素，S3新增固定 T/S/访问次数、仅改变 W 的工作集实验，并检查SASS中的load没有移出循环。该设计已实施；它回答新的问题，不假装能从另一kernel反推出本次精确hit率。

**从 offset 反例改变采样安排。** offset0和8都预测4个读sector，offset1预测5个；首轮却是o8比o1慢。若只保留o0/o1，就会把不完整证据写成alignment结论。于是保留第三点，并做A/B/B/A，选择同配置B重复作为漂移对照。B本身53.903→96.809 us，变化约80%，已大于首轮offset差异；因此撤回稳定性能排序，而不撤回可从地址计算的sector模型。这里的优化决策是修正证据强度，不是为了符合模型改写kernel或删掉数据。

**尚待执行的区分实验。** 同一进程中交错多轮variant并保留时间顺序，可减少跨进程状态混杂，但不能消除温度、频率或后台负载；同步记录这些状态可帮助筛查，仍不能仅由相关性断言因果。在NCU可用环境先查询指标，再比较同工作量的请求、L2和DRAM流量；若实际流量不随理论sector量同比增长，要进一步检查跨warp复用。以上都列为待测，不从当前CSV填补未知指标。

## 阶段交接

目前仅原生 SM89 / Toolkit 13.4；其他 SM7x/8x 和精确 Toolkit 13.1 未复现。ncu 在本轮每次工具探测中仍不可用，L1/L2 hit、sector/transaction、DRAM bytes、stall、achieved occupancy 皆 Not Tested。Stage 2 提供 NSYS 命令，但本阶段未另采时间线；S1 已验证计时范围差别。

基础实验验收完成：四类访问实验可运行，结果一致、边界与非法配置检查通过，模型和静态指令可复核，负面结果与漂移已保留。计数器机制归因仍待可用工具；可靠的小幅性能排序需在更稳定环境复测。下一步进入 S3 的共享内存转置与 padding，并保持每一优化步骤的基线和资源代价。

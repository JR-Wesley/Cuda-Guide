# Eviction Priority 实测分析

> 当前复现入口已迁到 `commands/<action>/run.sh`：先按专题 README 显式 compile，再执行相应动作。下文命令是当前复现写法，历史精确命令仍以原日志为准；本轮未重新采集 GPU 数据。

本报告区分提示语义、机器指令和历史实验观察，不用耗时反推缓存命中率。文本证据已合并到 [sm_89-release.log](results/sm_89-release.log) 的 `HISTORICAL IMPORT` 动作节，日期目录、旧二进制和源码/脚本副本已清理。本次未真实编译或运行 GPU，新独立动作入口尚未进行真实 CUDA/GPU 验证；源码核对与复现条件见 [记录索引](results/README.md)。

日志位于脚本同目录 `commands/<action>/<variant>.log`，同动作同配置重跑覆盖（包括失败），其他动作日志不变。命令、实际参数、原始输出及退出码均记录；检查器的临时步骤文件在退出后清理。不生成自动快照、时间戳归档或 history。已有 `results/` 原始日志和历史报告保持原样，当前动作不写入。

## 1. 环境与验证

| 项目 | 历史配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，sm_89，24 SM |
| L2 容量 | 32 MiB；本实验改变的是 L1 提示 |
| Toolkit | CUDA 13.4，nvcc V13.4.92 |
| Driver API / Runtime API 版本 | 13020 / 13040 |
| 编译 | C++17、`-O3 -lineinfo`、sm_89 + compute_89 PTX |
| 热点 | 16 KiB，每线程 128 节点地址环 |
| 背景数组 | 512 KiB，normal 提示读取；不保证清空缓存 |
| 保留 run 的干扰数组 | 16 KiB |
| 执行布局 | 每个 kernel 一个 block、32 个线程 |
| 采样 | 每配置预热 1 次、采样 11 次、轮换配置顺序 |

保留的 16 KiB run 中，10 组读写校验和 12 组热点探测均通过。缩小规模的 memcheck 也通过，报告 0 errors。

| 证据 | 单日志中的位置 |
| --- | --- |
| 16 KiB 干扰运行 | [配置日志](results/sm_89-release.log)：run 节的 `run.log`、`metadata.txt`、`status.tsv` |
| 静态分析 | [配置日志](results/sm_89-release.log)：static 节的 `eviction_priority.ptx/.sass`、`inspect.log` |
| 内存检查 | [配置日志](results/sm_89-release.log)：sanitizer 节的 `memcheck.log`、`sanitizer_version.log` |

256 KiB run 已由最新 16 KiB run 取代，原始证据不再保留。旧数值表、4.32 倍性能差异及跨规模量化比较已移除；static metadata 的 256 KiB 参数不代表进行了 GPU 计时，不能替代被删 run。

## 2. PTX 到 SASS 的映射

历史 sm_89 静态结果中，显式 global 地址的映射如下；generic-global 版本使用对应的 `LD.E` / `ST.E` 指令形式，保留同样的优先级后缀。

| L1 提示 | global load | global store |
| --- | --- | --- |
| normal | `LDG.E` | `STG.E` |
| first | `LDG.E.EF` | `STG.E.EF` |
| last | `LDG.E.EL` | `STG.E.EL` |
| unchanged | `LDG.E.EU` | `STG.E.EU` |
| no_allocate | `LDG.E.NA` | `STG.E.NA` |

这证明编译器保留了可区分的指令形式，不证明硬件在所有访问中都严格实现一个抽象的优先级队列。

全部 20 种 PTX 写法均存在。12 个 probe kernel 的两个 `SR_CLOCKLO` 读取之间均保留带 LDG 的回跳循环；噪声组计时区间为 `0x940..0x9f0`，无干扰组为 `0x590..0x640`。结合源码中加载结果到下一地址的依赖，排除了提前加载固定数据、只对加法计时的失效方式。

## 3. 16 KiB 干扰：容量之外还有提示与访问历史

热点和干扰各 16 KiB。下表是完成 128 步 probe 的 SM cycles，**不是单条加载延迟或 L1 命中率**；计时还包含循环控制、地址计算与同步成本。

| 配置 | probe 中位数（cycles） |
| --- | ---: |
| no_noise | 8713 |
| baseline | 8713 |
| hot_first | 27713 |
| hot_last_unchanged / hot_last_normal | 8713 / 8713 |
| hot_first_normal | 8713 |
| noise_first / noise_last | 8713 / 8713 |
| noise_unchanged / noise_no_allocate | 8713 / 8713 |
| last_plus_noise_first / last_plus_noise_no_allocate | 8713 / 8713 |

保留 run 中 baseline 与 no_noise 的 probe 中位数相同；没有跨规模对照可用来量化更大干扰的成本。`hot_first` 较慢，范围 24950～28621 cycles，而 `hot_first_normal` 为 8713。这支持在此访问流程下把需复用热点标成 first 可能不利的判断，不证明精确的内部替换机制。

不能只比较“热点＋干扰总字节数”与标称容量。优先级、背景访问、地址映射和已有状态均可能影响替换；本例没有独立测量可用 L1 容量、组相联或策略内部细节，仅将它们列为解释因素。

`hot_last_normal` 与 `hot_last_unchanged` 在保留 run 中相同，但不能据此宣称 normal 与 unchanged 语义相同或 normal 是无条件降级。`noise_unchanged` 与 `noise_no_allocate` 同样没有表现出 probe 耗时差异，也不意味着 unchanged 等于不分配：它们的 PTX 语义与机器码仍不同。

## 4. 如何理解已删参数对照

可以设计更大干扰与较小干扰的实验来检验缓存竞争假设；如果其他条件固定且 baseline 相对 no_noise 的成本随干扰规模变化，这与竞争解释一致。但旧 256 KiB 证据已删除，本报告不再据此声称某一保护策略在该规模下产生多少加速。

提示的语义仍然成立：last 表达尽量晚驱逐，first 表达优先驱逐，no_allocate 表达本次访问不分配该层缓存；它们不是常驻、命中或同步保证。没有硬件计数器时，不能把某个周期数直接标为“全部 L1 命中”或“全部 L2 命中”。

## 5. 正确性与适用边界

保留 run 的所有提示 load/store 都通过逐元素拷贝检查，包括 generic 指向 global 的访问形式。功能正确不等于相同缓存行为；缓存行为不同也不应使只读输入、按顺序取回输出的程序得到不同数值。

性能数据只来自 load 造成的热点竞争，未测量 store 的写分配或驱逐行为；L2 策略、跨 SM 生产者/消费者和 CPU/GPU 共享访问未纳入。没有采集 L1/L2 命中率或物理流量。

背景数组读取不是硬件 flush，也不保证独立 kernel 样本的 L1 初态完全相同。单次样本内各阶段位于同一 SM，跨样本可能换 SM；合并日志保留原始 SM 编号。

这些现象仅适用于记录中的编译器、sm_89、数组布局与访问流程。用于实际程序时，应先保证正确同步，再根据真实热点和竞争数据选择提示，并检查生成指令、正确性与性能。

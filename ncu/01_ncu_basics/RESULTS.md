# 第一课实测与解读

> **当前状态：** 本次 stride 模板源码仅通过静态文本检查，相关脚本 `bash -n` 通过；未编译运行，旧日志不能验证当前模板源码与调用。
>
> **改前版本的历史验证（Vector Multiply）：** 当时默认入口已编译并运行通过。已执行 `CUDA_ARCH=sm_89 bash commands/compile.sh`（nvcc 13.4、C++17、`-O3 -lineinfo`），见[编译日志](log/compile.log)；`bash commands/run.sh` 核对 N=16、mismatches=0、PASS/exit 0，见[运行日志](log/run.log)。四个共享 runner 和第 10 课已采用 `DeviceBuffer<T>`，共享计时使用 `CudaEvent`，第 11 课保留显式分配释放。
>
> 当时未测试全部参数；profile 仅通过 `bash -n`，未执行 NCU、NSYS、workers 或 SASS 检查，无性能结论。**下方 Vector Add 历史正文（包括“本轮新入口验证”）不代表当前 Multiply。** 历史命令、数字与旧证据不改；旧 commands 证据在 `log/legacy_commands/`，旧 build 产物与 `runs/` 保持原地。当前操作见 [lab01.md](lab01.md)，历史文字中的“本轮”“当前”“新”均指原记录当时。

实测日期：2026-11-06。操作方法见 [实验文档](lab01.md)，理论依据见 [第一课讲义](lec01.md)。

> 路径与证据边界：当前源码为 `kernels/main.cu`，页首编译/运行日志仅属 stride 模板修改前的历史结果，不验证当前源码；下面包括“本轮新入口验证”在内的 Vector Add 测量均为历史记录，旧 `main.cu` 命令保持原样，不与新日志混用。

本文记录的是上述日期的**历史实测**。`runs/` 记录由旧脚本生成，保留原地；目录内的日志、源码与脚本快照内容均未改动，下列既有实验数据保持不变。先前计划的历史迁移未执行，本文链接继续指向原 `runs/...`。**不创建或新增 `history/`，`runs/` 仅保留旧证据、不再新增归档，也不随本次入口调整移动或删除用户原始证据。**

**以下为旧入口的历史说明，不是当前操作步骤。** 当时入口统一为 `commands/<action>/run.sh`，顶层不再使用 `lab.sh` 分派。脚本进入本专题根目录执行，普通运行和采集使用共享的 `./build/vector_add`。各动作输出到终端及 `commands/<action>/output.log`，含 `set -x` 命令和退出码，重跑覆盖；尚未执行的动作没有该日志。新脚本不做哈希、自动快照、归档或自动构建。**本轮只验证新入口与最小采集，新日志不能冒充下面的大输入旧实测证据。**

用户已有报告 `build/vector_add_basic.ncu-repz` 保留原地，不移动、不删除、不以重采结果替代，本文不解读其数据。当时的新采集使用 `commands/<action>/report.ncu-rep`，显式后缀且不强制覆盖；已有同名报告须先手工改名保留再运行。历史权限失败不表示当前仍无报告；本轮新验证结果如下。

## 本轮新入口验证

| 动作 | 实际验证 | 新证据 |
|---|---|---|
| env | GPU/工具查询及指定 Metric 查询成功，无权限错误 | [output.log](log/legacy_commands/env/output.log) |
| compile | 显式 `CUDA_ARCH=sm_89`，编译成功 | [output.log](log/legacy_commands/compile/output.log) |
| run | N=16，逐元素核对 PASS | [output.log](log/legacy_commands/run/output.log) |
| basic | N=16，选中一次 `vector_add`，实际采集 8 passes，退出码 0 | [output.log](log/legacy_commands/basic/output.log)、[报告](log/legacy_commands/basic/report.ncu-rep) |

已执行 `ncu --import commands/basic/report.ncu-rep --page details`，可正常读取；报告的 Grid=1、Block=256，Duration=1.86 μs。这是 NCU 下小输入的测量，不与下文 $N=2^{26}$ 的普通运行数据直接比较。采集过程中应用自身输出的 Event 时间包含分析器干扰，不能当作独立性能基线。

本轮未修改系统权限；只确认目前计数器访问已可用，不推断之前具体如何修复。其余 memory/scheduler/metrics/launch 新入口仅做语法检查，未执行，也没有预填日志。没有重跑旧的大输入实验或 GUI 检查。各动作重跑会更新自己的日志，阅读时须核对 N 与命令是否仍对应本段描述。

## 1. 历史实测的完成状态

- **通过**：Bash 语法检查、CUDA 编译、N=1/16/257/1,048,576/67,108,864 的逐元素核对。
- **符合预期的负向检查**：N=0 被拒绝，退出码 1。
- **完成**：普通运行的 CUDA Event 批量计时、单变量 N 对照、大输入三次独立进程重复。
- **部分完成**：NCU 版本、Set、Section 查询成功；硬件 Metric 查询报权限错误。
- **失败/受阻**：当时 basic 和 LaunchStats 实际采集均为 `ERR_NVGPUCTRPERM`，退出码 1；这两次历史采集没有生成有效 `.ncu-rep` 可供解读，不代表当前报告文件的状态。
- **未执行**：Memory、Scheduler 和显式 Metric 的进一步采集、报告导入、GUI 查看。已知权限阻塞下不重复发起等价失败操作。
- **未测量**：NCU Duration、DRAM 字节数、SM/DRAM Throughput、指令数、Replay pass 数、缓存命中率；未运行 sanitizer、SASS 检查或 NSYS。这些不填零，也不借用理论笔记中的假设数据。

## 2. 环境与测量口径

| 项目 | 本次值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Compute capability | 8.9 |
| 驱动 | 596.21 |
| CUDA Toolkit | 13.4，nvcc V13.4.92 |
| Nsight Compute | 2026.3.1.0 |
| 编译 | `nvcc -O3 -lineinfo -arch=sm_89 main.cu -o build/vector_add` |
| Block | 256 线程 |
| 启动 | 10 次预热，随后 20 次 Event 区间内重复 |
| 核对 | 全部 N 个元素，不只检查打印出来的前几个 |

普通运行没有设置固定时钟、功率或缓存清理，也没有建立隔离 GPU 负载的严格性能环境。下列结果是本次入门实验观测，不是设备的性能上限。三次重复不是统计置信区间。

每个时间是 `Event 区间总时间 / 20`，包含可能的启动间隙；不含分配、CPU 初始化和 H2D/D2H 拷贝。有效带宽以每次启动的逻辑字节数 `12N` 为分子，单位为十进制 GB/s。

## 3. 普通运行原始记录

| N | Grid | 批量平均时间（μs/次） | 有效 GB/s | 状态与日志 |
|---:|---:|---:|---:|---|
| 1 | 1 | 6.246 | 0.002 | [PASS](runs/run.WKTWOI/output.log) |
| 16 | 1 | 5.837 | 0.033 | [PASS](runs/run.ClkdtD/output.log) |
| 257 | 2 | 5.376 | 0.574 | [PASS](runs/run.oo6k4N/output.log) |
| 1,048,576 | 4,096 | 11.520 | 1092.267 | [PASS](runs/run.6uLcNI/output.log) |
| 67,108,864，第 1 次 | 262,144 | 3534.387 | 227.849 | [PASS](runs/run.NEgFPL/output.log) |
| 67,108,864，第 2 次 | 262,144 | 3535.768 | 227.760 | [PASS](runs/run.RQJJ0P/output.log) |
| 67,108,864，第 3 次 | 262,144 | 3535.923 | 227.750 | [PASS](runs/run.RLmF1f/output.log) |

其他证据：[编译](runs/build.dmDmcO/output.log)、[环境与查询](runs/env.HZuf0h/output.log)、[N=0 拒绝](runs/run.v57MAp/output.log)。这些历史目录中的 `main.cu`、`lab.sh` 为当时该动作的源码和脚本快照，`exit_code.txt` 保存原始退出码；快照中的 `lab.sh` 名称保留历史原貌，不是当前入口。

### 3.1 为什么先看 N=16、257

N=16 输出从 `c[0]=2` 到 `c[15]=17`，与手算一致。
N=257 需要两个块，第二块只有 i=256 有效；全部 257 个结果通过，支持当前边界条件正常工作。结果核对不能代替内存检查器，但这里没有删除边界判断去制造错误。

这两组的平均时间相近，不说明 257 个元素与 16 个元素的“每条指令时间相同”。它们都非常短，Event 区间内的提交间隙、调度开销等可能显著；本次没有用时间线分解这些成分。

### 3.2 大输入的 227.8 GB/s 是怎样得到的

以第 1 次为例，原始 Event 总时间为 70.687744 ms：

$$
t_{\text{average}}=70.687744/20\ \mathrm{ms}\approx3.534387\ \mathrm{ms}
$$

$$
B_{\text{logical}}=67{,}108{,}864\times12=805{,}306{,}368\ \mathrm{Bytes}
$$

$$
BW_{\text{effective}}=\frac{805{,}306{,}368}{0.0035343872}\times10^{-9}
\approx227.849\ \mathrm{GB/s}
$$

**可以说**：本次条件下，按算法逻辑工作量计算，每秒完成约 227.8 GB 的等效数据处理。

**不能说**：已经测到 DRAM 传输了 805,306,368 字节，或已经证明 DRAM 达到峰值。前者需要硬件流量，后者还需要适当的峰值参照和进一步对照。

### 3.3 为什么小输入会算出 1092 GB/s

N=1,048,576 时，三个数组的总大小为 12,582,912 Bytes（12 MiB），批量平均时间 11.520 μs，因此逻辑有效带宽为 1092.267 GB/s。

- **观察**：这个数由程序输出的逻辑字节数和 Event 时间计算而来，不是 DRAM 计数器输出。
- **解释候选**：预热和重复访问同一批数组可能使缓存承担大量读写，CPU 提交节奏也影响短 kernel 的区间时间。
- **没有证明**：本次未测 L2 容量、命中率或 DRAM 流量，不能断言所有数据都命中了 L2，也不能把差异全归因于某一级缓存。
- **下一步验证**：在计数器权限可用时，对两个 N 采集相同 Memory 指标，并明确 NCU 的缓存控制与普通运行不同。缓存控制的单变量实验留给第二课。

因此，N 增大而有效带宽下降，并不与算法正确性矛盾，也不意味着较大问题“算错了”。

## 4. NCU 失败也是需要解释的实验结果

### 4.1 实际尝试

1. [Basic 采集](runs/basic.QC57CD/output.log)：`--set basic -s 10 -c 1`，选择第 11 次匹配的 `vector_add`。
2. [LaunchStats 采集](runs/launch.Bu1a5H/output.log)：缩小到启动信息 Section，判断是否仍被环境阻止。

两者均返回：

```text
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.
```

**程序 PASS 不等于采集 PASS**：这两个日志中程序仍完成了加法并打印 Event 时间，但 NCU 的退出码为 1，且没有有效报告。因此不把其中的 Event 时间混入上面的普通运行基线，也不把它称为 NCU Kernel Duration。

仅请求 LaunchStats 也失败，这是当时环境的历史实测行为；不能由“启动配置不是动态计数器”推断工具一定可以无权限采集。

### 4.2 为什么还不能回答 Replay 多少次

源码提交 30 次 Launch，命令意图是跳过前 10 次、选 1 次。这描述**应用启动次数与筛选条件**。
NCU 为所选启动实际做了多少采集 pass，必须看成功采集日志。上述历史采集受阻，不能把 30 次启动误认为 30 次 Replay，更不能编造 basic 的 pass 数。

### 4.3 退出码的一个陷阱

环境动作中，`ncu --query-metrics ...` 打印了 `ERR_NVGPUCTRPERM`，但本次返回 0。原始日志保留这一事实，不能把 `env` 的 `exit_code=0` 写成“指标查询全部通过”。

本次 `--list-sets` 确认 basic 包含 LaunchStats、Occupancy、SpeedOfLight、WorkloadDistribution；Set 的实际内容以安装版本输出为准，不把理论示意图当成固定配置。

## 5. 本课结论与下一步

已经实操建立：源码计算 → 线程索引 → 结果核对 → 多次启动 → Event 测量边界 → 逻辑流量计算。
NCU 的命令组织和失败诊断也已实践，但**仅凭上述历史记录，硬件指标证据链尚未闭合**。

当时的阻塞需要管理员确认/提供性能计数器访问权限；历史实验没有修改驱动配置、执行 sudo、重置 GPU 或切换其他用户绕过限制。此次未重新验证当前权限状态。

确认权限可用后，按当前 [lab01.md](lab01.md) 显式编译，再分别执行 `bash commands/run.sh 67108864`、`bash commands/profile.sh 67108864`。细分访存实验见第 05 课，不恢复 memory/metrics/scheduler 等跨课动作。记录实际 pass 数、报告单位与流量，完成报告核对；新结果与历史证据分开，不能改写原日志。新动作日志在本课 `log/`，重跑前明确保留方式；`log/profile.ncu-rep` 已存在则先手工改名保留，不新增自动归档。

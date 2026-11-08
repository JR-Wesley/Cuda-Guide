# Stage 1：先建立可信测量

## 当前独立动作约定

以下命令从项目根目录执行。先单独构建要使用的目标，再选择运行、校验或分析；后续动作缺产物只报错，不自动编译。源码或编译配置变化后必须重新执行构建。

`CUDA_ARCH=sm_89 python3 commands/build/run.py vector_add`
`CUDA_ARCH=sm_89 python3 commands/build/run.py memory_copy`

日志固定为 commands/<action>/log/，重复同动作覆盖该目录；先阅读或手动保留需要对照的结果再重跑。历史 results/ 不写入。本页历史分析链接不代表新入口已通过 GPU 复验。


BPG 是 **CUDA Best Practices Guide（CUDA 最佳实践指南）实验室**。优化前先回答两个问题：程序算对了吗？究竟量了哪段时间？否则，漏写元素的“提速”或把数据传输算进 kernel 的数字都无法用于比较。本阶段用两个独立程序练习：`vector_add` 算 `out[i]=a[i]+b[i]`，`memory_copy` 算 `out[i]=a[i]`。目标是自己能预测小输入结果、解释校验、区分计时区间，再比较一次只变一个条件的实验。指南背景可读 [Best Practices Guide 13.1](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-c-best-practices-guide/index.html) Ch.2（APOD：评估、并行化、优化、部署）、4、7、8、9；无需先学 profiler。

**前置知识**：C/C++ 数组 `a[i]` 与下标从 0 开始；编译把源码变成可执行文件；CPU（主机）与 GPU（设备）有各自的内存。这里 FP32 是每元素 4 字节的单精度浮点数，kernel 是 GPU 上并行执行的函数；block 是一组线程，grid 是本次启动的全部 block。尚未熟悉 CUDA 的话，先沿着 `vector_add.cu` 的 `main` 进入 `experiment.cuh` 的 `run<false>`、`execute`、`scalar_kernel` 阅读。

## 从预测到观察：每步在项目根目录 `cuda/cuda-bpg-lab` 执行

以下命令在有 NVIDIA GPU、CUDA Toolkit（含 `nvcc`）、CMake 和 Python 3 的环境中运行。示例设备为 compute capability（计算能力）8.9，故脚本用 `CUDA_ARCH=sm_89`；**先查本机实际值并替换**，项目仅接受 SM70–SM89，Toolkit 也须支持该架构。`python3 commands/env/run.py` 只记录工具/设备信息到 commands/env/log/，个别探测失败也可能返回成功，需看其中日志；也可直接运行 `nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv` 和 `nvcc --version`。如果没有 GPU，先读代码并手算，不把编译或 CPU 测试当作 GPU 验证。

### 1. 先在纸上预测 N=8 和 N=257

**目的**：用小数组看懂一个线程负责一个输出元素。读 `experiment.cuh` 的 `scalar_kernel`：`i=blockIdx.x*blockDim.x+threadIdx.x`，仅当 `i<N` 才访问；grid（block 数）是 `ceil(N/block)`。数组通过独立 `cudaMalloc` 分配，互不重叠。暂设 `--block 256`：N=8 只需 1 个 block，线程 0–7 写 `out[0]`–`out[7]`；N=257 需 2 个 block，第二个 block 只有线程 0 写 `out[256]`，其余线程跳过。例如若 `a[0]=2, b[0]=3`，Add 应为 5，Copy 应为 2；实际程序用固定生成的输入，不会恰好是这两个数字。

**动手/观察**：先画出 `blockIdx.x=1, threadIdx.x=0/1` 在 N=257 时的下标，再进入下一步执行。若预测访问越界，检查 `i<N` 是否在读取输入和写输出之前。

### 2. 编译单个实验，再运行小输入

**目的**：分清构建和运行。先用 CMake 配置（`-S` 源目录，`-B` 构建目录；架构填数字而非 `sm_89`），只构建 `vector_add` 目标；二进制会放在 `build/sm_89/bin/`。也可以改用 `CUDA_ARCH=sm_89 python3 commands/build/run.py vector_add`：它完成同样的配置和目标构建，不运行测试，构建日志写 commands/build/log/。

```bash
cmake -S . -B build/sm_89 -DCMAKE_CUDA_ARCHITECTURES=89 -DBPG_LABS=01_measurement
cmake --build build/sm_89 --target vector_add
./build/sm_89/bin/vector_add --help
./build/sm_89/bin/vector_add --variant scalar --n 8 --block 256 --warmup 1 --repeats 2 --iterations 3
./build/sm_89/bin/vector_add --variant scalar --n 257 --block 256 --warmup 1 --repeats 2 --iterations 3
```

**预期（不是本机实测）**：出现 `scalar PASS` 和计时摘要，而非打印 8 个数组元素；`--help` 不访问 GPU。源码在 `execute` 中用 CPU 计算 `reference`，GPU 输出先填成 NaN（未写元素易暴露），运行后下载并逐元素比较；Copy 逐位比较，Add 用绝对/相对容差 `1e-6`，拒绝 NaN/Inf。正式计时前及每次采样后还会校验。`PASS` 表示这些输入和版本通过，不是所有输入的证明。详见 [数值与计时规范](../../docs/methodology.md)。若配置失败，先查 `nvcc --list-gpu-code` 是否包含所选 `sm_XX`；若运行失败，先查 GPU/驱动、`--block` 上限和 `FAILED` 信息。CPU reference 不计作 CPU 性能基线。

### 3. 独立运行 Copy 并做边界核对

**目的**：分清“读一份、写一份”与 Add 的“读两份、写一份”。

```bash
cmake --build build/sm_89 --target memory_copy
./build/sm_89/bin/memory_copy --variant scalar --n 257 --block 256 --warmup 1 --repeats 2 --iterations 3
CUDA_ARCH=sm_89 python3 commands/verify/run.py vector_add
CUDA_ARCH=sm_89 python3 commands/verify/run.py memory_copy
```

**预期**：Copy 的第二个 block 同样只有一个合法线程；`verify.py` 还通过 CTest 测每个实验的 9 种 N（含 8 邻近的 1、31、32、33，及 257），并检查额外 block 大小、非法参数、CSV 格式与禁止覆盖。verify 不构建，将日志写 commands/verify/log/；若失败，先查具体 CTest/命令日志，不要把 `PASS` 当作性能结论。想只做无 GPU 的主机工具检查，先运行 `python3 commands/host-build/run.py`，再运行 `python3 commands/host-test/run.py`，但它不会验证 GPU 计算。

### 4. 明确计时区间，再测一个固定输入

**目的**：GPU launch 是主机提交任务，返回时设备可能还没做完。读 `common/benchmark/timing.hpp` 的 `measure_batch` 与 `experiment.cuh` 的 `upload`、`download`：同一 stream（顺序提交的任务队列）上的 CUDA Event 在 K 次操作前后记录，stop Event 同步后得到 `event_span_us_per_op`，即 **含设备操作间空隙** 的 Event 总跨度除 K，不等于纯 kernel 活跃时间。`host_batch_us_per_op` 是主机时钟从记录 start 到等到 stop 的跨度除 K；`host_e2e_us` 则是一次上传 H2D（主机到设备）、操作、下载 D2H（设备到主机）及等待，使用已有分配，不含分配时间。三者不是同一个量。

```bash
CUDA_ARCH=sm_89 python3 commands/run/run.py vector_add --variant scalar --n 1048576 --block 256 --warmup 5 --repeats 21 --iterations 100
```

**预期**：输出 `scalar PASS`、三种时间及有效带宽；`run.py` 将选项传给二进制，固定写入 `commands/run/log/samples.csv`（下次 run 会覆盖），命令行打印实际路径（脚本不接受额外 `--csv`）。`--warmup 5` 是每批之前的预热，`--iterations 100` 是每个 Event 区间操作数 K，`--repeats 21` 是独立采样次数 m；先看 CSV 的 `raw` 和 `median` 行，不要把单次波动误当作优化。若时间为零，尝试增加 `--iterations`；若报内存不足，减小 `--n`，勿擅自放大安全上限。

### 5. 只改一个变量，复算结果

**目的**：固定同一设备、构建配置、N、block、版本和预热/重复次数，仅改变 K，观察短操作批量测量的差异。每次 `run.py` 会生成独立证据目录；比较 CSV 的 `event_span_us_per_op`、`host_batch_us_per_op` 与 `host_e2e_us`，前两项按 K 平均，最后一项仍是一次完整传输+操作。

```bash
CUDA_ARCH=sm_89 python3 commands/run/run.py vector_add --variant scalar --n 4096 --block 256 --warmup 5 --repeats 21 --iterations 1
CUDA_ARCH=sm_89 python3 commands/run/run.py vector_add --variant scalar --n 4096 --block 256 --warmup 5 --repeats 21 --iterations 100
```

**预期**：两次都通过校验，但平均时间可能不同；不要预言哪个一定更快。若差异不稳定，先检查 CSV 中每次 raw 样本及环境/其他 GPU 负载，再重复；不要把 K 的变化解释成 kernel 算法变快。练习：保持 K=100，把 `--n 4096` 换成 `--n 1048576`，这时改变的是工作集，不要与 K 对照混为一谈。

## 回头解释：线程映射和带宽

grid-stride 版本将 grid 限定为 `min(ceil(N/block), SM数×grid_multiplier)`（SM 是流多处理器），每个线程按 `gridDim.x*blockDim.x` 步长循环。每轮相邻线程仍处理相邻元素，每个合法 i 恰好覆盖一次；但 block 数与循环开销都可能变化，所以它是**另一种任务分配假设**，不保证更快。Copy 的 `runtime` 版本使用同一 stream 的 `cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice)`；它完成相同复制，但 Runtime 内部实现不能由 API 名称推断，也不是只改变一条指令的公平 kernel 对照。先测 scalar，理解后再测其他版本。

| 量 | Vector Add | Memory Copy |
| --- | --- | --- |
| 逻辑请求字节 / 次 | `N×(4+4+4)=12N` | `N×(4+4)=8N` |
| 浮点运算 / 次 | N 次 FP32 加法 | 0 |
| 算术强度（按逻辑字节） | `1/12` FLOP/B | 0 |
| 活跃设备工作集 | 3 个 N 元素数组，12N B | 2 个数组，8N B |
| 初步预测 | 大数据低算术强度；小数据可能受提交/调度限制 | 大数据关注访存；小数据关注固定成本 |

N=8 时 Add 请求 96 B，Copy 请求 64 B；N=257 时分别为 3084 B、2056 B（这是手算，不是显存实际流量）。若 Event 批总时长为 `T_ms`、批内 K 次，则 `t_us=T_ms×1000/K`，`B_eff=Q/(t_us×1000)` GB/s，Q 为上表每次逻辑请求字节数。N=1,048,576 时 Add 是 12,582,912 B（12 MiB），Copy 是 8,388,608 B（8 MiB）。这些是**计算值，不是测试数据**；有效带宽也不是 DRAM 实测带宽。只有假设 Q 都经某一层且其带宽 B 已知，`Q/B` 才能作该层时间下界。Stage 1 未测 sector、cache hit、DRAM bytes；连续线程访问连续元素可从源码推断，但缓存结论留待后续实验。

## 进阶：多版本、工具和参数速查

完成前面步骤后，才尝试大工作集与不同版本；下列命令每次都重新建证据目录，注意固定其余条件再比较：

```bash
CUDA_ARCH=sm_89 python3 commands/run/run.py vector_add --n 1048576 --block 256 --warmup 5 --repeats 21 --iterations 100
CUDA_ARCH=sm_89 python3 commands/run/run.py memory_copy --n 1048576 --block 256 --warmup 5 --repeats 21 --iterations 100
CUDA_ARCH=sm_89 python3 commands/run/run.py vector_add --n 16777216
CUDA_ARCH=sm_89 python3 commands/run/run.py memory_copy --n 16777216
```

默认 `--variant all`：Add 运行 scalar/grid_stride，Copy 还包括 runtime。比较工作集时要显式固定版本、block、K 等；大输入若超出空闲显存的一半或 `--max-mib` 上限，会拒绝执行。

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--n` | 1048576 | FP32 元素数，必须 >0 |
| `--block` | 256 | 每 block 线程数；还检查设备/kernel 限制 |
| `--variant` | all | scalar / grid_stride；Copy 另有 runtime |
| `--grid-multiplier` | 4 | grid_stride 的 SM 数倍数 |
| `--warmup` | 5 | 每批测量前预热次数，至少 1 |
| `--iterations` | 100 | 每个 Event 区间内操作次数 K |
| `--repeats` | 21 | 独立采样次数 m；端到端也采样 m 次 |
| `--max-mib` | 256 | 设备数组总量上限，允许 1..1024 |
| `--device` | 0 | 选择单设备序号，不做多 GPU |
| `--csv` | 空 | 仅直接运行二进制时指定**不存在**的新路径；`run.py` 自动分配，禁止传入 |

下面是可选检查，不是第一次测量的前提；缺工具时脚本会记录 `Not Tested`，不应当成通过：

```bash
CUDA_ARCH=sm_89 python3 commands/static/run.py vector_add
CUDA_ARCH=sm_89 python3 commands/sanitizer/run.py vector_add
CUDA_ARCH=sm_89 python3 commands/nsys/run.py vector_add --n 1048576
CUDA_ARCH=sm_89 python3 commands/ncu/run.py memory_copy --n 16777216
```

`static.py` 用 `cuobjdump` 将 PTX（虚拟指令）、SASS（设备机器指令）和资源用量写入证据目录；先找 load/add/store，再看最终指令，不能用 PTX 条数直接推耗时。寄存器、shared memory（片上共享内存）与 occupancy API 的理论上限不是实际 occupancy。`sanitizer.py` 对 N=4099 的所选实验运行四种检查器，只覆盖此次输入和路径。`nsys.py` 采集时间线，用于区分 K 次操作活跃时长与间隔以及 H2D、D2H；`ncu.py` 先查询可用指标再采集基本指标。分析器插桩耗时不能当普通运行基线；缺少计数器就标记未测，不能从耗时断言 DRAM 流量。

## Questions：运行前先预测、运行后复盘

1. Vector Add 和 Copy 各读写多少字节？为什么不能只算输出？
2. host 的 launch 调用耗时和 kernel 完成耗时有什么区别？
3. block=256、N=257 时，哪些线程会访问数组？grid-stride 如何证明不漏写？
4. 预热、K 次批量和重复 m 个样本分别解决什么问题？不能解决什么？
5. 小工作集有效带宽高于某个 DRAM 标称值，可能有哪些解释？需要什么证据排除计量错误？
6. 如果 grid-stride 更慢，是不是循环一定是瓶颈？还需排查哪些变量？
7. 什么情况下不能把本机结果推广到别的 GPU、Toolkit、N 或输入驻留状态？

## Analysis 与验收

按 [分析记录](analysis.md) 对每组实验写预测、实测、指标、差异、硬件解释与边界。至少完成小/大工作集、K 批量对照、正确性边界、静态资源观察。计数器缺失时仍可完成测量实验，但不能把 cache/DRAM 假设写成事实。

验收：可重复的命令和 CSV；正确性通过；说清三种计时定义；手算一个有效带宽；列出负面结果与 Not Tested 项。独立完成 [练习](exercises.md) 后再读 [答案](answers.md)。


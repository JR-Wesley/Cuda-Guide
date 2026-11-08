# 实验 05：用向量读写理解 NCU 访存指标

## 本课要解决的问题

想象 32 位同学各取一本书：书挨着放、每隔八个位置放一本，或多人取同一本，经过的书架范围不同。GPU 上也如此。本课保持向量乘法每线程两读一写，用连续、偏移、跨步读、跨步写和只对 b 广播的五种地址模式，学习从**线程地址 → Request / Sector / Wavefront → Cache / DRAM 流量**逐层读懂 NCU 指标，不以寻找最快版本为主要目标。

源码决定线程取哪个数，地址分布决定一次请求覆盖哪些区段，缓存和并发再影响实际流量与时间。只看总耗时，无法知道中间哪一步起作用。本课先手算输出和地址，再扩大规模、对比分层请求。矩阵转置及其优化仍从[实验 06](../06_shared_memory_and_transpose/lab06.md)开始。

学完应能回答：stride=8 为什么仍计算 N 个输出？偏移 4B 为什么增加 sector，却不是非法 float 访问？只让 b 广播时，为什么整个 kernel 的 load sectors/request 不是 1？理论配读：[lec05 的第 2～5、7、12 节](lec05.md)。

必要前置知识：数组下标从 0 开始；本例 `float` 是 4 字节。CPU（host，主机）准备数据，GPU（device，设备）执行 kernel。block 是线程块，grid 是本次启动的全部块；warp 是块内一组 32 个线程，lane 是组内位置。L1、L2 是缓存，DRAM 是设备显存；请求可能在缓存得到满足，不一定每次都到显存。

**当前状态：** 五版本 kernel、本课 host、main 入口和 profile 完整过滤器均已实现，目前仅完成静态检查，尚未 CUDA 编译、运行或采集。下文输出与指标均为**预期或待填项，不是实测**；地址模型核对不等于运行正确性测试。只阅读和手算即可完成前两步，详细验证边界见第 5 节。

## 1. 沿着一次乘法读懂程序

这里的向量乘法不是点积：不把结果相加，而是每线程计算一个独立输出。先用 N=4 想象任务：a 的逻辑有效值为 `0, 0.125, 0.25, 0.375`，b 的有效值全为 2，四个逻辑输出均为 `0, 0.25, 0.5, 0.75`。改变的是这些值存放或读取的位置，不是乘法数量。

阅读[入口 kernels/main.cu](kernels/main.cu)，再读本课专用 [kernels/vector_access.cuh](kernels/vector_access.cuh)：它 include `../../kernels/vector_multiply.cuh`，复用共享 [vector_multiply.cuh](../kernels/vector_multiply.cuh) 的 `vector_mul<Stride>`，只在本课增加其余地址模式和轻量 host。当前接口与入口选择如下，不引入通用框架或自动扫描：

```cpp
enum class VectorAccess { contiguous, offset, strided_read, strided_write, broadcast };

template<VectorAccess Access, int Stride, int Offset>
int run_vector_access(int n, int block);

// kernels/main.cu 中的选择；每次只选一个版本，修改后显式重编。
constexpr auto access = VectorAccess::contiguous;
constexpr int n = 16, block = 256, stride = 8, offset = 1;
// main 中：
return run_vector_access<access, stride, offset>(n, block);
```

尖括号把选项作为编译期模板参数传给 host runner；runner 用 `if constexpr` 选择单个 kernel，不在 GPU 内做运行时版本分派。`stride` 只影响两种跨步版，`offset` 只影响偏移版；因此默认虽写 stride=8、offset=1，`contiguous` 仍是连续读写。不同模板实例的地址计算可能被不同地优化，未经编译与反汇编不能保证 SASS 形式，也不能声称“只影响内存行为”。

令 `i=blockIdx.x*blockDim.x+threadIdx.x`，仅 `i<n` 的线程工作；block=256 时第 1 块第 3 个线程的 i 为 259。五版均执行一次乘法，地址对应如下：

| `access` | 所选 kernel | 读取 a / b | 写入位置 `output_index` |
|---|---|---|---|
| `contiguous` | `vector_mul<1>` | `a[i]`, `b[i]` | `c[i]` |
| `offset` | `vector_mul_offset<Offset>` | `a[i+Offset]`, `b[i+Offset]` | `c[i]` |
| `strided_read` | `vector_mul<Stride>` | `a[i*Stride]`, `b[i*Stride]` | `c[i]` |
| `strided_write` | `vector_mul_strided_write<Stride>` | `a[i]`, `b[i]` | `c[i*Stride]` |
| `broadcast` | `vector_mul_broadcast` | `a[i]`, `b[(i/32)*32]` | `c[i]` |

host 按逻辑输入位置 p 初始化 `a[p]=(i%1024)*0.125f`、`b[p]=2.0f`，其中 offset 版 p=i+Offset，跨步读版 p=i*Stride，其他版 p=i。`i%1024` 是除以 1024 的余数。broadcast 的 b 仍分配 N 个元素并全部填 2，在本课 block=256 时每个 warp 只读其中首个元素；**只有 b 广播，a 仍连续读取**。这是 Global Memory 的同址重复读取，不是 warp 通信、shuffle 或 shared memory。`(i/32)*32` 按全局编号每 32 项分组；若 block 不是 32 的倍数，这种分组不再与硬件 warp 一一对应，不能直接套用本课的逐 warp 预测。

五版都应满足 `c[output_index]=(i%1024)*0.25f`。跨步写是 scatter（分散写出）：主机拷回完整输出 storage，但核对 N 个有效位置 `c[i*Stride]`，不把间隙当作逻辑输出，也不检查间隙值。输出缓冲区在计时前用 `0xff` 填充，便于有效位置漏写时核对失败。其余版核对 `c[i]`。初始化、拷贝、核对都在 kernel 计时之外；CUDA 错误检查、Event 计时和 RAII（自动管理设备缓冲区及事件生命周期）复用现有公共实现，计时口径不变。

**先预测：** i=3 时，连续版读 a[3]，offset=1 版读 a[4]，跨步读 stride=8 版读 a[24]，三处逻辑值均为 0.375。跨步写版读 a[3]、写 c[24]；广播版读 a[3]、b[0]。五版逻辑输出均为 0.75。N=16 时逻辑输出为 `0, 0.25, …, 3.75`，不是只剩两个元素。

本步不需命令。理解检查：i=15、stride=4 时，跨步读与跨步写分别访问哪里？前者读 a[60]/b[60]、写 c[15]；后者读 a[15]/b[15]、写 c[60]。若预测不一致，先分清逻辑 i、输入位置和输出位置，不要开始计时。

## 2. 手画每条 warp 访存指令的地址，而不是猜带宽

**Coalescing（合并访存）**是硬件把同一 warp、同一条访存指令中参与 lane 的地址组织成对若干内存区段的访问，不是修改数组。**Request** 是提交给某硬件单元的工作项，不是“一个线程取一个数”。**Sector** 是按 32B 边界划分的区段，如 `[0,31]`、`[32,63]`；编号为 `floor(字节地址/32)`，floor 表示向下取整。B 是字节，不是 bit（位）。

先只追踪 a 的一条 load（读取指令），以原数组基址为零点列相对字节偏移和 sector 编号；表中 `/` 分隔两个量，不是做除法：

| lane | 连续：字节偏移 / sector | offset=1：字节偏移 / sector | 跨步读 stride=8：字节偏移 / sector |
|---:|---|---|---|
| 0 | 0 / 0 | 4 / 0 | 0 / 0 |
| 1 | 4 / 0 | 8 / 0 | 32 / 1 |
| 7 | 28 / 0 | 32 / 1 | 224 / 7 |
| 8 | 32 / 1 | 36 / 1 | 256 / 8 |
| 31 | 124 / 3 | 128 / 4 | 992 / 31 |

假设各数组原始基址对齐 128B、每 lane 标量访问 4B、block=256、N 是 32 的倍数；对**单个满 warp**分别数 a load、b load、c store（写入指令）：

| 版本 | a load sectors | b load sectors | c store sectors | 两条 load 合计 sectors / requests 的典型预测 |
|---|---:|---:|---:|---:|
| contiguous | 4 | 4 | 4 | `(4+4)/2=4` |
| offset=1 | 5 | 5 | 4 | `(5+5)/2=5` |
| strided_read，stride=8 | 32 | 32 | 4 | `(32+32)/2=32` |
| strided_write，stride=8 | 4 | 4 | 32 | `(4+4)/2=4` |
| broadcast | 4 | 1 | 4 | `(4+1)/2=2.5` |

这是假设两条标量 load 各形成一个 L1TEX LSU request 的静态模型，不是已确认的机器指令或实测。**a、b 是两条 load，不能把 sector 合起来却只算一个 request。** broadcast 只有 b 是 1 sector，不能把整个 kernel 的 aggregate load sectors/request 写成 1。跨步写的区别在 store，不能从不变的 load 比值推出 store 合并良好。

跨步读中 stride=1/2/4/8 时，每条满 warp load 依次覆盖 4/8/16/32 个 sector；跨步写对 c 的单条 store 也可这样推算。stride≥8 后 sector 数不再无限增长，但 cache line 分布、地址翻译和请求处理成本仍可能变化。offset=1 是整体偏移一个 float（4B），仍满足 float 的自然对齐，不是未对齐 float 的非法访问；它改变的是 sector/cache line 边界覆盖。相邻 warp 的 offset 访问会重叠边缘 sector，缓存可能复用，**不能将每请求 5/4 的覆盖比例直接当成 DRAM 流量倍率**。

同一条指令只有未越界、真正参与的 lane 提供地址；这不是“仍驻留的 active warp”。**本步练习：** 分别手算 N=16（半 warp）、32（满 warp）、33（一个满 warp 加一个仅 lane 0 活跃的尾 warp）。以下每项为该 warp 的 a/b/c sector 数，不把尾 warp套进满 warp 公式：

| 版本 | N=16，首 warp | N=32，首 warp | N=33，尾 warp（i=32） |
|---|---|---|---|
| contiguous | 2/2/2 | 4/4/4 | 1/1/1 |
| offset=1 | 3/3/2 | 5/5/4 | 1/1/1 |
| strided_read，stride=8 | 16/16/2 | 32/32/4 | 1/1/1 |
| strided_write，stride=8 | 2/2/16 | 4/4/32 | 1/1/1 |
| broadcast | 2/1/2 | 4/1/4 | 1/1/1 |

N=33 的首 warp 与 N=32 相同；尾 warp 的 b 广播位置为 b[32]，不是全 grid 共读 b[0]。全 kernel 的平均值按实际 request 数汇总，不是对两行或不同 N 的比值机械取平均。也不能直接拿半 warp、广播、跨步写或每 lane 访问宽度不同的比值评优；先核对参与 lane、唯一地址、读写方向和分母。

再区分分配容量、lane 逻辑字节和硬件流量。runner 要求 N>0、Stride>0、Offset≥0；下表 storage 均以 float 元素计，分别对应程序输出的 `input_elements` 和 `output_elements`：

| 版本 | 每个输入的 `input_storage` | 输出 `output_storage` |
|---|---|---|
| contiguous | N | N |
| offset | Offset+N | N |
| strided_read | `(N-1)*Stride+1` | N |
| strided_write | N | `(N-1)*Stride+1` |
| broadcast | N | N |

每侧三数组的 `allocated_bytes=(2*input_storage+output_storage)*4`；CPU/GPU 各有一组，不把这个式子当作两侧合计。N=16、stride=8 时，跨步读分配 1032B，跨步写分配 612B，连续/广播分配 192B；offset=1 分配 200B。五版的 lane 逻辑字节均为 `12N`（每线程两次 4B 读、一次 4B 写），但广播的唯一输入数据更少：b 仅访问 `ceil(N/32)` 个 float。唯一输入量、分配足迹、sector 覆盖与 DRAM 字节不是同一个量。上传包含间隙，拷贝不在 kernel 计时内。

第 3.6 节的 float4/每 lane 16B 向量化访问仅作[可选阅读](lec05.md#36-情况-f向量化-global-load例如每-lane-读取-16b)，本实验不扩代码实现；本课的“向量乘法”不意味着使用向量化访存指令。

## 3. 单独编译最小例，再单独核对输出

**目的：** 先证明算对，再测快慢。下面均为 Bash 命令，工作目录为 `cuda/ncu/05_global_memory/`（相对工作区根目录为 `ncu/05_global_memory/`），不是要求在 PowerShell 中原样执行。这些编译、运行和采集步骤目前尚未执行。

执行前先确认环境：

```bash
nvidia-smi
nvcc --version
ncu --version
```

`nvidia-smi` 查看设备与驱动，`nvcc` 是 CUDA 编译器，`ncu` 是 Nsight Compute 单 kernel 性能分析器，`--version` 显示工具版本。第一条中显示的 CUDA 版本不是已安装 Toolkit 版本，编译器版本看第二条。另查设备 compute capability（计算能力，即架构版本），确认 Toolkit 支持它。下文 `sm_89` 只是架构示例，应替换为设备支持的目标。

读 [common.cuh](../kernels/common.cuh)：`CUDA_CHECK` 会报告 CUDA API 错误并退出；`time_kernel` 默认先预热 10 次，再用 CUDA Event（GPU 时间标记）测量 20 次启动的平均毫秒（`ms=10⁻³ s`）。预热让首次执行的影响不直接进入后续平均；Event 是插入 GPU 工作序列的时间标记，不是 CPU 秒表。

先以 contiguous、N=16 单独编译：

```bash
mkdir -p build log
nvcc -std=c++17 -O3 -lineinfo -arch=sm_89 kernels/main.cu -o build/lesson05
```

`./` 表示当前目录。`mkdir -p` 创建产物目录，已存在不报错；`-std=c++17` 选择语言版本，`-O3` 开启优化，`-lineinfo` 保留分析器需要的源码行信息，`-arch` 选目标架构，`-o` 指输出文件。编译失败应先读第一条错误，不要继续运行目录里的旧二进制。

然后才普通运行：

```bash
./build/lesson05
```

**看什么：** 预期 `PASS`、`checked=16`、`mismatches=0`，并打印小输入的有效结果；对照步骤 1 的手算。保持 n/block/stride/offset，只改 access，逐版重编后核对；跨步写应显示并检查实际 scatter 下标，五版的 N 个逻辑输出值相同，storage 按步骤 2 改变。核对是全量精确比较；本例数据可精确表示，不靠误差容限掩盖错位。失败先查所选模板、是否重编、输入初始化位置、输出索引和第一个错误下标。另用 N=32、33 检查满 warp 与尾部边界，每次改 N 后重编。

**如何理解时间：** `event_average_ms` 排除分配、CPU/GPU 拷贝和 CPU 核对，可能含提交间隙。N=16/32/33 仅用于理解与核对，不能据此宣称带宽饱和。改常量或头文件后都需显式重编。

## 4. 固定有效工作量，逐个选择地址模式

**目的：** 用同一个两读一写的向量乘法检验地址预测；时间仅作辅助观察，不要求获得加速。以下是对照步骤；当前仅有静态预测，观测值保持未测量。

在 `kernels/main.cu` 先单独改 `n=262144`，保留 block=256、stride=8、offset=1、access=contiguous，按步骤 3 重编并多次普通运行，建立此规模的新基线。随后只改 access，依次对照 offset、strided_read、strided_write、broadcast，每次显式重编、核对并独立运行数次。不自动扫描，不拿 N=16 与新规模算加速比。若要研究 stride=1/2/4/8，另固定 access=strided_read（或 strided_write）做单独一组，每次只改 stride；offset 也另开单变量对照。

五版 N 次乘法、12N lane 逻辑字节和逻辑输出相同，但输入唯一数据、分配足迹与请求布局不同。N=262144、stride=8 时，跨步读每侧三数组约 17 MiB，跨步写约 10 MiB；`MiB=2²⁰ B`，不同于 `MB=10⁶ B`。缓存容纳关系和可能的指令生成差异都属于影响因素，本实验没有隔离“只有 sector 数变化”。先查目标设备 L2 容量，不预称此规模“大于 L2”。

普通核对通过后，在同一工作目录先查看工具支持：

```bash
ncu --list-sections
ncu --query-metrics
```

NCU 分析单个 kernel；Section 是相关指标组，metric 是具体计数项。以下以 contiguous 报告为例，过滤全部候选函数基名，但程序只启动编译时选中的一个版本：

```bash
ncu --section LaunchStats --section SpeedOfLight --section MemoryWorkloadAnalysis \
    --kernel-name-base function -k 'regex:^vector_mul(_offset|_strided_write|_broadcast)?$' \
    -s 10 -c 1 --replay-mode kernel --cache-control all --clock-control base \
    -o log/vector_access_contiguous.ncu-rep ./build/lesson05
ncu --import log/vector_access_contiguous.ncu-rep --page details
```

`--section` 选择指标组，`-k` 过滤 kernel，`function` 使用不含模板参数的函数基名；正则的 `^`/`$` 限定完整名字，候选为 vector_mul、vector_mul_offset、vector_mul_strided_write、vector_mul_broadcast。contiguous 和 strided_read 都叫 vector_mul，须同时核对入口配置与报告的完整模板符号，不只看基名。反斜杠 `\` 是 Bash 续行符；`-s 10 -c 1` 跳过 10 次匹配启动，仅采随后一次，与默认预热对应。

`--replay-mode kernel` 允许重放 kernel 获取指标，`--cache-control all` 请求在重放采集时清缓存，`--clock-control base` 请求基础时钟控制；实际支持与权限看工具输出。`-o` 指报告路径，`--import` 读取报告，`--page details` 展示详细页。每次修改并重编后，将命令中的导出、导入名一起改为 `vector_access_offset_o1.ncu-rep`、`vector_access_strided_read_s8.ncu-rep`、`vector_access_strided_write_s8.ncu-rep`、`vector_access_broadcast.ncu-rep`，都放本课 log。另做 stride 组时在报告名中写 s1/s2/s4/s8，不覆盖基线或已引用证据。

**L1TEX** 是 SM（执行线程块的硬件单元）附近处理 Global/Local 等访问的路径，不只是 L1 缓存容量。load requests 是读取工作项数，sectors 是这些工作触及的 32B 区段次数；相同地址被不同请求访问可以重复计数，不能拿单 warp 的 4/32 个 sector 当全 kernel 总数。**Wavefront** 是 L1TEX 内部一轮能处理的工作包，不是 grid 的 Wave 容量，也不是 load 的延迟周期。**Transaction（事务）**泛指某层处理或传输，必须先确认层次，不能等同于 Request 或 DRAM 传输。

按以下顺序读，不先找最高百分比：

| 看什么 | 如何解释 | 不符合预期先查什么 |
|---|---|---|
| LaunchStats 的 grid/block、函数与模板配置 | 确认实际采到了这一版 | 二进制是否更新、过滤器是否命中；vector_mul 对应连续还是跨步读 |
| L1TEX load requests/sectors/wavefronts | 对照 a、b 两条 load 的预测及聚合分母 | 活跃 lane、指令宽度、广播仅作用于 b；不把 wavefront 当延迟 |
| L1TEX store requests/sectors/wavefronts | 对照 c 连续写或 scatter | 跨步写不能靠 load 比值解释；读写指标不可混用 |
| L1/L2 命中与流量、DRAM 读写字节 | 判断请求在哪层满足 | 缺失指标不是零；不同层 request 分母不可互换；offset 重叠 sector 可复用 |
| SpeedOfLight 与持续时间 | 提出资源压力假设 | 吞吐百分比不是时间占比，高百分比不独立证明瓶颈 |

下面是对照记录模板，**预测不是实测**。普通运行将来记录多次 Event 的中位数（排序后取中间值）和波动；NCU 栏只填实际可用指标，并注明单位、层次和读写口径：

| 配置（N=262144，block=256） | 满 warp 预测 a/b/c sectors | 预测 load sectors/req | 正确性 / 普通 Event 中位数与波动 | NCU load/store、Wavefront、L1/L2、DRAM、Duration | 差异解释与反例 |
|---|---|---:|---|---|---|
| contiguous | 4/4/4 | 4 | 未测量 | 未测量 | 待填 |
| offset=1 | 5/5/4 | 5 | 未测量 | 未测量 | 待填 |
| strided_read，stride=8 | 32/32/4 | 32 | 未测量 | 未测量 | 待填 |
| strided_write，stride=8 | 4/4/32 | 4 | 未测量 | 未测量 | 待填 |
| broadcast | 4/1/4 | 2.5 | 未测量 | 未测量 | 待填 |

可以另行追加 `WarpStateStats` 研究等待，但 Long Scoreboard（等待 L1TEX 路径相关结果，例如乘法等输入）高不等于 DRAM 带宽已满。NCU 的清缓存/replay 条件与普通重复运行不同；NCU 下打印的 Event 时间不作为独立性能基线。没有报告先读工具首个错误，不修改系统权限来“凑结果”。

## 5. 总结、验收与记录边界

不用“访存差所以慢”作为结论。请给出三项可核对的解释：① 五版的地址公式、N=16/32/33 的参与 lane 与 sector 推导；② lane 逻辑字节、唯一输入、分配足迹和硬件流量的区别；③ 固定 N 的预测与待测对照表，明确广播 load 聚合为典型 2.5、跨步写需看 store、offset 覆盖不能直接换算 DRAM 倍率。未测量的栏写“未测量”，不填零，也不编造 PASS 或加速。如果未来 sector 增加而时间不变，保留结果并检查缓存、并发和请求吞吐，不直接宣布预测失效。

**实现与验证边界：** 五版本 kernel、`run_vector_access`、main 常量和 profile 完整正则均已落地，源码静态审阅与文档一致。已完成静态接口、表达式、过滤器及地址模型核对；实现阶段已报告本课三个脚本 `bash -n` PASS，本次文档修订未重跑脚本。满 warp、半 warp 和 N=33 尾部的 sector 表只是地址模型结果，**不是 GPU 运行正确性测试**。当前尚未 CUDA 编译、运行、采集 NCU 或检查 SASS，无新报告、日志。[编译日志](log/compile.log)、[运行日志](log/run.log)及[旧 compile-only 日志](log/legacy_commands/compile/source_check.log)原样保留，仅属历史版本，不验证本课五版本入口。

理解底层命令后，可在本课目录分别使用以下三个独立入口；当前尚未执行这些实验动作：

```bash
CUDA_ARCH=sm_89 bash commands/compile.sh
bash commands/run.sh
bash commands/profile.sh
```

三个脚本均无位置参数、按自身位置定位；仅 compile 必填 CUDA_ARCH，按设备替换。compile 固定 C++17、`-O3 -lineinfo`，生成 build/lesson05，编译前移除旧目标，失败即停止；run/profile 缺产物报错，不隐式构建，改常量/头文件后须重编。profile 已使用 `--kernel-name-base function -k 'regex:^vector_mul(_offset|_strided_write|_broadcast)?$'`，采 LaunchStats、SpeedOfLight、MemoryWorkloadAnalysis；只采所选单版本，不扫描，保留 skip 10/count 1 与 kernel replay/cache all/clock base。

已有动作的日志路径为 `log/compile.log`、`log/run.log`、`log/profile.log`，含展开命令、输出与 exit_code；重跑覆盖各自日志，保留独立对照须先将日志和报告另存为有意义的配置/运行名。脚本报告 `log/profile.ncu-rep` 已存在则拒绝覆盖，NCU 失败即停止，报告内容由实验者检查。旧 `log/legacy_commands/<原相对路径>` 的原始命令和证据不改写，也不作为当前五版本的运行验证。

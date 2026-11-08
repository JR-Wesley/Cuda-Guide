# 深度解读：Using CUDA Warp-Level Primitives

## 阅读目标与资料来源

本文基于用户提供的解读材料，结合 NVIDIA 博客正文与 CUDA Programming Guide 整理，不是原文逐字翻译。核心问题不是记住几个 `__shfl_sync()` API，而是理解：**warp 内线程协作也必须有明确的参与者、数据来源和同步协议。**

主要来源是 Yuan Lin、Vinod Grover 于 **2018 年 1 月 15 日**发表的 [Using CUDA Warp-Level Primitives][blog]。文中的 Listing 编号均指该博客；对原例的修正、现代整数归约补充和练习属于本文分析，并非原作者结论的直接引用。API 约束依据 [CUDA Programming Guide — Warp Functions][warp] 和 [Warp Synchronization Function][sync]；本次核对的在线指南显示版本为 **v13.5**，访问日期为 **2026-11-10**。在线文档会更新，阅读历史博客时应同时核对所用 Toolkit 的文档。

**证据边界：本次只新增阅读分析，未编译本文片段、未运行 GPU、未测性能。** 文中数字是手算示例或预期值，不是实验日志。SM89（compute capability 8.9）仅作为架构适用性讨论，不表示已确认当前设备。已有程序、动作及历史证据仍以 [专题 README](README.md)、[动作索引](commands/README.md) 和 [RESULTS.md](RESULTS.md) 为准。

建议按以下顺序阅读：执行模型 → 寄存器交换 → mask → 内存同步 → 动态分组 → 指令与实验。首次阅读先关注单个完整 warp；除特别说明外，示例假设一维 block、32 个线程、没有提前退出。

## 一、Warp、SIMT 与 Independent Thread Scheduling

### 1. 32 个线程有独立数据，不等于独立的标量发射通路

Warp 是 NVIDIA GPU 组织线程执行的基本单位；这里一个 warp 有 32 个 lane，lane 是线程在 warp 内的编号。SIMT（Single Instruction, Multiple Thread，单指令多线程）允许线程执行共同的指令，同时各自持有寄存器值、访问地址和控制流状态。例如：

```cpp
__global__ void scale(const float* a, float* b) {
    int tid = threadIdx.x;
    float x = a[tid];
    b[tid] = x * 2.0f;
}
```

在 `<<<1, 32>>>` 下，lane 0 访问 `a[0]`，lane 1 访问 `a[1]`，以此类推。调度器选择可执行的 warp，参与当前指令的 lane 使用各自操作数执行。不能由“每个线程有独立状态”推导出“每个线程都有一套独立的标量指令发射通路”，也不能由“共同执行指令”推导出所有线程永远同步前进。[1]

### 2. 分歧与独立线程调度

```cpp
if (lane < 16) {
    // lane 0～15 的工作
} else {
    // lane 16～31 的工作
}
```

分支让同一 warp 的线程执行不同路径，当前指令的参与集合随之变化。Volta 之前使用共享程序计数器与 active mask 管理 warp 执行；Volta 引入 Independent Thread Scheduling，能维护更细粒度的线程状态，并以子 warp 粒度进行分歧与重新汇合。[1]

这不保证每个 lane 可以不受约束地独立执行任意协议，而是要求程序不能依赖隐式 lockstep（线程始终锁步前进）。下面的 shared memory 通信缺少跨线程同步：

```cpp
smem[lane] = value;
int x = smem[(lane + 1) & 31]; // 错误示意：另一线程的写入未与读取建立同步
```

即使某次运行结果正确，也不能证明它没有数据竞争。编译器优化与调度都可能暴露问题；`volatile` 不能补上缺失的同步关系。

## 二、三类原语与 Shuffle 数据流

博客按“同步式数据交换、active mask 查询、线程同步”分成三类。细分到用途如下：[1][2]

| 用途 | API | 返回什么或完成什么 |
| --- | --- | --- |
| Vote：汇总判断 | `__ballot_sync`、`__any_sync`、`__all_sync` | 条件位图、是否有人满足、是否全员满足 |
| Shuffle：寄存器交换 | `__shfl_sync`、`__shfl_up/down/xor_sync` | 另一 lane 提供的值 |
| Match：同值分组 | `__match_any_sync`、`__match_all_sync` | 按位相同值的成员集合；不是一般浮点数值相等比较 |
| Active mask 查询 | `__activemask` | 此刻执行集合的快照 |
| Warp 同步 | `__syncwarp` | 参与线程的协调与内存顺序 |
| 整数归约（现代补充） | `__reduce_*_sync` | 成员集合的整数归约结果，要求 CC ≥ 8.0 |

### 1. 四种 Shuffle 的寻址方式

默认 `width=32` 时，设 lane `i` 持有 `x[i]`：

| API | 来源 lane | 常见用途 |
| --- | --- | --- |
| `__shfl_sync(mask, x, src)` | `src` 对应的 lane | 广播 |
| `__shfl_up_sync(mask, x, delta)` | `i - delta` | 前缀和（scan） |
| `__shfl_down_sync(mask, x, delta)` | `i + delta` | 归约（reduction） |
| `__shfl_xor_sync(mask, x, bits)` | `i ^ bits` | 蝶形交换 |

这里的“来源”还需满足边界和参与条件。例如全部 32 个 lane 执行 `__shfl_sync(0xffffffffu, lane, 0)`，都得到 0；执行 `__shfl_down_sync(0xffffffffu, lane, 4)`，lane 0～27 得到 4～31，lane 28～31 因来源越过 width 边界得到自己的值。[2]

**越界返回自身与读取未参与线程不是一回事。** 来源在寻址范围内、却没有参加调用时，取回的值未定义。mask 不会自动给这个来源补零。

Shuffle 不需要先写 shared memory 再读回来，适合细粒度跨 lane 通信；但它不是通用内存屏障，不负责发布其他 shared/global memory 写入。

### 2. Listing 1：五轮归约为什么得到总和

```cpp
for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffffu, val, offset);
}
```

假设所有 lane 参与，初值是 `x0` 到 `x31`，所有中间加法均在类型可表示范围内。lane 0 的数据流是：

| offset | lane 0 更新后的内容 |
| --- | --- |
| 初始 | `x0` |
| 16 | `x0 + x16` |
| 8 | `x0 + x8 + x16 + x24` |
| 4 | 累加 8 个元素 |
| 2 | 累加 16 个元素 |
| 1 | 累加全部 32 个元素 |

每轮读的是来源 lane 在该轮提供的部分和，不是它最初的值。若初值为 lane 编号，lane 0 的预期结果是 `0 + ... + 31 = 496`。其他 lane 不保证持有完整总和。

若所有 lane 都需要总和，可以使用完整 warp 的 XOR 蝶形归约：依次用 `1、2、4、8、16` 作为 XOR 参数，每轮交换互补部分和。浮点数的加法顺序会改变舍入结果，不能默认与 CPU 串行求和逐位相同。

五个数据交换阶段不等于五个时钟周期。交换、加法与阶段间依赖都影响延迟；是否比 shared memory 或 CUB 更快，需要相同条件下实测。

## 三、Mask：参与协议，不是过滤器

### 1. Membership mask 与 active mask

`0x0000ffffu` 的低 16 位为 1，对应 lane 0～15。Membership mask 表示算法指定的协作集合；active mask 表示某一时刻实际一起执行的集合。两者可能不同。

对于一次 `_sync` collective，应检查：[2]

1. 每个调用线程自己的 bit 都在 mask 中。
2. mask 命名的所有未退出线程按调用契约到达相应操作，使用相同 mask；退出线程被忽略，但不能再作为 shuffle 数据来源。
3. 并发的独立集合应使用互不重叠的 mask，不能让重叠集合对同一操作形成不同协议。
4. 循环、分支与提前返回不能让参与者错配调用轮次。

mask 不会替调用者执行 `if`，也不会把稀疏集合压缩为连续 lane 编号。成员为 `{0,3,7}` 时，lane 0 的 `shfl_down(...,1)` 请求的是 lane 1，不是“下一位成员”lane 3。

### 2. Listing 3：先 Ballot，再分支

```cpp
unsigned members = __ballot_sync(0xffffffffu, lane < 20);
if (lane < 20) {
    // 使用 members 执行满足其余约束的 collective
}
```

所有 32 个 lane 都执行 ballot，得到 `0x000fffffu`，其中低 20 位表示条件为真的线程。Ballot 的输入 mask 是投票集合，输出 mask 是投票集合中条件成立的子集，二者不要混淆。

为什么不在分支内部使用 `__activemask()`？因为“都会进入这个分支”不保证“同时到达这次查询”。查询可能只看到满足条件线程中的一部分。对必须累加全部 20 个元素的算法，遗漏成员就会得到错误的部分和。这是博客 Listing 5 的主要问题，不是说任何使用 active mask 的算法都错误。[1][2]

### 3. Listing 3 还缺少来源合法性条件

博客给出的部分 warp 归约在获得 mask 后，仍执行固定 `16、8、4、2、1` 五轮 down shuffle。**本文依据 shuffle 来源约束推导：它不能作为任意元素数的正确归约模板。**

以只有 lane 0～19 参与为例，第一轮 offset=16 时，lane 4 请求 lane 20，而 lane 20 没参加调用；这是“范围内但来源不参与”，不是“超过 width=32”。该未定义值可能沿后续数据流传到 lane 0。因此，“正确算出 mask”只是必要条件，并非充分条件。[1][2]

最容易检查的修正是：保留完整 32 个线程，无效输入以加法单位元 0 填充。下面是本文教学改写，不是博客原代码，也不是完整 host 程序。

```cpp
// 契约：launch <<<1, 32>>>，0 <= n <= 32；input 至少含 n 个元素，
// output 至少含一个元素；无别名写入竞争；所有中间和不发生有符号溢出。
__global__ void warp_sum(const int* input, int* output, int n) {
    constexpr unsigned FULL = 0xffffffffu;
    int lane = threadIdx.x;
    int val = lane < n ? input[lane] : 0;
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL, val, offset);
    }
    if (lane == 0) output[0] = val;
}
```

条件表达式仅在 `lane < n` 时读取输入。以 `input[i]=i+1、n=20` 为例，lane 20～31 提供 0，lane 0 预期输出 210；`n=0` 时输出 0。注意这是完整 warp 的“部分有效输入”，不是只启动 20 个线程的“物理不完整 warp”。不能把 launch 改成 `<<<1,20>>>` 后继续套用这份证明。

### 4. 现代补充：整数原生归约

CC ≥ 8.0 支持以下做法，SM89 满足架构门槛；仍须使用支持该 API 与目标架构的编译器。[2]

```cpp
// 契约：launch <<<1, 32>>>，1 <= n <= 32；输入与输出有效，求和不溢出。
__global__ void warp_sum_int(const int* input, int* output, int n) {
    int lane = threadIdx.x;
    unsigned members = __ballot_sync(0xffffffffu, lane < n);
    if (lane < n) {
        int sum = __reduce_add_sync(members, input[lane]);
        if (lane == 0) output[0] = sum;
    }
}
```

这里归约直接针对成员集合，结果返回给每个参与者，不要求手写连续来源的 shuffle 树。它也适用于合法的稀疏集合，但输出 leader 需确实属于该集合。这个示例不处理 `n=0`：没有参与线程就没人写输出。

`__reduce_add_sync` 不是任意类型的泛型归约；这里讨论 32-bit `int/unsigned`，不能直接拿来替代 FP32 求和。浮点归约可考虑 shuffle 或 CUB，并遵守具体接口契约。

## 四、`__syncwarp()`：协调线程与建立内存顺序

### 1. `_sync` 后缀并不意味着相同的内存语义

| 操作 | 协调指定参与线程 | 为参与者建立内存顺序 |
| --- | --- | --- |
| shuffle / vote / match / reduce | 是 | 否 |
| `__syncwarp(mask)` | 是，单个 warp 内 | 是 |
| `__syncthreads()` | 是，block 内 | 是 |

指南用 strongly happens before 描述 `__syncwarp(mask)` 调用与参与线程解除等待或退出之间的关系。对入门读者，先理解具体用途：生产者在同步前写入，消费者在匹配同步后读取，从而建立跨线程访问顺序。[3]

这与 acquire/release 一样需要推导跨线程顺序，但不能把 warp barrier 简化成“只在当前线程执行一次 fence”。Fence 本身不等待其他线程到达；warp 同步也不是向其他 warp、CPU 或其他 GPU 发布数据的通用协议。背景见 [Memory Model](../01_memory_model/README.md)。

### 2. Listing 6：4×8 数据重新分配

```cpp
// 一个完整 warp，共享数组只归这个 warp 使用
__shared__ float smem[4][8];
int x1 = lane % 8;
int y1 = lane / 8;
smem[y1][x1] = val;
__syncwarp(0xffffffffu);
int x2 = lane / 4;
int y2 = lane % 4;
val = smem[y2][x2];
```

原先 lane `i` 写行优先位置 `[i/8][i%8]`，之后 lane `i` 读取 `[i%4][i/4]`，即来源 lane `8*(i%4)+i/4`。所以 lane 0、1、2、3 分别读到原 lane 0、8、16、24 的值。这里改变的是元素在线程之间的归属，并没有在内存中生成新的 8×4 数组。[1]

多个 warp 不能共用这一个数组而各自只执行 warp barrier；应分配独立区域，或设计覆盖全部生产者和消费者的 block 同步协议。二维/三维 block 的 lane 应从线性线程编号计算，不能总用 `threadIdx.x & 31`。

### 3. Listing 7–8：为什么阶段末尾有 Barrier 仍会竞争

博客错误片段的形式是：

```cpp
shmem[tid] += shmem[tid + 16];
__syncwarp();
```

即使假设所有下标都合法且数据已初始化，lane 0 读取 `shmem[16]` 与 lane 16 覆盖 `shmem[16]` 仍可能发生在同一个未被同步分隔的阶段。表达式含读、计算、写；阶段末的 barrier 不能修复阶段内部已发生的竞争。不能直接将这段代码放进只有 32 个元素的数组：上半 warp 的 `tid+16` 还会越界。

Listing 8 的启发是把读取与覆盖分开，而不是可不检查索引、初值就复制它作为完整归约。下面以循环邻居交换说明两个同步点各自保护什么：

```cpp
// launch <<<1, 32>>> 的 kernel 内片段
__shared__ int smem[32];
int lane = threadIdx.x;
smem[lane] = lane;
__syncwarp();
int x = smem[(lane + 1) & 31];
__syncwarp();
smem[lane] = x;
```

第一个 barrier 处理 RAW（Read After Write，先写后读）依赖：确保读到已初始化的值。第二个处理 WAR（Write After Read，先读后覆盖）依赖：确保所有消费者读完旧值，才能覆盖。若还要继续跨线程读取最后一行的新值，必须再建立下一阶段的同步。

因此同步检查要围绕访问依赖，而不是只数 barrier 的数量。对 shared/global memory 的有效保证只涉及该 warp 中正确参与协议的线程，不覆盖外部访问者。[1][3]

## 五、Opportunistic Programming：允许临时分组的算法

### 1. Listing 9：把同地址 Atomic 合并

假设当前一起执行的 8 个 lane 目标地址分别为：

| lane | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 地址 | A | B | A | C | B | A | C | A |

按地址分组得到 `{0,2,5,7}`、`{1,4}`、`{3,6}`。如果这些 lane 属于同一次检测集合，每组一个 leader 执行 `atomicAdd(ptr, group_size)`，可把 8 次逻辑增量合成 3 次原子更新。这是算法计数，不是本次测得的动态指令数。[1]

下面根据博客改写，显式初始化局部参数，并使用无符号位移：

```cpp
// 用于同一设备上的有效 int 计数器；无溢出；每个调用者只申请一个编号。
__device__ int atomic_aggregate_inc(int* ptr) {
    unsigned t = threadIdx.x + blockDim.x *
                 (threadIdx.y + blockDim.y * threadIdx.z);
    unsigned lane = t & 31u;
    unsigned active = __activemask();
    unsigned group = __match_any_sync(
        active, reinterpret_cast<unsigned long long>(ptr));
    int leader = __ffs(group) - 1;
    int base = 0;
    if (lane == static_cast<unsigned>(leader)) {
        base = atomicAdd(ptr, __popc(group));
    }
    base = __shfl_sync(group, base, leader);
    unsigned lower = (1u << lane) - 1u;
    return base + __popc(group & lower);
}
```

`__ffs` 返回最低置位 bit 的一基编号，因此减 1 得到 leader lane；`__popc` 计算置位 bit 数。`group & lower` 只保留比当前 lane 编号低的组员，所以返回值是“组的起始编号 + 自己在组内的排名”。例如 A 组获得 base=100，则 lane 0、2、5、7 分别得到 100、101、102、103。

### 2. 为什么这里可以用 Active Mask

完整归约要求预定元素一个不少；编号分配允许当前一起执行的线程临时组合。即使 A 组分两批执行，只要每个线程恰好申请一次，总增量仍为 4，分配的编号仍可唯一。因此调度可能影响分组大小和效率，但不必影响最终计数正确性。所有临时组员仍须到达匹配的 collective；快照不是免除 mask 契约的许可证。[1][2]

这不是任意原子协议的等价替换：它不保留逐线程原子操作的具体执行顺序，中间计数状态也会改变；不能直接替换依赖这些状态的锁、完成计数或发布协议。`atomicAdd` 的原子性和 shuffle 的寄存器广播，不等于把其他线程写出的 payload 发布给消费者。

### 3. 对通信 Kernel 的启发

例如多个 token 发往同一 expert，可按 expert ID 或实际计数器地址匹配，由 leader 申请连续空间，再按组内 rank 分配位置。这里仅是设计思路，不是对 DeepEP 等实现的源码结论。

是否值得取决于同地址重复率、竞争程度、match/shuffle 开销及后续 scatter 成本。每个 lane 地址都不同，通常没有可合并的更新；编译器也可能已对朴素代码做聚合。因此必须检查基线的实际 PTX/SASS 并实测，不能用“原子数理论减少”代替性能证据。

## 六、同步后为什么仍不能假设 Lockstep

Listing 10–14 说明，`__syncwarp()` 保证同步点的协调，不保证之后所有线程永久同时执行。所以不能要求下面第二次查询仍为 full mask：[1]

```cpp
__syncwarp(0xffffffffu);
unsigned snapshot = __activemask(); // 不保证等于 0xffffffffu
```

同理，旧式 `__shfl` 或 `__ballot` 前后加 `__syncwarp()`，并不等价于 `_sync` 版本。旧接口依赖调用时的隐式执行状态；barrier 不把其间所有指令变成同步式 collective。旧式接口自 CUDA 9 起弃用，不应作为现代实现基础。

**历史博客与当前指南的措辞差异：** 博客 Listing 4、13 明确展示 Volta 上两个分歧分支使用相同 full mask 的协作；本次核对的 v13.5 指南约束则写有 “same program point” 及 mask 内条件一致等要求。[1][2] 不应把博客例子泛化成“任何分支里都可以随便用 full mask”，也不把这两处不同表述假装成毫无差异。

本教程采用便于审核的写法：分支内计算局部值，再到共同程序位置进行 collective；如果两支确实独立协作，就使用互不重叠的成员集合。需要依赖跨分歧调用点的具体行为时，应另行核对目标 Toolkit 的 API 与 PTX 契约，本次不提供该路径的编译或运行证明。

## 七、现代架构与指令层阅读

### 1. CUDA API 到 PTX 的阅读索引

以下是理解指令族的索引，不是本文编译得到的 PTX/SASS；具体 lowering 受类型、目标架构和编译器影响。64-bit 值可能需要多条 32-bit shuffle，不能按一次 API 调用数机器指令。[4]

| CUDA API | 可查阅的 PTX 指令形式 |
| --- | --- |
| `__shfl_sync` | `shfl.sync.idx.b32` |
| `__shfl_up_sync` / `__shfl_down_sync` | `shfl.sync.up.b32` / `shfl.sync.down.b32` |
| `__shfl_xor_sync` | `shfl.sync.bfly.b32` |
| `__ballot_sync` | `vote.sync.ballot.b32` |
| `__match_any_sync` | `match.any.sync.b32` / `.b64` |
| `__activemask` | `activemask.b32` |
| `__syncwarp` | `bar.warp.sync` |
| `__reduce_add_sync` | `redux.sync.add.s32` / `.u32` |

PTX 是虚拟指令集，SASS 是目标 GPU 的机器指令。阅读时分别检查 membermask、寻址规则、目标架构限制与内存顺序条款，不能由助记符里的 `sync` 推断通用 fence。

### 2. SM89 上的选择与边界

SM89 满足本文 match（CC ≥ 7.0）和整数 reduce（CC ≥ 8.0）的架构门槛。广播可考虑 shuffle；FP32 归约可考虑 shuffle/CUB；INT32 可比较 native reduce 与 CUB；共享内存交换必须有覆盖实际依赖的同步。CUB 和 Cooperative Groups 提供更高层抽象，但并不自动支持任意稀疏 mask 或任意线程数，仍需选择合适的 group、接口和临时存储布局。[2][5]

`width` 与 mask 是不同参数，例如：

```cpp
int next = __shfl_down_sync(0xffffffffu, val, 1, 8);
```

全部 32 个 lane 参与，但 down 寻址按连续 8-lane 子组进行，lane 7、15、23、31 返回自身值。width 仅可为 1、2、4、8、16、32，不能用 width=20 解决 20 元素归约。

**不要把 down 的子组规则推广到所有 shuffle。** `__shfl_sync` 的 srcLane 是子组逻辑编号；XOR 模式允许访问更早的组，但试图访问更晚的组时返回自身。例如 width=8、XOR 参数为 8 时，lane 8 可以读 lane 0，而 lane 0 返回自身值。具体规则见指南 shuffle 小节及本专题 [README](README.md)。[2]

博客末尾用 `-arch=compute_60 -code=sm_70` 选择 Pascal 调度兼容行为的建议属于当时的 Volta 排障背景，不是 SM89 的通用修复命令。正确修复是补足协议，而不是依赖旧调度方式。

## 八、从阅读到实验：四个问题，而非四个已完成结果

下面是后续实验建议，本次没有新增 kernel、动作脚本或运行记录。已有功能覆盖与命令应从 [专题 README](README.md) 和 [动作索引](commands/README.md) 进入，不能将已有程序当作本文所有新示例都已实现。

| 实验 | 单变量对照与检查 | 与已有专题的关系 |
| --- | --- | --- |
| E1：完整 warp 归约 | 同输入比较 down、XOR、整数 reduce；检查结果在哪些 lane 有效 | 已有 shuffle/reduce/CUB 教学入口；本文算法间性能对照待实现 |
| E2：部分输入 | 固定完整 warp，改变 n=1、7、16、20、31、32；再单独研究物理尾 warp、稀疏 mask | 已有边界和稀疏集合测试；不要混淆参与集合与来源规则 |
| E3：共享内存交接 | 固定索引与输入，每次仅移除一个 barrier，分析 RAW/WAR | 已有安全交换路径；故意错误变体应作为隔离的负例 |
| E4：聚合原子 | 保持总更新量，改变同地址重复率；核对计数和编号唯一性 | 本文提出的扩展，不宣称已有聚合原子 benchmark |

先手算，再检查 CPU reference。E2 中 `input[i]=i+1、n=20` 应为 210；E3 读邻居后预期 lane 31 得到 0；E4 还应核对每个地址分配区间不重复、不漏号。错误程序不保证每次都失败；不得为观察错误而删除生产代码的必要检查。

运行时依次分清三层证据：

1. **功能与安全**：完整 host 程序需要有效内存、launch 参数检查、CUDA API/launch/异步错误检查与 CPU 对照。Racecheck 用于共享内存访问竞争，synccheck 检查支持范围内的同步错误；工具未报错不是 mask 或 shuffle 来源合法性的证明。[6]
2. **静态指令**：显式编译后用 `cuobjdump` / `nvdisasm` 查看目标 kernel，核对 shuffle/redux/barrier 与依赖链是否存在。先看底层输入输出，再使用现有 static 动作；教程见 [二进制工具](../../../cuda-sdk-lab/experiments/binary/README.md)。
3. **性能**：正确性通过后再使用 CUDA events 或 Nsight Compute；先定义测的是单条依赖链延迟、多 warp 吞吐还是端到端时间。单次 32 线程 kernel 的端到端时间容易被启动开销掩盖。使用预热、重复测量与相同输入/配置，分析器重放时间与普通运行时间分开。

正式实验遵循 [实验规范](../../EXPERIMENT_PROTOCOL.md)：先确认设备与 Toolkit，显式构建，保留基线和每次对照的原始证据。现有动作可能覆盖固定日志，复测前先保护已被报告引用的记录。本次没有提供虚构的 `./warp_lab` 可执行文件或新的测量数值。

### 阅读后的预测练习

- 把完整 warp 的 n=20 填零归约改成只启动 20 个线程，哪些来源变得无效？为什么单改 mask 不够？
- 稀疏 mask 为 `{0,3,7}` 时，如何让每个成员都取得 lane 3 的值？哪些 lane 必须到达调用？
- 删去邻居交换中的第二个 barrier，画出“一个线程覆盖、另一个尚未读取”的交错顺序。
- 原子聚合中相同地址的线程分成两批时，哪些性质保持，哪些编号顺序或效率可能变化？

参考判断：第一题要重新证明来源合法性；第二题所有成员以相同 mask 执行 `__shfl_sync(mask, value, 3)`；第三题是读完前被覆盖的 WAR 风险；第四题最终增量和唯一性可保持，但批次划分、组内外顺序和聚合收益不固定。

## 九、总结：四项 Warp 协作检查

| 检查项 | 必须能回答的问题 |
| --- | --- |
| Membership：参与集合 | 哪些 lane 必须参加？mask 从业务条件还是瞬时执行状态产生？ |
| Arrival：到达与轮次 | 所有被要求的线程能到达对应调用吗？分支、循环、提前返回会不会错配？ |
| Data validity：数据来源 | shuffle 来源确实参与吗？数据已初始化吗？width 是否正确？ |
| Memory ordering：内存顺序 | shared/global 通信是否同步？旧数据是否已读完才能覆盖？ |

Warp 是执行和协作的组织单位，**属于同一个 warp 并不天然建立正确的同步关系**。真正掌握 warp primitives，是能逐条推导参与者、来源、执行约束和内存顺序，而不只是能写一个五轮 reduction。

## 参考资料

下列 [1]–[3] 为本文本次直接核对的博客正文及指南章节；[4]–[6] 为指令、抽象和工具的进一步阅读入口，不作为本次编译或运行证据。

1. Yuan Lin、Vinod Grover：[Using CUDA Warp-Level Primitives][blog]，NVIDIA Technical Blog，2018-01-15。本文 Listing 1–14、隐式锁步风险和 opportunistic programming 的主要来源。
2. NVIDIA：[CUDA Programming Guide — Warp Functions][warp]，尤其是 Warp Active Mask、Vote、Match、Reduce、Shuffle 和 Warp `__sync` Intrinsic Constraints。本文 API 语义、mask/width 与来源有效性依据。
3. NVIDIA：[CUDA Programming Guide — Warp Synchronization Function][sync]。本文 `__syncwarp` 内存顺序说明的依据；同页含 block synchronization、memory fence 与 atomic functions，可用于对照。
4. NVIDIA：[PTX ISA — Parallel Synchronization and Communication Instructions][ptx]。进一步查阅 `shfl.sync`、`vote.sync`、`match.sync`、`redux.sync`、`bar.warp.sync`。
5. NVIDIA：[Cooperative Groups: Flexible CUDA Thread Programming][cg]；[CUB WarpReduce][cub]。进一步阅读高层协作抽象及具体集合/存储契约，不把库接口理解为任意 mask 的万能替代品。
6. NVIDIA：[Compute Sanitizer Documentation][sanitizer]。进一步阅读 Racecheck 与 Synccheck 的诊断范围与限制。

[blog]: https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
[warp]: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-functions
[sync]: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-synchronization-function
[ptx]: https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions
[cg]: https://developer.nvidia.com/blog/cooperative-groups/
[cub]: https://nvidia.github.io/cccl/unstable/cub/api/classcub_1_1WarpReduce.html
[sanitizer]: https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html

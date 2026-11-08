# 五类指令的案例、目录与实施规划

**状态：仅规划。** 盘点依据是当前源码与检查器，不是本次编译产物。下面“目标 SASS”表示要调查的指令族，绝不表示一个 C++/PTX 操作必然生成该 opcode。先完成一组例子，再扩展下一组；不以覆盖表中的名字数量作为完成指标。

## 1. 组织原则：案例保存证据，五类提供索引

保留 `02_memory`、`04_sync`、`07_async` 的语义专题和当前源码，不移动历史记录，不为五类各复制一份 kernel。新建的 `02_sass_atlas` 是跨专题阅读入口，只为现有材料无法清晰表达的机制新增最小源码。

```text
03_instruction/
├── 01_setmaxnreg/                  # 已有，保留原位
└── 02_sass_atlas/
    ├── README.md                   # 本轮：学习目标、方法与资料
    ├── PLAN.md                     # 本轮：盘点、五类矩阵、阶段计划
    ├── CARD_TEMPLATE.md            # 本轮：指令分析记录格式
    ├── families/                   # 后续：有实际分析后再创建
    │   ├── memory.md
    │   ├── address_integer.md
    │   ├── synchronization.md
    │   ├── control_flow.md
    │   └── data_movement.md
    └── examples/                   # 后续：仅放新例子，不复制已有专题
        ├── 01_indexed_copy/
        ├── 02_integer_bits/
        ├── 03_control_flow/
        └── 04_local_storage/
```

本轮只创建前三份有实际内容的文档，不建立空的 `families/`、`examples/`、动作目录或假日志。上面是拟定名称，不是现成命令入口。`families/*.md` 将按本页矩阵为每条实际观察到的指令建立小节，链接案例证据；新 opcode 的解释放在所属类别，跨类只交叉引用。

- **新基础例子**：一个例子一份 `main.cu` 与 README，包含少量命名 kernel；不拆成每条 opcode 一个工程。先能从 `main` 顺着读完分配、启动、结果核对和释放。
- **已有特性例子**：分析写入原专题的 `INSTRUCTIONS.md`，类别页链接具体小节；原 `RESULTS.md` 的历史数字和链接不变。
- **原子对照**：优先在 `04_sync/03_thread_scopes` 增加小型 `atomic_return.cu`，不把其带到新通用框架。
- **tensor-map TMA 缺口**：后续在 `07_async/02_tma_tile` 实现独立小专题，由 Synchronization 与 Data Movement 共同引用；本轮不创建目录。
- **寄存器配额**：仍由 `01_setmaxnreg` 负责，不把 `USETMAXREG` 强行塞进基础访存课；local/spill 分析向它交叉链接。

每个未来新增例子的产物与动作遵守 [记录布局](../../RECORD_LAYOUT.md)：

```text
<案例>/
├── README.md / main.cu
├── build/<arch>-release/            # PTX、cubin、需要时的可执行程序
└── commands/
    ├── environment/run.sh          # 工具版本；运行前核对设备/驱动能力
    ├── compile/run.sh              # 唯一显式生成上述产物的动作
    ├── static/run.sh               # 读取产物；SASS 原文与日志在本动作下
    └── run/run.sh                  # 可选、小规模正确性，不计时
```

按需增加 `cfg`、`sanitizer` 或 `profile` 独立动作，不能让 `static` 隐式编译或运行，也不新增 `all/lab.sh`。每个动作只写自己的 `<variant>.log` 或固定 `log/`，记录命令、参数、输出和退出码；重跑覆盖当前动作同配置日志/报告，不覆盖其他配置或旧证据。修改源码和编译参数后显式重编，失败不能继续使用旧产物冒充本轮成功。

## 2. 当前材料盘点与入口缺口

以下链接均指当前源码；静态检查能力只描述检查器写了什么规则，不表示这些规则本轮已运行通过。

| 素材 | 精确入口与函数 | 可复用部分 | 尚缺部分 |
| --- | --- | --- | --- |
| 单操作 global 访存 | [intrinsics/probe.cu](../../02_memory/03_load_store_intrinsics/probe.cu)：`probe_load_*`、`probe_store_*`、`probe_float4_ca` | 命名 kernel、缓存/类型对照；[检查器](../../02_memory/03_load_store_intrinsics/check_static.py) 按 kernel 检查 PTX 和 `LDG/STG` | 宽度、地址、操作数的逐条解释；probe 不能直接按多线程共同写同一地址运行 |
| 地址相关加载 | [cache/main.cu](../../02_memory/01_cache_operators/main.cu)：`read_reuse`、`read_stream`、`write_stream` | 指针链与缓存指令；[检查器](../../02_memory/01_cache_operators/inspect_sass.py) 检查读循环内 `LDG` | 不运行原长循环/streaming 测量来替代指令阅读 |
| eviction 提示 | [eviction/main.cu](../../02_memory/02_eviction_priority/main.cu)：`copy_hint`、`hot_probe` | generic/global 地址、提示与布尔操作候选 | 不把时钟区间或 cache 提示当命中率；后续再读复杂 probe |
| local/spill 候选 | [setmaxnreg/main.cu](../01_setmaxnreg/main.cu)：`experiment`、`advance` | 活跃累加器与配额；检查器统计 `LDL/STL` | 数组可能标量化；尚无简单受控 local 课，统计为零也是合法观察 |
| shared 与 warp | [warp/main.cu](../../04_sync/04_warp_functions/main.cu)：`shared_exchange_kernel`、`shuffle_kernel`、`linear_tid` | [检查器](../../04_sync/04_warp_functions/check_static.py) 按 kernel 查 `STS/LDS/WARPSYNC/SHFL` | 细化 shuffle 操作数、mask、width 与数据依赖 |
| 原子计数 | [scopes.cu](../../04_sync/03_thread_scopes/scopes.cu)：`count_gpu`、`count_shared` | `fetch_add` 返回旧值作为 ticket；global/shared 可分别观察 | 使用/丢弃旧值的单变量对照；不是改变地址空间同时改变 scope 的性能比较 |
| 内存发布 | [memory_model/main.cu](../../04_sync/01_memory_model/main.cu)：`message_passing` | release/acquire、fence 变体；已有 static 输出 SASS | 按发布代码段归因，排除外围 grid/block 同步的指令 |
| block 与异步同步 | [barriers/main.cu](../../04_sync/02_async_barriers/main.cu)：`exchange`、`tiled_copy`；[copies.cu](../../04_sync/03_thread_scopes/copies.cu)：`copy_single`、`copy_pipeline` | 到达/等待、普通与异步拷贝、commit/wait | `BAR/LDGDEPBAR/DEPBAR` 和 mbarrier lowering 的逐 kernel 分析 |
| SM90 线性 bulk | [tx90.cu](../../04_sync/02_async_barriers/tx90.cu)：`transaction_copy` | 1024 bytes 搬运与 `barrier_arrive_tx` 事务计数 | 普通 static 不覆盖此分支；无 tensor-map descriptor 与多维坐标示例 |
| 特殊寄存器补充 | [async/common.cuh](../../07_async/01_async_execution/common.cuh)：`timer_ns`、`delay_kernel` | `%globaltimer` 的 PTX 来源 | 尚无专用 SASS 检查；不运行 delay 循环来推断指令周期 |

**实施前的入口核对**：`02_memory/02_eviction_priority`、`02_memory/03_load_store_intrinsics`、`03_instruction/01_setmaxnreg` 的 `commands/compile/run.sh` 与 `commands/static/run.sh` 均存在；intrinsics 已生成独立 probe PTX/cubin。复用这些入口，不重建另一套包装。脚本存在不代表本轮执行通过，依赖、工具版本及当前配置可用性仍需实际展开时验证；本轮未执行这些动作，也不据顶层“已有专题”计数承诺全部路径已验证。

## 3. Memory：数据在哪里

先区分地址空间，再看缓存和原子。以下 PTX 是来源候选，不是本次生成文本。

| 编号 | 实际例子与单变量对照 | PTX 来源候选 | 目标 SASS 与必须回答的问题 |
| --- | --- | --- | --- |
| M1 | 现有单操作 probe；先 plain，再只改 cache operator；另用合法对齐的 scalar/`float4` 对照 | `ld.global.*`、`st.global.*`，向量类型可能产生 `.v4` | `LDG/STG`：地址寄存器、数据寄存器、访问宽度分别在哪？SASS 后缀不按 PTX 字面机械翻译 |
| M2 | 现有 shared exchange；先分析固定参与者的正常路径 | `ld.shared`、`st.shared` | `LDS/STS`：哪个线程写、哪个线程读？数组布局和两次同步分别保护哪条依赖？ |
| M3 | 新 local 例子：每线程小数组，常量索引与受限的运行时索引对照；再独立研究寄存器压力 | `ld.local/st.local`；spill 也可能在 PTX 之后引入 | `LDL/STL`：local 对象、调用栈和 spill 不能混称；结合 ptxas 资源/溢出报告区分，不能只看是否存在 opcode |
| M4 | 原子 ticket：同一 global counter、同一类型/scope/order，仅改变旧值是否写入输出 | `atom.global.add`、`red.global.add` 或等价 lowering | `ATOM/ATOMG/RED/REDG` 家族：有无结果依赖？保留完整实际 mnemonic；不得承诺丢弃返回值一定变 RED |
| M5 | 对照阅读已有 `count_shared`，不与 M4 合并成一个多变量实验 | `atom.shared.*` | `ATOMS` 等实际形式；shared 存储与 block scope 是不同维度 |
| M6 | 扩展阅读 kernel 参数/只读常量和 generic 指针 | `ld.param/ld.const`、generic `ld/st` | `LDC/ULDC`、`LD/ST` 或专门地址空间形式；constant bank 取参不是任意 global load |

M3 的运行时索引必须在数组边界内；若优化后没有 local 指令，记录原因，再把显式 PTX local 探针作为单独对照，而不是偷偷扩大数组或用 `volatile` 改变原实验。M4 小运行先用单线程验证旧值，再用一个 warp 验证计数与 ticket 集合；不假定线程编号决定原子顺序，不进行冲突度扫描。

理解检查：看到 `LDL` 能否断言寄存器不够？看到 `LDG` 能否断言本次一定访问 DRAM？两者均不能，需补不同的证据。

## 4. Address & Integer：从索引到位模式

整数例子统一从 `uint32_t` 入手，溢出按无符号模运算解释；64 位地址运算显式区分“乘法前扩宽”和“乘法后转换”。

| 编号 | 例子/一次只改什么 | PTX 来源候选 | SASS 候选及阅读任务 |
| --- | --- | --- | --- |
| A1 | 新索引复制；先 `i=threadIdx.x`，再独立加入 block 索引 | `%tid.x/%ctaid.x/%ntid.x`、`mad.lo`、`mul.wide`、`add` | `IMAD`、`IMAD.WIDE`、`LEA`：由线程号到元素号到字节地址；地址计算不访问内存 |
| A2 | 分开比较 32/64 位索引；再比较乘数为编译期常量/运行时值 | `cvt`、`mul.wide`、`mad.wide`、`add` | `LEA`、`IMAD`、`IADD3` 和高位/进位序列；不要求一定选择 `LEA` |
| A3 | 运行时输入 `a+b+c`，结果写回；与两输入加法对照 | `add.u32` 序列或融合形式 | `IADD3`：三输入在哪里？是否使用零寄存器？是否被地址运算中的同名指令干扰？ |
| A4 | 合法移位量的 rotate/funnel shift；先固定 8，再单独观察变量 shift | `shf` 或 `shl/shr/or` | `SHF`：画两个 32 位输入拼接与选取的位区间；C++ 移位不能用负数或大于等于位宽的值 |
| A5 | `(a & b) \| (~a & c)`，对照天然表达式与明确 PTX `lop3.b32` 探针 | `and/or/not` 或 `lop3` | `LOP3`：列 8 行真值表，解释 LUT（查找表）立即数；SASS 重排操作数后不能照抄 PTX 的立即数解释 |
| A6 | 第二批：bit count、bit reverse、最高有效位位置 | `popc`、`brev`、`clz/bfind` | `POPC/BREV/FLO` 等候选；检查输入零与有符号边界，未观察到则保留等价序列 |

常量输入全部写死可能被折叠。用运行时输入并将结果写出，让被研究的计算对输出有贡献；内联 PTX 只用于明确语义的对照，不保证它与 SASS 一一对应。单课只引入一个 intrinsic 或 PTX 操作，不先搭通用参数生成器。

理解检查：同一条 `IMAD` 是在计算用户数值还是构造地址？必须追踪其结果的消费者才能回答。

## 5. Synchronization：先分清等待对象

| 编号 | 语义与实际素材 | PTX 来源候选 | SASS 观察目标与边界 |
| --- | --- | --- | --- |
| S1 | `exchange<0>`：shared 写完才能读、读完才能覆盖 | `bar.sync` | `BAR`：block 参与集合与两个交接点；不在部分线程提前退出的路径上随意加 barrier |
| S2 | `shared_exchange_kernel`：full/half mask 正常路径 | `bar.warp.sync` | `WARPSYNC`：哪些 lane 必须到达？哪些 shared 访问被排序？ |
| S3 | `message_passing` 的 release/acquire 与显式 fence；先同 block 分析 | `ld.acquire/st.release`、`fence.*` 或版本相关等价形式 | `MEMBAR`、带语义的访存序列及架构相关 `FENCE` 候选；没有独立 `MEMBAR` 不等于没有顺序保证 |
| S4 | `copy_single<0>` 普通搬运与 `<2>` thread-level async 拷贝；分析 commit、wait、group.sync 各自作用 | `cp.async`、`cp.async.commit_group/wait_group`、`bar.sync` | `LDGSTS/LDGDEPBAR/DEPBAR/BAR` 候选；等自己提交的数据完成不等于所有线程完成交换 |
| S5 | `exchange`/`tiled_copy`：arrival、phase、wait；再读 SM90 `transaction_copy` | `mbarrier.*`、transaction 计数、async copy | 目标相关 `SYNCS` 等实际序列；不可把一个 barrier API 简化成固定一条机器指令 |
| S6 | 后续真正 cluster barrier 的最小例子，与现有 cluster 原子计数区分 | `barrier.cluster.*` | `UCGABAR_ARV/WAIT` 等架构相关候选；现有 `count_cluster` 不是此例，暂不实施 |

每课画“谁生产 → 哪条保证 → 谁消费 → 何时允许复用”的依赖图。对 `DEPBAR` 的操作数和等待掩码，先记录观察与官方可确认部分；不从反汇编臆造完整 scoreboard（未完成指令依赖的跟踪机制）规范。

理解检查：`BAR`、`MEMBAR`、`DEPBAR` 能互换吗？不能；分别解释参与线程、内存顺序和机器依赖的不同职责。`BSSY/BSYNC` 的控制流重汇合也不等同于 shared 数据发布。

## 6. Control Flow：不要从源码 if 直接推断分支

现有 kernel 能提供边界判断和循环素材，但尚无独立的分支/调用教学对照。拟在 `examples/03_control_flow` 用少量命名 kernel 逐步补齐，不改同步示例来制造非法分歧。

| 编号 | 小例子与对照 | PTX 来源候选 | SASS 目标与可接受的不同结果 |
| --- | --- | --- | --- |
| C1 | 复制尾部保护；再比较短 `if` 与选择表达式 | `setp`、谓词执行、`selp`、`bra` | `ISETP/PSETP/PLOP3`、`@P`、`SEL`、`BRA/EXIT`；`@P` 是执行条件，不是另一条 opcode；可能没有分支 |
| C2 | 只改条件来源：整个 warp 一致，或依赖 lane 奇偶；每支保留可观察计算 | 谓词与 `bra` | `BRA/BSSY/BSYNC` 等候选；画基本块与可能的活跃 lane 集合，不用静态图证明运行时分歧开销 |
| C3 | 很短的固定次数循环与运行时次数循环，记录是否展开 | `bra` 回跳与比较 | 回边 `BRA` 和退出条件；区分有用循环与函数结束后的不可达/填充序列 |
| C4 | 相同设备函数，内联版与 `__noinline__` 版；返回值参与输出 | `call/ret` 或内联代码 | `CALL/RET`：调用位置、参数/返回值、保存与恢复；可能仍有其他优化，不能只靠关键词判断 |
| C5 | 稠密 `switch`，不同分支做非线性且有输出的运算；枚举全部 selector 核对 | `brx.idx` 或比较/分支序列 | `BRX/BRXU` 或比较树；跳转表不是 C++ `switch` 的保证，未生成 `BRX` 就记录未生成 |

优先读真实优化结果，不用 `-G` “强制出指令”后当作 release 行为。若为 C5 增加 PTX 级间接跳转探针，单独命名并解释它只保证虚拟指令契约；函数指针与设备链接等更复杂例子暂缓。

理解检查：静态 SASS 同时存在两条分支路径，是否说明每个线程都会执行两边？不能；需要知道条件、谓词与实际活跃线程。

## 7. Data Movement：从寄存器到异步搬运

| 编号 | 小例子与对照 | PTX 来源候选 | SASS 目标与阅读任务 |
| --- | --- | --- | --- |
| D1 | 索引复制中读线程坐标、传递运行时值；单独补特殊寄存器探针 | `mov`、`%tid.x/%ctaid.x` | `MOV/S2R`、`S2UR/UMOV` 等；赋值可能消失，也可能映射为 `IMAD.MOV` 等形式，不能强制一一对应 |
| D2 | 现有 `shuffle_kernel` 的 idx/up/down/xor；先一个完整 warp、输入为 lane 编号 | `shfl.sync.*` | `SHFL`：源 lane、width、成员 mask、边界值与可选谓词结果；不等于内存 fence |
| D3 | 新字节拼接：两个运行时 `uint32_t`，先固定字节选择器再单独改选择器 | `prmt.b32`，例如 `__byte_perm` | `PRMT`：按字节画输入到输出映射；不要把它与跨 lane 的 `SHFL` 混淆 |
| D4 | 对照 S4 的普通 global→寄存器→shared 与异步 global→shared | `ld/st` 与 `cp.async` | `LDG+STS` 与 `LDGSTS` 等；追踪搬运的数据路径和完成条件，不宣称已经发生重叠 |
| D5 | 现有 `transaction_copy`：线性 bulk 与事务 barrier | bulk copy/`mbarrier` 的实际 lowering 待核对 | Hopper 的 `UBLKCP` 等候选；API 源码存在不代表此 opcode 已生成 |
| D6 | 新 tensor-map TMA：一个满足描述符约束的二维小 tile，明确维度、步长、对齐、坐标与 shared 布局 | `cp.async.bulk.tensor.*` 及完成/可见性协议 | `UTMALDG/UTMASTG` 等候选；先 global→shared 并在完成后校验，再将反向 store 与其不同的完成协议作为独立课 |
| D7 | 第二批扩展：warp vote/match/reduce；更晚再读矩阵搬运 | `vote/match/redux`；`ldmatrix/stmatrix` 按架构 | `VOTE/MATCH/REDUX`、`LDSM/STSM` 等；`REDUX` 的寄存器归约不是 Memory 类的原子 `RED` |

D5 是线性 bulk transaction 示例，不能当作已经有 tensor-map 的 D6；也不应由“没有 tensor map”推断它必然不使用 TMA 硬件。D6 的 descriptor（描述张量布局的元数据）创建/传递、异步代理可见性和 phase 复用规则必须先按所用 API/PTX 版本解释清楚，不能省略为“启动后直接读 shared”。

Host 的 `cudaMemcpyAsync`、stream/event/Graph 不直接对应 kernel 内 `cp.async/DEPBAR`。现有 `07_async/01_async_execution` 的 API 时间线不作为这些机器指令的证据。

理解检查：`PRMT`、`SHFL` 与 TMA 分别搬谁的数据？先画单线程字节、warp lane、global/shared tile 三种图，再看指令。

## 8. 架构与工具范围

| 层级 | 计划做什么 | 不作出的承诺 |
| --- | --- | --- |
| 当前可用目标 | 第一次实际编译前查询 Toolkit 支持列表；运行前另查 GPU、CC、驱动。`sm_89` 只作候选示例 | 本轮没有确认本机 GPU 或 Toolkit，不把历史架构作为当前事实 |
| Ampere/Ada | 基础五类与常规 `cp.async` 优先；同一份简单源码按支持情况比较 sm_80/sm_89 | 不因某指令可用于较早架构就宣布整个现有专题可编译；例如 warp 整组含 native reduce |
| Hopper | bulk、transaction barrier、tensor-map TMA；需要时交叉编译与实际运行分开 | 普通 sm_90 与 sm_90a 不可替换；配额等特定指令另按支持表核对 |
| Blackwell/Rubin | 先读官方对应表，选已有小例子做有限交叉编译差异 | 文档列出不等于本机工具可编译/反汇编，更不等于硬件运行通过；不本轮铺全架构矩阵 |

跨架构对照固定源码、输入和非目标编译选项；改变 Toolkit 版本是另一项实验。优先保留原生 cubin 静态结果，PTX JIT 另算驱动执行路径。某目标编译失败、工具不识别、GPU 不匹配分别记录原始错误，不用降级回退结果冒充目标成功。

## 9. 分阶段交付与停止条件

| 阶段 | 具体交付 | 完成标准 |
| --- | --- | --- |
| P0，本轮 | 本规划、源码盘点、分析卡与导航 | 五类均有真实例子或明确缺口；不伪造编译/运行日志 |
| P1，下一轮优先 | 复用 M1 的现有 probe 入口，补首份 `INSTRUCTIONS.md`；随后 A1/D1/C1 索引复制 | 同一 kernel 的源码/PTX/SASS 可追踪；能解释一次访问的地址和数据；小正确性运行单列状态 |
| P2 | A3～A5、D3、C2～C5；按课逐项实现 | 位表/字节表/控制流图与输出一致；未生成目标 opcode 有解释，不靠扩大测试规模追指令 |
| P3 | M2～M5、S1～S3、D2；补 local 与原子返回值对照 | 分清寄存器/local、原子旧值、参与 mask 与 happens-before；不做延迟/冲突度排名 |
| P4 | S4～S5、D4～D6；先现有 cp.async/bulk 再新 tensor-map | 完成与复用协议讲清，准确记录架构门槛；无硬件可只交付编译/静态结果，运行仍标未验证 |
| P5，可选扩展 | A6、M6、D7、S6、uniform 指令与新架构差异 | 每次只加一个有源码来源的问题；矩阵计算性能仍留在 GEMM 主线 |

每课最小验收单：

- [ ] 背景、必要前置、具体输入与预期输出，可手算或画图验证。
- [ ] 当前目录、底层命令及选项解释，随后才给等价独立脚本；路径与参数已实际核对。
- [ ] 编译与静态日志绑定同一目标/配置，按函数范围保存实际指令及上下文。
- [ ] 操作数解释形成完整“输入 → 计算/地址/条件 → 输出”链，未知 modifier 明写未知。
- [ ] 至少一个单变量对照；如果优化消去、融合或改写，保留其事实。
- [ ] 正确性已运行或清楚标注未运行及原因；同步论证不依赖一次 PASS。
- [ ] 功能语义、执行管线、延迟、吞吐分列；NCU 字段只填实际采集值或未测量。
- [ ] 有一个预测后验证的理解练习；不会把脚本退出码为零当作学习目标全部完成。

P1 不需要先验证所有旧专题的完整动作或完成整套 M0/M1 性能扫描。性能主线保留原规划，本线只读取必要环境信息、编译、静态分析与少量正确性验证。

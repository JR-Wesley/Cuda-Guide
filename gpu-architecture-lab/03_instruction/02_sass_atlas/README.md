# SASS 指令图谱：从小例子读懂五类机器指令

**当前阶段：规划与源码盘点，不是已经完成的指令实测库。** 本轮只核对当前源码、检查器、入口和官方指令参考；未编译、未反汇编、未运行 GPU，也未采集性能指标。后续按 [案例与目录规划](PLAN.md) 逐个展开，不一次实现全部案例。

## 1. 为什么需要这条学习线

读通信或计算 kernel 时，常会遇到三个问题：

- 一次 `out[i] = in[i]`，为什么除了读写，还有线程编号、整数运算和条件判断？
- `__syncthreads()`、内存 fence 和异步 copy 的 wait 都叫“等待”，它们究竟等什么？
- 源码写了一次赋值、一个 `if`，为什么机器码里找不到对应的 `MOV`、`BRA`？

这里不按 opcode 字母顺序背指令，而是用**可手算的小任务串起源码、PTX 和 SASS**。PTX 是 CUDA 的虚拟指令表示；`ptxas` 将它编译成特定 GPU 的机器码；cubin 是保存设备机器码的文件；SASS 是反汇编工具展示的机器指令。源码行、PTX 指令与 SASS 指令通常不是一一对应。

前置知识只要求逐步掌握：数组元素与字节地址的区别、线程编号、32 位整数、条件判断、寄存器保存中间值。遇到 warp、谓词、原子和同步时，再读对应特性专题，不要求先理解性能测量框架。

**本线的完成目标是解释“这段代码生成什么、数据怎样流动、语义由什么保证”，不是得出延迟或吞吐排行榜。**

## 2. 五类入口与阅读顺序

五类是检索分类，不是五套互不相干的实验。一个 shared exchange 同时涉及 Memory、Synchronization 和 Data Movement，源码只保留一份。

| 类别 | 第一批问题与目标指令族 | 例子与展开位置 |
| --- | --- | --- |
| Memory | 数据在哪个地址空间？宽度、原子返回值怎样影响 `LDG/STG`、`LDS/STS`、`LDL/STL`、`ATOM/RED` 家族？ | [规划 §3](PLAN.md#3-memory数据在哪里) |
| Address & Integer | 元素索引如何变成地址？`IMAD/LEA/IADD3/SHF/LOP3` 做了什么？ | [规划 §4](PLAN.md#4-address--integer从索引到位模式) |
| Synchronization | 等线程、等内存顺序和等异步完成有何不同？`BAR/DEPBAR/MEMBAR/WARPSYNC` 如何定位？ | [规划 §5](PLAN.md#5-synchronization先分清等待对象) |
| Control Flow | 谓词如何产生和消费？`BRA/BRX/EXIT/CALL` 何时出现或消失？ | [规划 §6](PLAN.md#6-control-flow不要从源码-if-直接推断分支) |
| Data Movement | 寄存器、lane、字节和异步搬运分别是什么？`MOV/SHFL/PRMT/S2R` 与 TMA 如何读？ | [规划 §7](PLAN.md#7-data-movement从寄存器到异步搬运) |

建议按案例学习，而不是按上表顺序清空一整类：

1. **单元素复制 → 索引复制**：读懂参数、地址、`LDG/STG` 与尾部判断。
2. **整数和字节小练习**：加法、位选择、移位、字节重排。
3. **分支与函数**：谓词、回跳、调用、控制流图。
4. **线程间交换与计数**：shared、shuffle、原子、barrier、内存发布。
5. **异步完成协议**：普通搬运 → `cp.async` → SM90 bulk → tensor-map TMA。

## 3. 第一课的具体问题：复制为什么需要地址运算

这是**拟新增案例的手算设计，不是当前已有源码或实测输出**。设 `in` 和 `out` 是两个互不重叠的 `uint32_t` 数组，`in = [10, 20, 30, 40]`；启动一个 block、32 个线程，只允许 `i < 4` 的线程执行复制。

| 线程 i | 是否读写 | 输入字节地址 | 输出的预期值 |
| --- | --- | --- | --- |
| 0 | 是 | 输入基地址 + 0 | 10 |
| 1 | 是 | 输入基地址 + 4 | 20 |
| 2 | 是 | 输入基地址 + 8 | 30 |
| 3 | 是 | 输入基地址 + 12 | 40 |
| 4～31 | 否 | 不得访问 | 不写入 |

每个元素 4 bytes，所以 `in + i` 并不是地址加 `i`，而是加 `4*i`。先从现有单操作 probe 学加载/存储，再用这个索引案例观察特殊寄存器读取、缩放地址计算、比较和有条件执行。预期寻找 `S2R`、`IMAD.WIDE` 或其他等价地址序列、`ISETP`、`LDG/STG`、`EXIT`；**不规定它们必须全部出现，更不预先填写 SASS 寄存器编号。**

理解检查：把 `n` 从 4 改为 3，哪些线程不能访问数组？只改运行时的 `n`，是否一定改变静态指令数？答案应来自地址表和控制流，而不是计时。

## 4. 每课只做这些动作

1. **读与预测**：指定一个 kernel、一组小输入，画地址表、位表或依赖图；指出唯一对照变量。
2. **编译**：核对 Toolkit 与目标架构。显式生成需要的 PTX/cubin；不为对比改用 `-G` 调试构建冒充优化产物。
3. **静态阅读**：按函数定位源码 → PTX → SASS，逐条追踪有用值的来源和去向。保留实际指令及上下文，不只贴 opcode 计数。
4. **小规模正确性**：有匹配 GPU 时才单独运行并与 CPU 参考核对；无设备时可以完成静态部分，运行状态保持未验证。
5. **单变量对照**：只改返回值用途、类型、索引宽度或一个控制条件，解释指令变化；未出现目标 opcode 也是需要记录的结果。
6. **总结**：使用 [分析卡模板](CARD_TEMPLATE.md) 写明已知、推断和未知；最后给一道预测再验证的小练习。

`nvdisasm -g` 注释行号，`-bbcfg` 输出基本块控制流图，`-plr` 展示寄存器值的活跃区间；这些都是离线检查，不需要 GPU 执行。JSON 输出仅在安装版本支持时使用，不把新版文档中的选项当作本机必有功能。底层命令教学复用 [SDK 二进制阅读导航](../../../cuda-sdk-lab/experiments/binary/README.md)，各课再给出自己的确切输入、参数、目录和独立动作入口。

默认不做带宽/工作集/occupancy 扫描，不跑长依赖链计时，不用 Nsight Compute 的全量指标集合。NCU（Nsight Compute，kernel 分析器）只作为后续可选的小范围观察；本轮所有动态指标、延迟和吞吐均为**未测量：当前目标是规划与静态指令分析**。

## 5. 四个问题分别记录

| 问题 | 可以依赖的证据 | 本线边界 |
| --- | --- | --- |
| 功能语义 | CUDA/PTX 规范；官方 SASS opcode 简述；具体数据流 | 未公开 modifier 不猜含义，不能把缓存策略当同步 |
| 执行管线 | 对应架构资料、工具定义与有限的动态观察 | 指令分类不等于物理执行单元；没有依据写未知 |
| 延迟 | 专门设计并校验的依赖实验 | 不在本轮实施；一次 kernel 耗时不是指令延迟 |
| 吞吐 | 独立性、并发、资源和计时条件受控的实验 | 不在本轮实施；静态指令数不是单位时间完成数 |

## 6. 官方资料与版本边界

2026-11-08 查阅的 [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#instruction-set-reference) 页面标为 **v13.5**，页面更新时间为 2026-11-02。这是查阅文档版本，不是本机 Toolkit 检测结果。

- §4.1：Turing；§4.2：Ampere/Ada；§4.3：Hopper；§4.4：Blackwell/Rubin。按目标架构选表，而非将所有表合并为同一 GPU 的支持列表。
- 表中可见 `ATOM/ATOMG/ATOMS`，以及不同架构表中的 `RED/REDG/REDS` 等区别；不能用一个 `ATOM` 或 `RED` 字符串匹配代表全部原子路径。
- Hopper 表列有 `SYNCS`、`FENCE`、`UBLKCP`、`UTMALDG/UTMASTG` 等。本线将它们作为后续观察候选，而不是宣称现有 API 已生成这些形式。
- 官方表是 opcode 名称和功能参考，不是完整编码、修饰符、管线与周期规范。PTX 语义查 [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)，动态指标定义查 [NCU Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)。实际开展时记录所用版本和章节。
- [DocumentSASS](https://github.com/0xD0GF00D/DocumentSASS)、[MaxAs](https://github.com/NervanaSystems/maxas)、[CuAssembler](https://github.com/cloudcores/CuAssembler)、[turingas](https://github.com/daadaada/turingas) 仅作社区研究线索；引用时标出架构、版本和非官方属性，不将旧架构实验推广到新 GPU。

目录归属、当前缺口、每阶段交付与完成标准见 [PLAN.md](PLAN.md)。

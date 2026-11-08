# 5.4.6. Warp Functions：参与集合、寄存器交换与内存顺序

> 记录整理（2026-11-06）：历史证据按真实编译配置合并为 `results/<variant>.log`，各 action 仅保留最新一节；见 [记录索引](results/README.md)。无日期目录或源码/脚本副本。本次未编译、未运行 GPU。下文数值仅对应标有 `HISTORICAL IMPORT` 的原记录；链接进入配置日志后按 `ACTION` 与 `FILE` 标界查找。本轮迁移保留这些历史证据；新日志写入独立动作目录，旧结论不自动套到新日志。

本专题把提供的 Warp Functions 材料转换成带 CPU 校验的 CUDA 实验。入口是 [main.cu](main.cu) 与 [独立动作目录](commands/README.md)，调试步骤见 [DEBUGGING.md](DEBUGGING.md)，本次设备、运行证据与限制见 [RESULTS.md](RESULTS.md)。

**当前动作索引：[commands/README.md](commands/README.md)。** 每个动作独立脚本、独立目录和同目录日志；先 compile，缺产物不会隐式构建。

## 博客精读

[深度解读：Using CUDA Warp-Level Primitives](USING_CUDA_WARP_LEVEL_PRIMITIVES.md) 从 NVIDIA 2018 年博客的具体例子出发，解释 SIMT、参与 mask、shuffle 来源有效性、共享内存 RAW/WAR 同步与聚合原子；补充现代整数归约、历史博客与当前指南的措辞差异及四组实验建议。文中列明原作者、发表日期和官方参考来源，区分原文、分析与未执行的实验，不增加本专题已验证结果。

## 运行

在本目录每次选择一个动作：

```bash
CUDA_ARCH=sm_89 bash commands/compile/run.sh
CUDA_ARCH=sm_89 bash commands/run/run.sh
CUDA_ARCH=sm_89 bash commands/static/run.sh
CUDA_ARCH=sm_89 bash commands/sanitizer/run.sh
CUDA_ARCH=sm_89 bash commands/debug/run.sh
CUDA_ARCH=sm_89 BUILD_MODE=debug bash commands/compile/run.sh
CUDA_ARCH=sm_89 bash commands/gdb/run.sh
CUDA_ARCH=sm_89 bash commands/jit/run.sh
```

`89` 对应本次实测设备，换设备时重新查询。框架整体要求 compute capability ≥ 8.0，因为直接使用六种整数 `__reduce_*_sync`；不是说所有 warp API 都要求 8.0。脚本未设置 `CUDA_ARCH` 时使用通用的 `sm_80`，不据此判断本机设备。编译器也必须支持目标架构。

要求 C++17、CUDA Runtime 与 CUB 头文件。`<cuda/warp>` 存在时测试 libcu++ 广播；`CCCL_VERSION >= 3001000` 时增加 `warp_match_all` 测试。其他安装使用 intrinsic 广播并明确输出可选 API 的 `SKIPPED`，不能把它记为该 API 验证通过。仅本次 Toolkit/CCCL 组合已实测，版本表见 [CCCL Warp API](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/warp.html)。

普通动作使用 `build/<arch>-release/bin/warp_functions`，包含目标 cubin 与对应 PTX；只有 `gdb` 使用 `build/<arch>-debug/bin/warp_functions` 的 `-O0 -g -G` 构建。**名为 `debug` 的动作仍是 release 配置**，与 run/static/sanitizer/jit 同用 `-O3 -lineinfo`。先显式 compile，后续动作只读取对应产物；编译失败撤销成功标记，禁止继续运行旧程序。`jit` 强制驱动从 PTX 编译，失败时保留错误并返回非零。

日志位于脚本同目录 `commands/<action>/<variant>.log`，同动作同配置重跑覆盖（包括失败），其他动作日志不变。命令、实际参数、原始输出及退出码均记录；检查器的临时步骤文件在退出后清理。不生成自动快照、时间戳归档或 history。已有 `results/` 原始日志和历史报告保持原样，当前动作不写入。

程序返回 `0` 表示所执行的必选测试通过，`1` 表示 CUDA 调用或校验失败，`2` 表示参数错误，`77` 表示没有设备或设备不满足框架要求。脚本中缺工具、静态解析错误、sanitizer 错误均不会当作成功。`nvidia-smi` 查询失败会记录原始退出码，随后仍尝试 CUDA Runtime；实际使用的设备以程序打印的可见 device 0 为准。

## 实验矩阵

| kernel / 测试 | 输入与边界 | 校验内容 |
| --- | --- | --- |
| `collectives_kernel` | 13 种长度（1～1025）、2 个确定性种子、6 种 block：1/17/32/33/64 线程及 `(8,8)` | ballot/all/any、match any/all、六种整数 reduce、稀疏 mask、正负零的位比较 |
| `shuffle_kernel` | 2 个完整 warp，width=1/2/4/8/16/32，5 种 delta，4 种 XOR laneMask | 广播、srcLane 回绕、up/down 边界、XOR 的跨分组行为、inclusive scan |
| `library_kernel` | 64 个线程；8 个逻辑 8-lane scan 组、2 个 32-lane reduce 组 | CUB 与独立 CPU 求和、libcu++ 广播和 match |
| `shared_exchange_kernel` | 2 个 warp、组内/跨半 warp 交换、1/2/101 轮 | barrier 范围与缓冲区复用，奇数轮反转、偶数轮恢复 |
| `width_debug_kernel` | 1 个完整 warp | 复现默认 width 导致的 16 个逻辑错误，然后验证修复 |
| `debug_contracts.py` | 9 个显式模型案例 | 不正确的 mask、shuffle source、width 和跨半 warp 访问 |

输入是有正负值的确定性整数，求和范围不会溢出。CPU 直接遍历每个成员求期望值。`quick` 只缩减集合测试的长度集合，保留 1/17/33/65、所有 block 布局和其余测试，用于 sanitizer；不会因此省略稀疏集合或物理尾 warp。

对原材料的 `value=31-lane`，可手算得到 32-lane 总和为 496；每组 8 个线程扫描时，四组末 lane 的值分别为 220、156、92、28。这些是理解 scan/reduce 的示意期望值；本次可执行测试使用上述随机整数输入，避免仅靠这一组规律数据判断正确性。

## mask 是参与集合，width 是寻址规则

对一次 collective，先确定哪些线程参加，再检查每个调用者都在 mask 中、mask 指定的每个未退出线程最终都以相同 mask 到达对应调用。分支可以使用互不相交的集合；不要求参加者在同一个时钟周期到达。对 shuffle，还必须保证每个源 lane 也参加这次调用。

二维 block 的 lane 不能直接用 `threadIdx.x % 32`。本例先计算：

```cpp
int t = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
int lane = t % 32;
```

本例从 block 形状计算物理存在线程的 mask，所有这些线程在任何提前退出之前执行 `__ballot_sync(physical, i < n)`，得到有效输入集合，然后才退出无效输入线程。这同时覆盖 block 末尾的物理不完整 warp 和数据尾部。

`__activemask()` 仅记录到 `snapshot`，不据此构造后续协议。校验只要求当前调用 lane 的 bit 存在，不要求快照总等于有效输入 mask。在 `if (pred)` 内看到非空 active mask，不能反推出所有 `pred` 为真的线程集合；当前调用者本来就在其中。另一方面，快照之后调度发生变化也不自动意味着 collective 非法：还要看被命名线程最终是否遵守该次调用的契约。

稀疏集合不会把物理 lane 重新编号。例如成员为 `{0,3,7}`，`shfl_down(...,1)` 仍然请求相邻的物理 lane。本例用 `ballot` 得到稀疏集合，分别运行 native reduce 和遍历置位 lane 的 shuffle 求和；后者是便于检查的参考实现，不是性能优化。

`width` 只接受 1、2、4、8、16、32。测试还覆盖 XOR 的特殊规则：在 width=8 时，lane 8 执行 XOR 8 能读取前一个分组的 lane 0；lane 0 企图访问后一个分组的 lane 8 时返回自己的值。因此不能把 XOR 的 width 理解成所有访问都被隔离在本组。本节 API 语义依据 [CUDA Programming Guide §5.4.6](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-functions)，材料中的具体示例修正如下。

## 原材料中需要修正或补充的地方

| 原片段 | 问题与本例处理 |
| --- | --- |
| `0xb1111` | 这是十六进制数，不是 4-lane 的二进制 mask。应使用 `0b1111` 或 `0xf`。主机诊断展示其多出/缺少的参与者问题 |
| `__shfl_sync(FULL, value, 16)` 注释为两半分别广播 | 默认 width=32，全部 lane 都接收 lane 16。分半广播应为 `__shfl_sync(FULL, value, 0, 16)`；`debug` 在 GPU 上对照 |
| 上半 warp 用 `0xffff0000` 同步，再读 `shared[31-lane]` | 读的是下半 warp 写入的数据，barrier 不覆盖生产者。组内反转改为 `shared[47-lane]`；跨半交换则在双方读写之间使用 full mask |
| 32 个线程调用 `WarpScan<int,8>`，共用一个 TempStorage | 有 4 个逻辑组；应按逻辑组分配。此处 64 个线程分配 8 份。某版本的 shuffle 路径未使用临时空间，也不能据此形成可移植的共用规则 |
| 广播前仅初始化 lane 0 的局部变量 | 本例初始化所有参与者的参数，避免依赖其他 lane 未初始化的 C++ 值 |

对 `__match_any_sync` 的额外检查使用 `+0.0f` 与 `-0.0f`：数值上相等，bit pattern 不同，结果分成偶数与奇数 lane 两组。`match_all` 同时检查 mask 返回值和 predicate，覆盖成员值全部相同及不相同的情况。

## CUB 与 libcu++ 对照

`WarpScan<int,8>` 为每个逻辑组使用独立 `TempStorage`；`WarpReduce<int,32>` 为每个物理 warp 使用独立存储。这里只把 CUB aggregate 的 lane 0 结果作为广播源，不假设其他 lane 的 aggregate 有意义。scan/reduce 使用不同存储，没有跨 collective 复用；若后续复用，需要在读写完成后加适当同步。对应接口约束见 [CUB WarpScan](https://nvidia.github.io/cccl/unstable/cub/api/classcub_1_1WarpScan.html) 和 [CUB WarpReduce](https://nvidia.github.io/cccl/unstable/cub/api/classcub_1_1WarpReduce.html)。

libcu++ 组在完整 warp 中测试 `cuda::device::warp_shuffle_idx`，并对广播结果执行 `warp_match_all`。本例没有使用包装函数来绕过参与集合或架构约束。`warp_match_all` 也不提供内存顺序，参见 [接口说明](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/warp/warp_match_all.html)。

## 内存顺序与静态证据

vote、match、reduce、shuffle 在这里交换寄存器值，不替 shared/global memory 建立发布关系。共享内存实验每轮遵守：写入 → 覆盖生产者和消费者的 `__syncwarp(mask)` → 读取 → 同 mask 的第二次 `__syncwarp` → 下一轮覆盖写。第二次同步保护尚未完成的读者；同半组用恰好覆盖该组的 mask，跨半组用 full mask。这里不进行跨 warp 通信，不能将结论推广到 block/grid 同步。内存模型背景见 [01_memory_model](../01_memory_model/README.md)。

`static` 按 kernel 解析 PTX 和 SASS，而非只搜索整个二进制。检查 `activemask`、vote、match、全部六种 redux、四种 shfl、`bar.warp.sync` 以及相应机器指令类别。计数是编译产物中指令文本的静态数量，受展开、分支和编译器 lowering 影响，不是动态执行次数，也不是延迟或吞吐率。

本专题验证功能语义和定位错误，没有性能计时。缓存命中率、DRAM 流量、指令延迟、吞吐率均未测量；不能由 native reduce 的指令形式直接断言它在任意工作负载上快于 CUB。

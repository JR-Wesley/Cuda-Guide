# 9.7.10.1. Cache Operators：从 PTX 提示到真实访存

本实验覆盖全部 5 个 load cache operators（`.ca/.cg/.cs/.lu/.cv`）、全部 4 个 store cache operators（`.wb/.cg/.cs/.wt`），并加入省略 cache operator 的默认指令作对照。提供可执行代码、数据校验、重复采样、PTX/SASS 检查及实测分析。

对应的 CUDA C++ 接口见 [Low-Level Load and Store Functions](../03_load_store_intrinsics/README.md)：`__ldca/__ldcg` 等 intrinsic 的类型校验与指令映射，以及额外的 `__ldg` 只读约束。

**核心问题：保持程序的同步关系不变，只改变访存指令的缓存提示，生成指令、执行时间和结果分别有什么变化？**

这些操作符是性能提示，不建立原子性、release/acquire 或 happens-before。`.cg` 绕过 L1，不等于“完成线程间同步”；`.cv` 和 `.wt` 对系统内存有特别定义，不等于任意地址上的“强制访问显存”。语义依据为你提供的材料及 [PTX ISA §9.7.10.1](https://docs.nvidia.com/cuda/parallel-thread-execution/#cache-operators)。

**当前动作索引：[commands/README.md](commands/README.md)。** 每个动作独立脚本、独立目录和同目录日志；先 compile，缺产物不会隐式构建。

## 1. 文件与运行入口

| 文件 | 用途 |
| --- | --- |
| [main.cu](main.cu) | 内联 PTX、3 组设备内存实验、mapped host memory 校验 |
| [独立动作目录](commands/README.md) | 编译、运行、生成 PTX/SASS、memcheck；保存命令和日志 |
| [inspect_sass.py](inspect_sass.py) | 提取 11 种指令形式，检查重复读取循环内是否仍存在加载 |
| [RESULTS.md](RESULTS.md) | 本机环境、指令映射、实测数据与解释边界 |

从本目录运行：

```bash
bash commands/compile/run.sh  # 显式编译，后续动作只复用
bash commands/run/run.sh
bash commands/static/run.sh
bash commands/sanitizer/run.sh

# 拉长地址依赖链，减小短 kernel 的计时干扰
REUSE_PASSES=2048 bash commands/run/run.sh

# 调整大数组规模与采样次数
STREAM_MIB=128 SAMPLES=15 bash commands/run/run.sh
```

默认编译目标为本机 `sm_89`；其他 GPU 设置 `CUDA_ARCH`，例如 `CUDA_ARCH=sm_80 bash commands/run/run.sh`。需要 C++17 CUDA 编译器；`static` 还需 `cuobjdump` 和 Python 3。PTX 规范中的 `sm_20+` 是这些指令的历史支持下限，不表示当前 Toolkit 仍能为 `sm_20` 编译。

独立动作支持 `compile/run/static/sanitizer`。`NVCC`、`CUOBJDUMP`、`COMPUTE_SANITIZER` 可指定工具路径。`run` 默认参数为 `STREAM_MIB=64 SAMPLES=9 REUSE_PASSES=64`；`sanitizer` 默认缩小为 `1/1/2`，其计时不得用于性能比较。

动作入口为 `commands/<action>/run.sh`；先显式执行 `commands/compile/run.sh`，其余执行/检查动作只复用 `build/<variant>/` 的已有产物，缺失即提示先编译，不隐式构建。修改源码、头文件、编译选项或工具链后必须主动重新编译；没有哈希缓存或自动失效判断。

日志位于脚本同目录 `commands/<action>/<variant>.log`，同动作同配置重跑覆盖（包括失败），其他动作日志不变。命令、实际参数、原始输出及退出码均记录；检查器的临时步骤文件在退出后清理。不生成自动快照、时间戳归档或 history。已有 `results/` 原始日志和历史报告保持原样，当前动作不写入。

历史证据的合并范围、状态和精确复现配置见 [results/README.md](results/README.md)。退出码 0 表示全部完成，1 表示校验或运行失败，77 表示 GPU 初始化不可用或 mapped memory 测试跳过。

## 2. 逐项理解语义

### Load

| 提示 | 应怎样理解 | 不能推出什么 |
| --- | --- | --- |
| 默认 / `.ca` | 正常缓存策略，允许在 L1、L2 等层级缓存，适合有复用的数据 | 不能保证每次都命中 L1，也不能保证默认与显式 `.ca` 在所有架构生成相同 SASS |
| `.cg` | 不在 L1 缓存，在 L2 及更低层级缓存 | 不是绕过所有缓存，也不是 acquire |
| `.cs` | 流式访问，采用 evict-first 倾向，减少临时数据占用缓存 | 不是“不缓存”，也不保证所有流式程序更快 |
| `.lu` | last use；可用于恢复溢出寄存器、栈帧等不再复用的数据；global 地址上具有 streaming 语义 | 不是 free，不缩短对象生命周期，也不保证与 `.cs` 的机器码或性能一致 |
| `.cv` | 对 global System Memory 地址，丢弃匹配的 L2 行并重新取数 | 不能把它概括为任何 `cudaMalloc` 地址上的“每次访问 DRAM” |

`.cs` 应用于 Local window 时按 `.lu` 处理。本例使用 `ld.global`，不测试 local spill/stack 路径。

### Store

| 提示 | 应怎样理解 | 不能推出什么 |
| --- | --- | --- |
| 默认 / `.wb` | 对一致缓存层级使用正常 write-back 策略 | “all coherent levels” 不等于所有 SM 的 L1 自动一致 |
| `.cg` | 不在 L1 缓存，在 L2 及更低层级缓存 | 不能修复其他线程错误的同步协议 |
| `.cs` | 用 evict-first 倾向处理流式输出，减少缓存污染 | 不保证读回这些数据更快 |
| `.wt` | 对 global System Memory 地址，写穿 L2 | 不是 release、系统级 fence 或持久化存储保证 |

选 cache operator 的依据是访问模式和目标架构。保证并发正确性需要单独分析同步与内存顺序，参见 [CUDA C++ Memory Model 实验](../../04_sync/01_memory_model/README.md)。

## 3. 最小内联 PTX 写法

源码中的模板封装本质上是下面两种指令：

```cpp
// device function 内；p 指向对齐的 global uint32_t
uint32_t value;
auto address = __cvta_generic_to_global(p);
asm volatile("ld.global.cg.u32 %0, [%1];"
             : "=r"(value) : "l"(address) : "memory");

asm volatile("st.global.cs.u32 [%0], %1;"
             : : "l"(address), "r"(value) : "memory");
```

`r` 对应 32 位整数寄存器，`l` 对应 64 位地址操作数。`__cvta_generic_to_global` 显式转换地址空间；`cudaMalloc` 和映射得到的设备指针都用于 global 地址访问。所有元素为自然对齐的 4 字节整数。

这里的 `asm volatile` 是编译器层面的限定，不是 PTX 的 `ld.volatile`；`"memory"` 声明汇编的隐式内存影响，也不是设备 fence。它们不能赋予本例线程同步语义。约束写法与注意事项见 [NVIDIA Inline PTX Assembly](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html)。

尤其要检查后端优化：固定地址的普通加载即使来自 `asm volatile`，仍可能在 PTX 到 SASS 的过程中被合并或移出循环。不能只看 `.cu` 或 PTX 中写了几次 `ld`，就按次数计算实际访问量。

## 4. 实验 A：小工作集的依赖地址环

每个 block 有 256 个线程，拥有 8 KiB 数据。每个线程沿 8 个槽位构成的环读取；同一 warp 的同一步访问相邻元素，保持合并访问布局。

```text
线程 t 的地址序列：
base+t → base+256+t → base+512+t → … → base+1792+t → base+t

关键循环：
index = load<Op>(data + index);
sum += index;
```

环节点存放下一个节点的索引。加载得到的值决定下一条加载的地址，因此后端不能简单把节点值提前读进寄存器，再重复加法。8 次加载完成一圈；`REUSE_PASSES` 指圈数。

初始化 kernel 先建立环，随后同一 stream 启动测试 kernel。加载的数据不被并发修改，每个线程只写自己的 checksum。CPU 根据已知环布局计算精确的 `uint32_t` 模 2³² 校验和。

本机启动 24 个 block，但**block 数等于 SM 数不保证每个 SM 恰好运行一个 block**。这是小工作集、地址依赖的复用实验，时间包含加载等待、指令执行和调度；既不是单条 L1/L2 指令的精确延迟，也不是 GPU 峰值带宽测试。

合理假设是：当数据确实能在 L1 复用时，`.ca` 可能受益；`.cg` 的路径可能付出更高等待成本。这个假设需结合指令、工作集和计时核对，不能作为预设结论。

## 5. 实验 B/C：大数组顺序读取与写入

设备数组长度为 `STREAM_MIB × 2²⁰ / 4 + 37` 个元素，多出的 37 项用于覆盖非整 warp/block 的尾部。默认约 64 MiB，本机 L2 为 32 MiB；程序会打印实际 L2 容量，并提示工作集是否能整体放入 L2。

每个 block 有 256 个线程，grid 大小为 `4 × SM 数`。线程用 grid-stride loop 访问数组，同一 warp 同一步访问连续地址。

读取组仅改变 load cache operator，数据、线程布局、归约方式一致；每线程把加载值累加并写出 checksum。写入组仅改变 store cache operator，直接生成确定性整数模式，避免混入输入数组的读取流量。每次计时写入使用不同 generation，防止旧结果掩盖没有更新的数据。

每种 store 后启动一个依赖的验证 kernel，以 `.ca` 逐元素读回并检查模式。**写后可见性来自同一 stream 中 kernel 的顺序依赖，而非 `.wb/.cg/.cs/.wt` 的同步能力。** 这也对应原文关于 dependent grids 之间保持正确读回的要求。

本例的 streaming 组只有一个主要数据流。它能比较提示在该负载上的表现，但没有加入另一个需要保留的热点数据集，因此不直接证明 `.cs` 改善了其他数据的缓存命中率。

## 6. 实验 D：在真正的系统内存上使用 `.cv/.wt`

程序检查 `canMapHostMemory`，使用 `cudaHostAllocMapped` 分配约 256 KiB 的 host pinned memory，再通过 `cudaHostGetDevicePointer` 获取设备访问地址。

读测试执行两轮：CPU 在 GPU 空闲时写入不同 generation，GPU 分别使用默认、`.ca/.cg/.cs/.lu/.cv` 读取，kernel 完成后校验 checksum。写测试让 GPU 分别使用默认、`.wb/.cg/.cs/.wt` 写入不同 generation，CPU 在 `cudaDeviceSynchronize` 返回后逐元素验证。

这些测试证明示例可以在合法的 mapped 系统内存访问流程中执行，并得到正确数据。它们**没有测量每次 `.cv` 的 L2 失效次数或 `.wt` 的总线写事务**。由于每次都在完成同步后观察结果，其他 cache operators 同样可以通过校验；这并不否定 `.cv/.wt` 的特殊定义。

CPU 与 GPU 不并发访问同一对象，所以这里不要求通过 host-native atomics 建立跨处理器同步。请勿改成 CPU/GPU 对普通 flag 的并发轮询，再试图用 `.cv` 或 `.wt` 补救数据竞争。

## 7. 计时与结果解释

每种设备内存变体先预热 3 次，再采样默认 9 次；每一轮轮换起始算子，减轻固定顺序偏差。CUDA events 夹住单个 kernel，校验、分配、初始化和 CPU 参考计算均在计时区间之外。

原始 `SAMPLE` 行保留每次耗时，`RESULT` 输出最小值、中位数、最大值及逻辑吞吐：

```text
logical_GBps = logical_bytes / (median_ms × 10^6)
reuse logical_bytes = blocks × 8192 × passes
stream logical_bytes = element_count × 4
```

逻辑字节数不等于 DRAM 或 PCIe 的物理流量。reuse 会重复读取缓存内数据；读取 kernel 还会写 checksum，而写入 kernel 还会计算模式。events 只测其间的设备时间，但短 kernel 的区间仍可能受提交间隙、调度、频率与并发负载影响。

本例不显式清空缓存，不保证冷缓存状态。预热、前一个变体和写入验证 kernel 都会影响缓存状态；它属于重复运行条件下的对照。增大工作集也不等于“每次读取都来自显存”。

如果各算子的时间范围重叠，应避免宣布赢家。需要调查趋势时，先延长 kernel 内工作、单独运行测量，再查看实际生成指令；不要直接把一次加速归因于命中率。

## 8. 静态分析：确认测试对象确实存在

`bash commands/static/run.sh` 读取已有 PTX、生成 SASS，并执行 `inspect_sass.py`。脚本按本例的模板符号识别 6 个 load kernel 和 5 个 store kernel，输出对应指令形式；对于 6 个 reuse kernel，检查 backward branch 范围内是否含有 `LDG`。

```bash
bash commands/static/run.sh
```

该动作自动调用检查器，无须另外执行 Python 命令。当前配置的生成文件位于 `build/sm_89-release/cache_operators.ptx` 和 `cache_operators.sass`；检查输出在 `results/sm_89-release.log` 的 static 节。其他目标架构使用对应配置名。

这是一项防止测量无效的结构检查，仍需人工确认加载之间的寄存器依赖。反汇编格式或符号发生变化时脚本会失败，应检查文件后更新解析规则，不能忽略失败继续使用吞吐数字。

默认 PTX 指令与显式 `.ca/.wb` 可能生成不同的 SASS 文本；`.cs` 和 `.lu` 也可能使用不同编码。SASS 中的 `STRONG.GPU`、`STRONG.SYS` 等机器指令名称不应直接解读成 CUDA C++ acquire/release。源码层面的保证仍以 PTX 指令语义为准。

## 9. 本例能回答与尚未回答的问题

本例能核对每个提示的可编译性、生成指令、合法顺序下的结果，以及三类设备负载的执行时间差异。mapped memory 部分补齐 `.cv/.wt` 的适用地址场景。

本次环境未找到 `ncu`，所以 L1/L2 命中率、DRAM bytes、系统内存事务及精确 stall 原因均未采集。没有把耗时变化换算成这些指标。后续若使用 Nsight Compute，应按该工具及 GPU 实际提供的指标查询；分析器 replay 和缓存控制也可能改变本例的重复运行条件。

另外未覆盖 local memory 的 `.lu`、多 GPU、CPU/GPU 并发同步、持久化存储语义，以及与 §9.7.10.2 的 eviction priority hints 混用。详细实测与证据链接见 [RESULTS.md](RESULTS.md)。

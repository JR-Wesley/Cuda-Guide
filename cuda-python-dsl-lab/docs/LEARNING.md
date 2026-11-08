# GPU Kernel DSL 系统学习计划

更新日期：2026-11-06。实现与证据见 [逐项实验索引](experiments/README.md)：本轮复用现有环境，优先小规模功能验证，不执行大规模扫描或性能测试；原始计划和验收边界保留。

按每周 **6–8 小时、12 周**安排，以 SM89 单卡完成前 10 周，再在 H20 上完成 Hopper 专题。周数表示学习顺序，不绑定日历；未达到阶段验收时顺延，扩展任务可以放到第二轮。

学习对象是四种编程接口、三套工具链：Triton 与 Gluon 共用环境，CuTe DSL 和 cuTile Python 各自隔离。路线围绕同一组算子逐步深入：**Tile → Layout → 数据搬运 → MMA → 异步流水线 → 跨 DSL 分析**。

## 1. 当前起点与最终目标

### 已有基础

当前环境与证据见 [ENVIRONMENT.md](ENVIRONMENT.md)，创建和升级流程见 [SETUP.md](SETUP.md)。截至本次更新：

| 项目 | 当前状态 |
| --- | --- |
| 本机 | RTX 4060 Laptop，SM89，8188 MiB，Arch Linux / WSL2 |
| Python 环境 | Python 3.12.15；Triton 3.9.0 / Gluon、CuTe DSL 4.8.1、cuTile 1.7.0 |
| 基础 GPU 验证 | 四种 DSL 的本地 Vector Add 已通过，五组规模、每组两次 |
| 实验入口 | commands/ 下的 inspect/setup/check/run/learn/static/compare 独立脚本；各自 log/ 保存命令、退出码与日志 |
| 本轮新增 | 共享输入/误差记录；四接口小功能共 112 cases 通过；Gluon/CuTe 布局、基础 MMA 与离线 PTX/SASS 分析，见逐项索引 |
| 尚未完成 | GPU 稳态计时验证、完整 Profile、性能扫描/报告、CuTe 多 Stage、CUDA C++/库基线；详情见逐项状态 |
| H20 | 尚未在本学习项目中连接、采集环境或运行实验 |

现有示例属于环境冒烟测试，不等于已经独立掌握算子或完成官方教程。已有通过记录：[最终 GPU 验证](../runs/20261104T135724Z-run-hb2rwl3u/manifest.json)。正常开始学习无需重装环境。

### 最终能力与产出

| 能力 | 验收方式 |
| --- | --- |
| Kernel 设计 | 从数学定义推导 Tile、边界、线程/数据映射，独立实现并验证 |
| 性能分析 | 用计时、资源信息和硬件计数器支持结论，明确未测量项 |
| 架构编程 | 解释 MMA、数据搬运和同步依赖；在对应设备上验证异步流水线 |
| 编译器理解 | 对照程序、实际导出的 IR/PTX/SASS，区分显式控制与编译器决策 |

最终留下三项产出：可运行的算子实验、可复用的正确性与 benchmark 工具、带证据的硬件映射及性能笔记。第 10 周形成单 GPU 对照报告，第 12 周追加 Hopper 专题报告。

## 2. 十二周总览

| 周次 | 主要主题 | 本周核心产出 | 设备 |
| --- | --- | --- | --- |
| 1 | 实验规范、Triton 向量算子 | 共享输入与计时原型；Copy/Add/AXPY | SM89 |
| 2 | Triton Reduction、Softmax、基础 GEMM | 数值稳定的 Softmax 和带尾块处理的 GEMM | SM89 |
| 3 | Triton 参数优化、Transpose、Fusion | 至少两组受控优化及一份证据分析 | SM89 |
| 4 | Gluon 显式 Layout | 一维/二维元素映射图、Reduction 对照 | SM89 |
| 5 | Gluon Layout Conversion、Shared Memory、编译产物 | Transpose、搬运对照及 IR/PTX 分析 | SM89 |
| 6 | CuTe Shape/Stride、Layout Algebra | 索引推导、组合/分块验证、基础 Copy | SM89；部分索引练习可在 CPU 完成 |
| 7 | CuTe TiledCopy、MMA Atom | Global→Shared→Global Copy、最小 MMA 验证 | SM89 |
| 8 | CuTe Tile GEMM 与基础流水线 | 完整 GEMM、两组 Tile/Stage 对照 | SM89 |
| 9 | cuTile 与自动硬件映射 | Transpose、Reduction、Matmul 及配置对照 | SM89 |
| 10 | 跨 DSL 同算子比较 | Copy/Add、Transpose、FP16 GEMM 对照报告 | SM89 |
| 11 | TMA 与双缓冲 | 普通搬运、TMA、Pipeline 的逐步对照 | H20 / SM90 |
| 12 | Warp Specialization 与高级 GEMM | 同步依赖图、资源预算和流水线报告 | H20 / SM90 |

每周建议分配：阅读和手工推导 1–2 小时，实现与正确性验证 3 小时，分析与记录 2–3 小时。每周只推进一个主主题；先完成核心任务，再做扩展任务。工具链投入可粗略按 Triton/Gluon 45%、CuTe 35%、cuTile 20% 分配，公共 benchmark 与报告时间贯穿全程。

H20 暂不可用时，先以第 10 周为阶段终点；第 11–12 周可做源码阅读和依赖推导，但 GPU 验收保持未完成。此时可用剩余时间补 LayerNorm、Gluon GEMM 或编译器分析。

## 3. 统一实验方法

### 3.1 每个实验的五步流程

| 步骤 | 要做的事 | 留下的证据 |
| --- | --- | --- |
| 定义计算 | 写输入/输出、公式、dtype、布局、边界和误差要求 | 算子说明、理论字节数/FLOPs |
| 阅读实现 | 分清 kernel、host wrapper、编译常量、启动配置 | 官方版本/commit、关键代码注释 |
| 独立实现 | 先写最小正确版本，再加入优化 | 基线源码、结果校验、边界用例 |
| 受控实验 | 固定输入，一次改变一个因素，先正确性后计时 | 参数表、原始计时、必要的 Profile |
| 解释结论 | 对照 IR/PTX/SASS、资源信息和硬件模型 | 已证实结论、推断、待验证问题 |

### 3.2 共享算子与输入

所有 DSL 复用同一份 CPU 输入及参考结果；可以保存数组文件并记录 SHA256，也可以使用同一个生成器后转换到各框架。**相同 seed 不保证 PyTorch 与 NumPy 生成相同数据。** 当前 CuPy 示例与 PyTorch 示例的输入生成方式不同，第 1 周需要补齐共享数据层。

以下是起始测试集，先跑小规模，再按空闲显存扩大；形状不是性能保证。

| 算子 | 建议规模 | 主要边界与精度要求 |
| --- | --- | --- |
| Vector Copy/Add/AXPY | 保留 `1,255,256,4099,65536`；增加 `2^10,2^20,2^24` 及相邻非整除长度 | FP32 起步；验证 mask 与尾块；扩展 FP16 |
| Transpose | `256×256`、`257×509`、`1024×1024`；另选长宽不等形状 | 校验真实数据重排；必须产生连续输出，不能拿仅改 stride 的 view 计时 |
| Reduction/Softmax | 行数 `1,128`；行长 `127,128,1023,1024,4097`，选代表性组合 | 非二次幂、负值、较大幅值；明确 reduction 轴和累计类型 |
| GEMM | `(M,N,K)=(128,128,128),(512,512,512),(1024,1024,1024),(257,509,131)`；补一组瘦长矩阵 | FP16 输入/FP32 累计起步；覆盖 M/N/K 尾块，记录输出 dtype |

Copy/Transpose 应逐元素一致；加法、Reduction、Softmax、GEMM 按运算误差分别制定容差，记录最大绝对/相对误差。不要把 Vector Add 的 `1e-6` 直接套到 FP16 GEMM。Softmax 先测试有限输入，再单独定义 NaN/Inf 的支持范围；空张量等未支持输入应明确拒绝或记录约束。

参考结果从 CPU NumPy/PyTorch 或更高精度实现获得。GEMM 比较需统一输入、累计与输出精度、TF32/FP32 模式、融合策略、布局和 epilogue；PyTorch/cuBLAS 可作性能参考，但必须记录实际调用路径。混合精度或融合语义不同的结果单独分组。

### 3.3 计时与分析规范

1. 编译、张量分配、数据传输与结果校验放在稳态 kernel 计时之外；另外记录首次调用/JIT 和端到端耗时。
2. 统一使用同一设备、同一 CUDA stream 的 Event 或经核对的框架计时器。异步调用的 Python 返回时间不能作为 kernel 时间；使用多 stream 时建立显式依赖。
3. 起步可采用 10 次预热、每批 50 次执行、5 批测量，保存批次数据、中位数和波动范围。根据耗时调整后必须记录；批量事件时间除以执行次数应标为“批量启动平均时间”，不能写成 profiler 测得的单次 kernel 时间。
4. 记录 JIT 缓存状态、输入复用方式、GPU 温度/功耗状态和后台负载。交替运行对照配置，避免笔记本升温或动态频率造成单向偏差；不以修改系统时钟或权限作为默认步骤。
5. 参数扫描先做小范围单因素实验，再交叉组合。调优搜索与最终报告分开，保留全部候选及选择依据，避免只报一个偶然最快值。
6. 普通运行与 Nsight Compute 下的耗时分别存放。寄存器、spill、Shared Memory、occupancy、访存事务和缓存指标按工具实际结果记录；拿不到计数器就写“未测量：原因”。

FP32 Vector Add 的算法流量为 `D = 3 × N × 4 = 12N bytes`，因此 `B_eff = 12N / t_seconds`；除以 `10^9` 得 GB/s。它是有效带宽，不是实测 DRAM 带宽。GEMM 常用计算量 `2MNK`，`TFLOP/s = 2MNK / t_seconds / 10^12`；额外 epilogue 及不同数学语义须另行说明。

### 3.4 实验组织与记录格式

现有 `examples/` 保留环境冒烟示例。2026-11-05 已建立下列算子/公共目录，具体功能和未实现项见 [实验索引](experiments/README.md)；`hopper/` 尚未建立，不代表 H20 已实现：

```text
experiments/
  vector/       # reference、cases、各 DSL 实现
  transpose/
  reduction/
  gemm/
  hopper/
benchmarks/     # 公共计时、结果格式和汇总
docs/experiments/  # 按实验编号撰写分析
commands/      # 每动作独立脚本 + log/；源码单份，共享 build/
runs/          # 历史原始证据，不再写入
```

每条结果至少记录：实验 ID、设备/驱动/工具链版本、所用源码路径、原始命令、退出码、输入标识、shape/stride/dtype、精度模式、Tile/Warp/Stage、stream、warmup/repeat、计时范围、正确性及误差、时间统计、工具日志路径。报告中的每项优化都应链接基线和改动后的记录。

实验入口为 python3 commands/learn/run.py cpu|triton|gluon|cute|cutile|all，默认小功能验证；--ids 筛选计划编号，--bench / --compare-configs 显式开启计时/扫描。环境、安装、检查、Vector Add 分别使用 inspect/setup/check/run 独立动作。离线分析用 commands/static/run.py，跨 DSL 记录对照用 commands/compare/run.py。每个动作覆盖自己的 log/，不创建快照/哈希/日期目录，不修改历史 runs/。JIT 首次调用可能触发框架编译，属于允许的例外而非迁移缺项；见根 README 对首次调用、预热与计时的说明，不要求 AOT。

## 4. 第 1 周：实验基准与 Triton 基础

阅读：[Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) 中的 Vector Addition，结合 [本地示例](../examples/triton_vector_add.py) 和 [统一校验代码](../examples/torch_inputs.py)。

| 编号 | 核心任务 | 验收 |
| --- | --- | --- |
| E0.1 | 阅读环境记录；环境变化后重做 inspect/check/run | 能区分驱动、系统 Toolkit、wheel 编译器和 runtime；已有记录无需重复采集 |
| E0.2 | 逐行解释四个 Vector Add 的 host/kernel 边界 | 说明 JIT、启动、同步和 CPU 参考分别发生在哪里 |
| E0.3 | 建立共享输入、误差与计时格式 | 同一向量在四种 DSL 中使用相同数据；首次调用与稳态计时分开 |
| E0.4 | 检查 ncu、nvdisasm/cuobjdump 的可用性和架构支持 | 选一个 kernel 获取指令与 Profile；若工具受限，分别标记完成/未完成，不调整系统权限 |
| T1.1 | 独立实现 Copy、Add、AXPY | 覆盖尾块；解释 grid、program、逻辑 block 和实际线程数的关系 |
| T1.2 | 扫描 `BLOCK=128,256,512,1024`，再比较 `num_warps=4,8` | 输出参数表，先校验后计时；不把 `BLOCK=1024` 解释成 1024 个线程 |

本周产出：公共实验格式与一个向量算子报告。扩展任务：Scale、stride 输入，以及小规模 kernel 的启动成本分析。

## 5. 第 2 周：Triton Reduction、Softmax 与 GEMM

阅读官方 [Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) 和 [Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)。

| 编号 | 核心任务 | 验收 |
| --- | --- | --- |
| T1.3 | 写 Row Sum，再实现减最大值的稳定 Softmax | 解释 broadcast、reduction、padding 的中性元素；验证短行、长行与尾块 |
| T1.4 | 实现基础 FP16 GEMM | 正确处理 M/N/K 边界；解释 M/N/K Tile 和数据复用 |
| T1.5 | 比较两个行长度及两种 GEMM Tile | 记录资源和耗时，提出可验证的瓶颈假设 |

必须能回答：一行由多少 warp 协作？reduction 的通信发生在哪里？行长增加为何可能提高寄存器需求？增大 GEMM Tile 为什么不保证提速？寄存器、occupancy 和真实访存行为需要工具证据。

扩展任务：LayerNorm、FP32 GEMM 精度模式。不要把它们与首次实现 GEMM 同时设成必须完成项。

## 6. 第 3 周：Triton 受控优化

| 编号 | 核心任务 | 控制变量与证据 |
| --- | --- | --- |
| T2.1 | 实现普通与分块 Transpose | 固定 shape/dtype，改变 Tile/访问方式，分析访存合并 |
| T2.2 | 优化 Softmax 或 GEMM 中的一个 | 先扫描 warp 或 Tile，再扫描合法的 Stage；记录资源上限与失败配置 |
| T2.3 | 比较两步逐元素操作与融合实现 | 保持数学语义一致；分别报告整条算子链与单 kernel 时间 |
| T2.4 | 为至少两组基线/变体整理分析 | 每组有正确性、计时、生成代码；可用时补 Profile 并解释差异 |

读 [Triton Debugging](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html)。每个性能解释标清依据：文档语义、静态指令观察、硬件实测或尚待验证的假设。一次速度变化不能直接证明缓存命中率变化。

阶段验收：至少独立实现五个 kernel，包含向量算子、Softmax、Transpose、GEMM 和融合变体；有可复用 benchmark，能解释一次优化的收益及代价。工具缺失时保留部分完成状态，不能将“未 Profile”写成完成硬件归因。

## 7. 第 4 周：Gluon Layout 与线程映射

阅读 [Gluon 教程目录](https://triton-lang.org/main/getting-started/tutorials/gluon/index.html) 的 Introduction 和 [Tensor Layouts](https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html)。从 [本地 Gluon 示例](../examples/gluon_vector_add.py) 入手。

| 编号 | 核心任务 | 验收 |
| --- | --- | --- |
| G3.1 | 改写 Triton Add 和 Row Sum | 对比哪些算法部分相同、哪些布局决策需要显式写出 |
| G3.2 | 推导一维 `BlockedLayout([1],[32],[4],[0])` | 对 128 元素列出 warp/lane/寄存器槽位映射，再分析 256 元素情况 |
| G3.3 | 修改每线程元素数和二维 order | 先手工推导，再用输出验证；检查每个元素是否遗漏或重复 |
| G3.4 | 比较两个 Reduction Layout | 解释线程内、warp 内、warp 间的通信需求，并查生成指令 |

产出：一维与二维映射图、Layout 参数表、Triton/Gluon Reduction 对照。Gluon 的 layout 描述张量在硬件线程层次上的分布，后续 CuTe 的 layout 还涉及坐标到索引及组合关系，不能只按 API 名称互相替换。

扩展阅读：[Linear Layouts 论文](https://arxiv.org/abs/2505.23819)。先理解小型位映射例子，再研究编译器表示；本轮不要求修改编译器 pass。

## 8. 第 5 周：Gluon 数据搬运与编译器分析

| 编号 | 核心任务 | 验收 |
| --- | --- | --- |
| G3.5 | 用二维 Layout/Conversion 实现 Transpose | 对照两种映射，定位数据重排的位置 |
| G3.6 | 阅读并实现一个 SM89 适用的 Shared Memory / Async Copy 例子 | 说明普通搬运、异步提交、等待和线程同步的职责 |
| G3.7 | 比较单缓冲与一个双缓冲变体 | 先画出数据依赖，确认旧 buffer 何时可以重用，再检查正确性和资源变化 |
| G3.8 | 导出一个简单 kernel 的编译产物 | 关联 layout 操作、访存、shuffle/barrier 与源代码，不假设每种 layout conversion 都使用 Shared Memory |

资料：[Async Copy in Gluon](https://triton-lang.org/main/getting-started/tutorials/gluon/async-copy.html)。优先选择当前安装版本和 SM89 支持的路径；TMA 与 Hopper 专属操作留到第 11 周。

当前安装的 Triton 3.9.0 已核对支持下列调试开关。可在项目根目录针对现有 Gluon 示例导出产物：

```bash
dump_dir=$(mktemp -d "$PWD/.cache/gluon-ir.XXXXXX")
TRITON_KERNEL_DUMP=1 \
TRITON_ALWAYS_COMPILE=1 \
TRITON_DUMP_DIR="$dump_dir" \
TRITON_CACHE_DIR="$PWD/.cache/triton" \
  .venvs/triton/bin/python examples/gluon_vector_add.py
```

这是静态分析辅助命令，不是独立动作脚本的日志记录；正式实验时保存命令、退出码和导出文件。强制重编译运行不能用于稳态性能报告。按实际产物分析可用的 TTIR、Triton GPU IR、LLVM IR、PTX、cubin；不同前端/版本未必经过或暴露相同阶段，Gluon 缺少某一层文件不能直接认定编译失败。

扩展任务：补一个 SM89 适用的最小 Gluon MMA/GEMM，供第 10 周比较；复杂多阶段 GEMM 可延后。本周核心产出是搬运实验和一份编译产物分析。

## 9. 第 6 周：CuTe Layout Algebra

从 [CuTe DSL Quick Start](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html) 的 DSL Programming Model、CuTe Core、Educational Notebooks 导航阅读。官方示例须选择与本地 4.8.1 兼容的 tag/commit，记录来源；`main/latest` 只作为查找入口。

| 编号 | 核心任务 | 验收 |
| --- | --- | --- |
| C4.1 | 阅读 [本地 CuTe 示例](../examples/cute_vector_add.py)，解释 JIT 与 launch | 区分 Python 编译时值、运行时参数和 GPU 执行代码 |
| C4.2 | 构造 1D、row-major、column-major 和嵌套 Layout | 手算坐标→线性索引，并逐点验证 |
| C4.3 | 练习 composition、逻辑分块和 partition | 写清输入/输出域、shape/stride 与组合顺序 |
| C4.4 | 实现 Vector Copy 和基础 Transpose | 解释 layout 如何决定索引与线程数据分工，保留朴素基线 |

产出：至少四个索引映射练习、一个组合/分块示例和正确的 Copy/Transpose。不要只依据打印出的 layout 字符串判断实际访存效率。

## 10. 第 7 周：CuTe TiledCopy 与基础 MMA

| 编号 | 核心任务 | 验收 |
| --- | --- | --- |
| C4.5 | 实现 Global→Shared→Global Copy | 解释 Copy Atom、TiledCopy、线程 partition 与各次同步 |
| C4.6 | 改变两个合法 Copy Layout | 对比访问对齐、向量宽度、Shared Memory 分布；结合证据讨论 bank conflict |
| C4.7 | 阅读 MMA Atom / TiledMma，完成最小 MMA | 标明输入 fragment、累计 dtype、线程协作和目标架构，校验小矩阵结果 |

参考 [Warp-Level MMA 指南](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/mma/wmma_programming.html)，按当前版本筛选 Ampere/Ada 适用的部分。只有实际生成代码才能证明本次用了哪种 MMA；某个指南或 API 的名字不是指令证据。

本周只要求打通 Copy 与最小 MMA；完整 GEMM 放到下一周。SM89 不执行 Hopper WGMMA 或 Blackwell tcgen05 示例。

## 11. 第 8 周：CuTe GEMM 与基础流水线

| 编号 | 核心任务 | 验收 |
| --- | --- | --- |
| C4.8 | 组装普通 Tile GEMM | 识别 CTA/Warp Tile、Global/Shared/Register 分工、mainloop 与 epilogue |
| C4.9 | 补齐边界与参考校验 | 通过共享 GEMM 测试集，记录精度模式和尾块处理 |
| C4.10 | 比较两组 Tile，再比较合法 Stage 配置 | 保存资源、计时、同步依赖图；若资源超限保留失败配置和原因 |

验收重点：能沿源码追踪一个 Tile 从全局内存进入 MMA 再写回的过程，说明每个等待/同步保护了哪条依赖。Stage 更多不必然带来收益，可能提高 Shared Memory 或寄存器需求；实际结论必须有证据。

本周以 SM89 支持的搬运/流水线为范围，Hopper 的 TMA/WGMMA 延至第 11–12 周。扩展任务：更复杂 epilogue、多个 K 长度的流水线分析。

## 12. 第 9 周：cuTile 自动映射与 Matmul

阅读顺序：从 [cuTile Quickstart](https://docs.nvidia.com/cuda/cutile-python/quickstart.html) 导航到 Execution Model、Data Model、Operations、Memory Model，再读 [Performance Tuning](https://docs.nvidia.com/cuda/cutile-python/performance.html) 和 [Debugging](https://docs.nvidia.com/cuda/cutile-python/debugging.html)。

| 编号 | 核心任务 | 验收 |
| --- | --- | --- |
| U5.1 | 复述 [本地 Vector Add](../examples/cutile_vector_add.py) 的 Array/Tile 关系 | 解释加载、padding、写回边界及 launch，而不是将 Tile 大小当作线程数 |
| U5.2 | 实现 Transpose 和 Row Sum | 使用共享输入，覆盖非整除维度，检查布局/归约语义 |
| U5.3 | 运行匹配版本的 Matmul，再写最小变体 | 能说明 Tile 与 K 循环、累计类型、尾块处理和数据复用 |
| U5.4 | 改变两个 Tile 配置，观察编译输出 | 对照生成指令与资源，区分已观察现象和对自动映射的推断 |

核心目标是通过外部证据理解编译器承担的映射工作，不要求逆推出未公开的所有决策。Performance Hint 必须满足语义前提；错误的对齐或整除假设不能作为优化。

扩展任务：autotuning、Softmax、更多矩阵形状。Hopper 支持按实际 `cuda-tile`、TileIR 编译器及驱动版本重新验证，不能沿用本机 SM89 的通过结论。

## 13. 第 10 周：跨 DSL 对照报告

以前三周形成的公共 benchmark 为基础，复用已有实现；本周主要补齐缺项、控制实验条件和写报告。

| 对照算子 | 最低覆盖要求 | 比较重点 |
| --- | --- | --- |
| Vector Copy 或 Add | Triton、Gluon、CuTe、cuTile 四种接口 | 启动成本、逻辑 Tile、访存及程序表达 |
| Transpose | 四种接口；CuTe 可升级到 TiledCopy 版本 | 数据重排、线程映射、Shared Memory 使用 |
| FP16 GEMM | Triton、CuTe、cuTile 三套工具链，加库参考 | Tile/MMA、精度、资源与形状敏感性；Gluon GEMM 为扩展覆盖 |

为匹配每周时间预算，Gluon GEMM 不作为首轮报告完成条件；若第 5 周已实现则纳入比较。报告必须列覆盖矩阵，缺项写“未实现/未验证”，不能将三套 GEMM 写成四种接口全部比较。

每个算子至少比较一个规则形状和一个非整除或瘦长形状，配置先通过正确性。保留 CUDA C++ 基线；若本项目尚未准备该算子的 C++ 基线，补一个最小实现或明确报告缺项，不借用其他项目的历史耗时。GEMM 添加 cuBLAS 或明确记录路径的框架参考。

报告应包含六个维度：编程复杂度、Layout 控制、线程/内存映射、生成指令、性能随 shape 的变化、架构移植成本。对每个主要结论链接源码与原始记录；不以少量输入宣布某种 DSL 普遍更快。

阶段验收：另一人按记录可以重建环境、运行相同输入和配置、核对结果，并知道哪些指标尚未测量。

## 14. 第 11 周：H20 TMA 与双缓冲

开始条件：可访问 H20；重新采集设备、驱动、Toolkit、DSL 版本；基础示例在 SM90 上通过；所选官方示例明确支持该架构。迁移流程见 [SETUP.md](SETUP.md)，不要直接复制带绝对路径的 venv。

| 编号 | 核心任务 | 验收 |
| --- | --- | --- |
| H7.1 | 建立普通 2D Global→Shared→Global Copy 基线 | 输入、输出布局与参考结果固定 |
| H7.2 | 在 Gluon 或 CuTe 中实现 TMA Copy | 写清 descriptor、对齐/布局条件、barrier 与数据可见性要求 |
| H7.3 | 加入双缓冲与分阶段搬运 | 画 producer/consumer 时间线，说明 buffer 何时可读、何时可重用 |
| H7.4 | 比较基线、TMA、双缓冲 | 分离 descriptor/初始化与稳态 kernel 时间，记录资源和不同规模表现 |

资料：[Gluon TMA](https://triton-lang.org/main/getting-started/tutorials/gluon/tma.html) 及匹配版本的 CUTLASS Hopper 示例。首轮选一种 DSL 完整实现，第二种作源码对照；有余力再实现双份，避免两周内同时重写复杂流水线。

同步说明必须分别覆盖：异步操作的完成、线程之间的先后关系、内存可见性、必要的 proxy 同步。写出所用 API/指令的作用域及前提；不能靠 `asm volatile`、缓存提示或一次运行正确证明同步正确性。

## 15. 第 12 周：Warp Specialization 与高级 GEMM

阅读 [Gluon Warp Specialization](https://triton-lang.org/main/getting-started/tutorials/gluon/warp-specialization.html)。该教程说明 Gluon 的 warp specialization 支持 Hopper 及更新架构，但页面也包含 Blackwell Tensor Memory/tcgen05 示例；**H20 只运行明确支持 SM90 的部分**，其余用于阅读或留待对应硬件验证。

| 编号 | 核心任务 | 验收 |
| --- | --- | --- |
| H7.5 | 分析并运行一个 SM90 普通流水线 GEMM | 标注 load/compute/store、MMA 路径和所有跨阶段依赖 |
| H7.6 | 对照一个 SM90 producer/consumer 实现 | 解释各 warp 的职责及同步协议，验证匹配的输入/精度 |
| H7.7 | 改变一项合法的 Stage 或角色资源配置 | 记录寄存器、Shared Memory、barrier 和耗时；确认不是遗漏同步换取速度 |
| H7.8 | 对照 Gluon 与 CuTe 的表达方式 | 用依赖图和代码位置说明控制能力；明确哪些路径实际运行过 |

验收问题：warp specialization 和普通 software pipelining 的区别是什么？哪些操作真正重叠？多一个 producer warp 或更深 pipeline 为何可能变慢？寄存器预算、Shared Memory 和同步成本如何限制设计？

如受版本或设备限制不能运行目标高级 GEMM，保留第 11 周 Copy/Pipeline 成果，并将对应 GEMM/warp specialization 验收标为未完成；阅读 Blackwell 示例不代替 H20 实测。

## 16. 学习进度：8 / 18 项已完成

以下按证据更新，不按周数自动勾选。“完成”表示本条所列交付已满足；本轮范围为小规模功能，不表示整周的计时/Profile 也完成。证据和逐项限制见 [实验索引](experiments/README.md)。工具或设备受限时保留未勾选状态。

### 实验基础

- [x] 三套环境、四种 DSL 已通过 GPU Vector Add 校验；证据见本页当前起点。
- [x] 建立共享输入、正确性校验与统一 benchmark 格式；GPU 功能通过，事件计时仅实现及 CPU 替身测试，本轮未测 GPU 稳态性能。
- [ ] 完成一次指令检查和完整 kernel Profile，解释计时范围及主要指标。指令检查已做；2026-11-06 已确认 ncu 2026.3.1.0 可从 PATH 调用（[环境记录](ENVIRONMENT.md)），但当前采集权限/兼容性尚未验证，完整 Profile 未完成。

### Triton

- [x] 独立完成 Vector、Softmax、Transpose、GEMM 和 Fusion；小规模与尾块功能通过。
- [ ] 完成 Block/Warp/Tile 参数扫描并保留正确性与原始计时。
- [ ] 结合生成代码和 Profile 解释一次优化及其代价。

### Gluon

- [x] 手工推导并验证 BlockedLayout 的 warp/lane/元素映射；10 组 CPU 映射与 GPU ownership probes 通过。
- [ ] 完成二维 Layout、Layout Conversion 和搬运对照实验。功能及静态对照已做，计时/sanitizer 未做。
- [x] 导出并解释一个 kernel 的主要编译产物；真实 IR/PTX/cubin/SASS 和静态解释已保留。

### CuTe DSL

- [x] 完成 Shape/Stride、Layout 组合和 partition 的索引验证；17 项检查、4468 索引通过。
- [x] 完成 TiledCopy 和基础 MMA 的正确性实验；32/128-bit copy 和最小 MMA 小规模通过，实际指令已检查。
- [ ] 实现普通 Tile GEMM，并分析至少两种配置。同步单 Stage 功能通过；可选 Tile 对照未运行，多 Stage 未实现。

### cuTile 与横向比较

- [x] 完成 cuTile Transpose、Reduction 和 Matmul；本地最小实现规则/尾块通过，匹配版本官方完整示例另待运行。
- [ ] 通过配置变化和生成代码分析自动映射，明确推断边界。
- [ ] 按第 10 周覆盖要求完成可复现对照报告，并列出缺项。

### H20 高级专题

- [ ] 完成普通 Copy 与 TMA Copy 的受控实验。
- [ ] 完成双缓冲/Pipeline 并写出同步依赖与可见性要求。
- [ ] 完成 SM90 warp specialization 对照与资源预算分析。

## 17. 十二周之后

选择一个方向持续推进，不必继续横向增加 DSL：

| 方向 | 起始项目 | 前置条件 |
| --- | --- | --- |
| GPU 编译器 | Layout Conversion、Linear Layout、一个小型 pass 分析 | 第 4–5 周的映射与 IR 分析 |
| 高性能算子 | FlashAttention、Grouped GEMM、低精度 GEMM | GEMM、数值精度、流水线 |
| 数据搬运 | Gather/Scatter、Transpose、TMA | 布局、访存、同步 |
| 分布式 Kernel / MoE EP | Token Packing、Dispatch/Combine、计算通信融合 | 单 GPU 基础完成后，另行准备多 GPU 环境与通信验证 |

官方文档入口于 2026-11-04 核对。在线 `main/latest` 会变化；每次实验记录实际教程 commit、包版本和设备前提。每周至少留下一个可重复运行的实验和一份短分析，再进入下一主题。

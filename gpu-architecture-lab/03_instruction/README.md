# 03 · Language / Registers / Instruction / Dependency

本主题承接语言特性、设备函数与寄存器，并横跨 M3 与后续所有 microbenchmark。先理解源码契约和编译资源；当前优先补充以小例子为基础的指令分析，短依赖链、ILP/TLP 与延迟/吞吐测量保留为另一条后续路线。

## 五类 SASS 指令分析：先规划，再逐课展开

新增 [02_sass_atlas](02_sass_atlas/README.md)，按 Memory、Address & Integer、Synchronization、Control Flow、Data Movement 五类建立跨专题索引。阅读 [案例与目录规划](02_sass_atlas/PLAN.md) 查看现有源码、待补最小例子、架构限制和阶段验收；用 [分析卡](02_sass_atlas/CARD_TEMPLATE.md) 分别记录语义、操作数、PTX/SASS 映射、管线依据与未测指标。

**当前只完成规划和源码盘点，没有新增可执行专题或 GPU 实测。** 先单操作/索引复制，再整数位操作与控制流、shared/原子/同步，最后异步搬运与 tensor-map TMA。已有特性源码保持原位，不按 opcode 复制工程；默认只做编译、静态阅读和少量正确性验证，不启动大规模性能测试。复用已有独立动作脚本，实施前核对依赖和目标配置；入口存在不等于本轮已经编译或运行通过。

## 语言限定符与设备函数

| 概念 | 编程语义 | 硬件映射的检查点 |
| --- | --- | --- |
| `__global__` | kernel 入口，按 launch 配置执行多个线程实例；设备侧启动另有动态并行条件 | grid/block、资源元数据、实际执行目标 |
| `__device__` / `__host__` | 指定函数的执行侧；组合限定符可生成两侧版本 | 设备函数不等于新 kernel，不隐含并发或同步；是否内联看产物 |
| `__shared__` / `__constant__` | 指定相应存储属性与访问规则 | shared 容量、bank、生命周期；constant 访问模式与实际加载形式 |
| 自动局部变量 | C++ 对象的作用域与生命周期 | 编译器可能使用寄存器、local、常量折叠或直接消去，不能按变量数数寄存器 |
| `const` / `__restrict__` | 通过该指针不可写 / 调用方提供不别名承诺 | 可能改变加载和优化；不保证实际只读生命周期，也不是同步 |
| `volatile` / `asm volatile` | 各自语言或汇编接口的编译约束 | 不替代 atomics、fence、barrier；目标指令必须检查 SASS |

模板可表达编译期类型、布局或立即数条件，但会产生不同实例；比较性能时记录实例和编译参数。跨翻译单元设备调用需按实际项目核对 separable compilation、device link 与优化设置，不为文档示意杜撰统一构建命令。内联、循环展开和链接期优化都可能改变指令及寄存器分配。

[Load/Store Intrinsics](../02_memory/03_load_store_intrinsics/README.md) 已有真实重载与 `const`/`__restrict__` 探针；它们是本节的可执行素材，不代表全部语言特性已独立实现。

## 已有 setmaxnreg 专题

[01_setmaxnreg](01_setmaxnreg/README.md) 研究 CTA 内 warpgroup 的寄存器配额交换、消费者寄存器范围及 spill。先核对其初始化配额、warpgroup 一致参与、显式同步与回收预算，再理解生成代码。

- 静态寄存器元数据、编译约束、某阶段动态配额和实际驻留率是不同概念；归还到 CTA 池不意味着另一个 block 可因此驻留。
- 历史 `sm_89` 记录只能运行固定基线；动态版本交叉编译/静态观察不等于真实执行。普通 `sm_90` 不等于 `sm_90a`，其他目标按专题支持表和安装工具链核对。
- 本例没有完整 TMA/WGMMA 生产者消费者流水线；不能把寄存器结果当作 Tensor Core GEMM 加速证据。

命令、`RESULTS.md` 和调试记录以专题 README 为准，没有统一构建入口。

## 保留的指令微基准规划（不属于当前指令图谱阶段）

计划从 `integer`、`fp32`、`dependency`、`instruction_latency` 四类小实验开始。每组固定计算结果和主要指令数，改变依赖关系、独立链数量或展开程度；同时查看寄存器用量、PTX/SASS 与可用时的调度/warp 状态指标。静态反汇编只能说明生成了哪些指令，不能单独给出某条指令的实际延迟或流水线吞吐。

复用 [编译追踪](../../cuda-sdk-lab/experiments/compile/README.md)与 [`cuobjdump` 教程](../../cuda-sdk-lab/CUOBJDUMP.md)的方法。上述四类独立延迟/吞吐扫描仍无结果；已有 setmaxnreg 记录不替代它们。延迟、吞吐、ILP/TLP 与资源取舍的模型见 [体系结构基础](../ARCHITECTURE.md)，修改测量循环后须按 [实验规范](../EXPERIMENT_PROTOCOL.md) 检查目标指令与依赖链未被优化掉。

# CUDA Toolkit 工具使用实验

本目录的目的不是建立大型 CUDA 应用或性能框架，而是用小程序看懂 **编译、链接、二进制检查、加载、诊断与采集**。每个可执行动作都有自己的短脚本、README 和就地日志；总体概念仍集中在 [Toolkit 导览](TOOLKIT_GUIDE.md)，不再为每个工具另建一份顶层总论。

## 1. 先建立问题与工具的对应

必要基础：函数、数组、终端当前目录。host 是 CPU 侧，device 是 GPU 侧；编译生成文件，加载准备执行，运行才会启动 GPU。静态指令不证明结果正确，分析器报告也不能替代普通运行校验。

| 想弄懂的问题 | 阅读入口 |
| --- | --- |
| Toolkit 各组件做什么，CUDA 程序怎样走完生命周期？ | [Toolkit 导览](TOOLKIT_GUIDE.md)，含官方参考及 nvdisasm 学习路线落实情况。 |
| 先学哪些动作，已有结果在哪？ | [工具与证据索引](LEARNING_FRAMEWORK.md)，链接每个功能的文档和状态。 |
| nvcc 怎样调用前端、ptxas、nvlink 和主机链接器？ | [编译课程](experiments/compile/README.md) / [NVCC 导航](NVCC_COMPILATION.md)。 |
| PTX、cubin、fatbin 中装了什么？ | [二进制课程](experiments/binary/README.md) / [cuobjdump 导航](CUOBJDUMP.md)。 |
| 如何检查正常程序与故意错误？ | [正常 memcheck](experiments/validation/README.md)、[故障案例导航](experiments/sanitizer/README.md)。 |

## 2. 第一个闭环：预测、编译、运行、解释

下面手动进入各步骤注明的目录。每份动作 README **先给底层命令与参数解释，再给记录脚本**；不要从工作区根目录直接调用依赖相对路径的入口。`sm_89` 是本次设备目标，换机器须重新确认。

1. **确认环境**：在 `experiments/environment/` 阅读 [说明](experiments/environment/README.md)，执行 `bash check.sh`。分清 Toolkit、驱动和计算能力。
2. **读小输入**：打开 [共享源码](examples/vector_transpose.cu) 的 `vector_add`。先手算四项，再预测 257 个元素会启动两块、共 512 个线程，第二块只有一个有效线程。
3. **只编译**：在 `experiments/build/` 按 [构建说明](experiments/build/README.md) 执行 `CUDA_ARCH=sm_89 bash run.sh`。动态链接 Runtime，共享可执行文件仅约 44 KiB。
4. **只运行**：在 `experiments/execution/` 按 [运行说明](experiments/execution/README.md) 执行 `CUDA_ARCH=sm_89 N=257 REPEAT=1 bash run.sh`。本次 `correct=yes`、checksum `3508.375000`；`N=4` 的校验和 `2.25` 是另一个练习预期，不混作本次日志。
5. **选择一个后续问题**：想查访问错误就进入 validation；想看时间线就进入 nsys；想读单 kernel 指标就进入 ncu。每个动作单独执行，不存在 `all` 调度器。

普通运行的 event 时间、分析器重放时间、报告中的 kernel 时间口径不同，不直接相加或比较倍数。小输入用于解释工具，不代表峰值带宽与稳定性能。

## 3. 单功能动作分组

所有动作按 `experiments/<功能>/run.sh` 组织，环境查询使用 `check.sh`。本表是导航，不是自动串联脚本；精确输入、命令、日志解释及练习见链接。

| 学习单元 | 当前独立动作 |
| --- | --- |
| 环境、共享构建、普通运行 | [environment](experiments/environment/README.md)、[build](experiments/build/README.md)、[execution](experiments/execution/README.md) |
| 编译链观察与输出 | [dryrun](experiments/compile-dryrun/README.md)、[PTX](experiments/compile-ptx/README.md)、[cubin](experiments/compile-cubin/README.md)、[fatbin](experiments/compile-fatbin/README.md)、[object](experiments/compile-object/README.md)、[主机链接](experiments/compile-host-link/README.md) |
| 跨文件设备代码 | [RDC 编译](experiments/compile-rdc-compile/README.md)、[设备链接](experiments/compile-rdc-device-link/README.md)、[主机链接](experiments/compile-rdc-host-link/README.md)；[LTO 编译](experiments/compile-lto-compile/README.md)、[设备链接](experiments/compile-lto-device-link/README.md)、[主机链接](experiments/compile-lto-host-link/README.md) |
| 二进制检查 | [镜像清单](experiments/binary-list/README.md)、[PTX](experiments/binary-ptx/README.md)、[SASS](experiments/binary-sass/README.md)、[资源](experiments/binary-resources/README.md)、[ELF/符号](experiments/binary-elf/README.md) |
| nvdisasm | [SASS/行号](experiments/nvdisasm-sass/README.md)、[CFG](experiments/nvdisasm-cfg/README.md)、[寄存器活跃区间](experiments/nvdisasm-live-ranges/README.md) |
| 正常程序检查 | [memcheck](experiments/validation/README.md)、[racecheck](experiments/validation-racecheck/README.md) |
| 故意错误教学 | [调试构建](experiments/sanitizer-build/README.md)、[SASS](experiments/sanitizer-static/README.md)、[memcheck](experiments/sanitizer-memcheck/README.md)、[racecheck](experiments/sanitizer-racecheck/README.md)、[initcheck](experiments/sanitizer-initcheck/README.md)、[synccheck](experiments/sanitizer-synccheck/README.md) |
| 加载与 JIT | [Runtime 构建](experiments/module-build/README.md) / [运行](experiments/module-loading/README.md)；[Driver 构建](experiments/module-driver-build/README.md)；[cubin 构建](experiments/module-cubin-build/README.md) / [加载](experiments/module-driver-cubin/README.md)；[PTX 构建](experiments/module-ptx-build/README.md) / [加载](experiments/module-driver-ptx/README.md) |
| Nsight Compute | [基本采集](experiments/ncu/README.md)、[内存采集](experiments/ncu-memory/README.md)、[只读报告](experiments/ncu-report/README.md) |
| Nsight Systems | [时间线采集](experiments/nsys/README.md)、[只做统计](experiments/nsys-stats/README.md) |

`compile/`、`binary/`、`static/`、`sanitizer/` 保留源码或导航，不再提供聚合运行入口。编译课程不持久化大预处理文本或 `--keep` 全量中间文件；分析器只选少量 section、一个目标启动，不使用全量采集。

## 4. 本次真实结果与未通过项

2026-11-08 本轮记录：RTX 4060 Laptop / CC 8.9 / 驱动 596.21；nvcc 13.4.92；ncu 2026.3.1.0；nsys 2026.3.2。详情在 [environment](experiments/environment/README.md) 和各动作 README。

- **编译与静态检查**：12 个编译/链接教学动作、5 个二进制动作、3 个 nvdisasm 动作均执行成功；不将编译或反汇编成功视为 GPU 校验。
- **运行与正确性**：共享向量程序正确；正常 tiled/65 输入通过 memcheck 和 racecheck；四类故障检查均保留原始退出码 86，并匹配对应错误诊断。
- **加载**：Runtime 和 Driver-cubin 的 64 元素校验成功；**Driver-PTX 失败**：PTX `.version 9.4` 超过当前驱动 JIT 所接受的 9.2，API 错误 222、程序退出 1。见 [原始失败与解释](experiments/module-driver-ptx/README.md)。未修改驱动或篡改 PTX 头。
- **性能工具**：ncu 基本/内存采集成功，分别 8/32 次重放；`.ncu-repz` 可独立导入。nsys 采集和独立统计成功，统计得到两次 `vector_add`。这些是工具使用验证，不是优化效果证明。
- **未执行**：Graphviz 图片渲染（`dot` 未找到）；NVRTC/nvJitLink、cuda-gdb、NVTX 等仍只是总体文档的后续方向，不声称已经补齐这些实验。

本轮 LTO 初次参数组合被编译器拒绝、ncu 首次后缀检查错误的证据，都以有含义的固定名称留在各自动作 `log/`；最终正常路径另由当前 `status.tsv` 标识，不把失败改写成成功。

## 5. 产物、日志与历史清理

```text
examples/vector_transpose.cu       # 一份共享计算输入
experiments/compile/*.cu          # 编译课程输入，不复制到各动作
experiments/module-loading/*      # 小型加载源码
experiments/sanitizer/cases.cu     # 共享故障源码
build/sm_89/
  vector_transpose                # 44,984 字节，动态 cudart
  compile/artifacts/             # 小型 PTX/cubin/fatbin/object/可执行文件
  module-loading/                # runtime、driver、image.cubin、image.ptx
  sanitizer/cases
experiments/<功能>/
  run.sh                         # 只处理一个动作
  README.md                      # 背景 → 命令 → 真实日志解释 → 练习
  log/                           # 本动作命令、输出、状态、报告
```

本次编译产物合计 **290,431 字节**；不生成大应用、持久化 `.ii`、源码快照或按动作重复编译目录。`experiments/record.sh` 只提供小型命令记录与缺输入检查，不调度其他动作，不要求先读懂它才能学习工具。

**清理范围**：盘点时 SDK 下仅有旧 environment 日志，已清理后重新采集；未发现旧 `runs/`、`results/`、`history/` 或大型产物。删除被拆分替代的当前聚合脚本，移除已替换的 raw CSV 阅读日志，不新增历史归档。没有删除工作区其他项目或来源不明的文件。

重跑只覆盖当前动作同名日志/报告；源码或参数改变后显式重编译，采集更新后显式重读报告。失败构建删除其部分输出；分析不隐式编译。报告保留在产生它的动作，nsys 统计导出的 SQLite 在 `nsys-stats/log/`，与采集报告分离。

## 6. 维护验证

本轮所有动作均实际执行，已知 PTX 失败与预期故障如上。脚本语法、本地文档链接及 CPU fake-tool 回归测试另作工程检查，不能冒充 GPU 结果。

```bash
# 当前目录：cuda-sdk-lab/
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_actions.py -v
```

测试仅在临时目录模拟工具，覆盖独立动作边界、缺输入、失败清理、报告读取与负例诊断断言。没有统一“构建+运行+采集”入口。

官方参考：[CUDA Toolkit](https://docs.nvidia.com/cuda/)、[CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)、[Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。编程语义与体系结构实验另见 [GPU 体系结构实验室](../gpu-architecture-lab/README.md)。

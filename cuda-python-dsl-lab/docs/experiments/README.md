# 学习计划逐项实现与证据索引

更新：2026-11-06（新增 ncu 版本检查；不改变原 GPU 功能结果）。对应 [LEARNING.md](../LEARNING.md) 原编号，保留原计划不重新编号。

**本轮使用现有三套 venv，仅验证小规模功能。** 四个 GPU suite 共 **112/112 cases 通过**，每个两次 poison/check；另有 10 组 Gluon GPU ownership probes。没有安装/升级依赖，没有大规模测试、性能扫描或 GPU benchmark。

## 使用方法

从 `cuda-python-dsl-lab/` 运行，以下命令覆盖本动作 commands/learn/log/，不写历史 runs/；需要保留对照时先自行保存：

```bash
python3 commands/learn/run.py cpu
python3 commands/learn/run.py triton
python3 commands/learn/run.py gluon
python3 commands/learn/run.py cute
python3 commands/learn/run.py cutile
# 或按上述 DSL 顺序串行执行小功能集
bash tests/test_small.sh
# 只重跑一个计划项，不需要整套复测
python3 commands/learn/run.py triton --ids T1.4,T2.3
python3 commands/learn/run.py gluon --ids G3.7
```

`learn` 不调用 setup。默认无 benchmark、无可选配置扫描；只有显式 `--bench` 或 `--compare-configs` 才启用。后者在 Triton/CuTe/cuTile 选择小配置目录，不自动扩到大输入；本轮未执行。原 `run all` 仍仅是 Vector Add 环境冒烟。

每个 case 的 `[序号]-[ID]` 目录与 `results.json` 一一对应；同一计划 ID 可以包含多个 shape/config（例如 T1.1）。命令、退出码、输入、参考、实际输出、误差及可用编译产物写入本动作 log/，不自动复制源码或计算制品哈希。默认功能 PASS 不等于整项性能/学习验收，细节见 [基础记录协议](01-foundation.md)。

## 本次原始证据

| 标记 | 范围 | 结果及入口 |
| --- | --- | --- |
| B | 初轮 CPU 测试/布局 | [manifest](../../runs/20261105T060401Z-learn-fi68rcot/manifest.json)：11 单元测试、10 Gluon CPU 映射、17 CuTe 检查/4468 索引通过 |
| T | Triton | [40/40 results](../../runs/20261105T060430Z-learn-ne9a7tkh/triton/results.json)、[命令与快照](../../runs/20261105T060430Z-learn-ne9a7tkh/manifest.json) |
| G | Gluon | [34/34 results](../../runs/20261105T060458Z-learn-cbx4cfro/gluon/results.json)、[10/10 ownership probes](../../runs/20261105T060458Z-learn-cbx4cfro/G3.2-layouts/summary.json)、[manifest](../../runs/20261105T060458Z-learn-cbx4cfro/manifest.json) |
| C | CuTe | [23/23 results](../../runs/20261105T060524Z-learn-9jj6_yq0/cute/results.json)、[manifest](../../runs/20261105T060524Z-learn-9jj6_yq0/manifest.json) |
| U | cuTile | [15/15 results](../../runs/20261105T060543Z-learn-osusvmpm/cutile/results.json)、[manifest](../../runs/20261105T060543Z-learn-osusvmpm/manifest.json) |
| S | 静态指令 | [修正后的离线分析](../../runs/20261105T061056Z-static-3xunxkwj/analysis-v2/SUMMARY.md)：19 cubin、39 工具命令退出 0；原始记录未改写 |
| X | 跨 DSL 功能 | [CPU 审计](../../runs/20261105-functional-comparison/SUMMARY.md)：15 共享算子/shape 组，输入 hash 一致，原功能验证通过 |
| R | 最终 CPU 回归 | [manifest](../../runs/20261105T062844Z-learn-ew2vqklq/manifest.json)：14 单元测试通过，含无写入检测、失败证据保存、源码 drift 与不覆盖记录；布局检查再次通过 |
| R-G | 最终局部 GPU 复查 | [G3.7 4/4](../../runs/20261105T062914Z-learn-o0vx6clm/gluon/results.json)：验证计划前缀筛选与新版 runner；未重复全套 |
| F | 保留的开发失败 | [manifest](../../runs/20261105T062806Z-learn-j2fx_ucp/manifest.json)：新增测试缩进错误导致 CPU 导入失败，已修复并由 R 验证；不删除/改写失败日志 |
| N | ncu 重新安装后检查 | [manifest](../../runs/20261106T140618Z-ncu-check-rx5bgfry/manifest.json)、[原始日志](../../runs/20261106T140618Z-ncu-check-rx5bgfry/tools.log)：PATH 中 `/usr/sbin/ncu`，版本 2026.3.1.0，`--version` 退出 0；未进行新的采集 |

Gluon 进程退出时出现 nanobind 引用计数泄漏警告；原日志保留，进程退出码 0。未通过升级包或隐藏警告处理。

## 按顺序一一对应

“功能通过”只覆盖当前小数据和配置；“部分”明确尚缺的验收。表中证据标记对应上表链接，各专题包含源码链接、数学定义、同步/布局说明和复现命令。

### 第 1 周：基础与向量

| ID | 实现/脚本 | 本轮结果与未完成项 |
| --- | --- | --- |
| E0.1 | `scripts/inspect.sh`、`scripts/check.py` | B/T/G/C/U：现有环境与 SM89 重新确认 |
| E0.2 | [四个示例逐段说明](01-foundation.md#e02-原有四个-vector-add-的逐段边界) | host/JIT/kernel、同步与 CPU 参考边界已记录 |
| E0.3 | `benchmarks/core.py`、`runner.py`；`tests/test_core.py`、`test_runner.py` | B/T/G/C/U/X：共享输入与误差记录通过；事件计时实现存在但 GPU 路径未实测 |
| E0.4 | `scripts/learning_tools.py`、`learning_static.py` | S/N：PTX/SASS 指令检查完成；ncu 2026.3.1.0 已可用，当前采集权限/兼容性未验证，完整 Profile 未完成 |
| T1.1 | [Triton vector](../../experiments/vector/triton_ops.py)、`triton_suite.py` | T：Copy/Add/AXPY、整块和尾部通过 |
| T1.2 | `triton_suite.py --scan` | 基线功能通过；可选 BLOCK/warp 扫描未执行，无计时 |

### 第 2–3 周：Triton 算子与受控变体

详细说明：[02-triton.md](02-triton.md)。

| ID | 实现 | 本轮结果与未完成项 |
| --- | --- | --- |
| T1.3 | `experiments/reduction/triton_ops.py` | T：Row Sum、稳定 Softmax 短/长/尾行通过 |
| T1.4 | `experiments/gemm/triton_ops.py` | T/S：FP16 GEMM 规则及 M/N/K 尾块通过，真实 MMA 指令 |
| T1.5 | `triton_suite.py` 两行长、两 M Tile | T/S：功能和静态资源记录；无计时/瓶颈实测 |
| T2.1 | `experiments/transpose/triton_ops.py` | T/S：朴素/分块连续输出通过，保留两份编译结果 |
| T2.2 | `triton_suite.py --scan` | 基线功能通过；warp/stage 候选未跑，不宣称优化收益 |
| T2.3 | `experiments/vector/triton_ops.py` | T/S：两步/融合数学语义一致；未测整链或单 kernel 时间 |
| T2.4 | [两组静态对照](06-static.md) | T/S：正确性、源码、PTX/SASS 已有；无性能/Profile 归因 |

### 第 4–5 周：Gluon

详细说明：[03-gluon.md](03-gluon.md)；suite：`experiments/gluon_suite.py`。

| ID | 实现 | 本轮结果与未完成项 |
| --- | --- | --- |
| G3.1 | `vector/gluon_ops.py`、`reduction/gluon_ops.py` | G：Add/Row Sum 通过 |
| G3.2 | `experiments/gluon_layouts.py` | B/G：128/256 元素 warp/lane/逻辑槽推导，CPU/GPU 所有权验证 |
| G3.3 | 同上与 SPT=2 Add | G：1D/2D order 映射、无遗漏/重复检查与 GPU probe 通过 |
| G3.4 | `reduction/gluon_ops.py` 两布局 | G：一/四 warp 每行通过；实际 IR/PTX 已保存，无计时 |
| G3.5 | `transpose/gluon_ops.py` | G/S：直接与 convert_layout 转置通过，观察共享内存交换 |
| G3.6 | `vector/gluon_ops.py` | G/S：register/shared_sync/async_single 功能通过 |
| G3.7 | 同文件 `async_double` | G/S：双槽流水线功能通过，实际 wait_group 差异；未做 sanitizer/性能验证 |
| G3.8 | `gluon_suite.py` artifact 回调 | G/S：真实编译产物导出与说明完成，不将所有 conversion 一概视为 Shared Memory |

### 第 6–8 周：CuTe

详细说明：[04-cute.md](04-cute.md)；kernel：`experiments/cute/kernels.py`，suite：`experiments/cute_suite.py`。

| ID | 实现 | 本轮结果与未完成项 |
| --- | --- | --- |
| C4.1 | `Vector` 与 host wrapper 说明 | C：Add 功能通过；JIT/launch 分层已说明 |
| C4.2 | `experiments/cute_layouts.py` | B/C：1D/行主/列主/嵌套 layout 索引通过 |
| C4.3 | 同上 composition/divide/partition | B/C：真实 CuTe API、域与索引验证 |
| C4.4 | `Vector`、`Transpose` | C：Copy、朴素和 shared 转置通过 |
| C4.5 | `SharedCopy(values=1)` | C/S：TiledCopy Global→Shared→Global、尾部通过 |
| C4.6 | `SharedCopy(values=4)` | C/S：32/128-bit 完整 Tile 路径与标量尾部；未测 bank conflict |
| C4.7 | `WarpGemm((16,8,16))` | C/S：最小 MMA 数值通过，实际 mma.sync/HMMA |
| C4.8 | `WarpGemm` 单 stage | C/S：同步 Tile GEMM 通过，不是异步流水线 |
| C4.9 | `cute_suite.py` GEMM 尾块 | C：37×53×29 通过；计划中大 GEMM 未跑 |
| C4.10 | `cute_suite.py --tile-comparison` | 第二小 Tile 源码已备、未运行；多 Stage 未实现，明确拒绝 stages≠1 |

### 第 9–10 周：cuTile 与横向对照

详细说明：[05-cutile.md](05-cutile.md)、[07-comparison.md](07-comparison.md)。

| ID/阶段 | 实现 | 本轮结果与未完成项 |
| --- | --- | --- |
| U5.1 | `vector/cutile_ops.py`、Array/Tile 说明 | U：Copy/Add 小向量通过 |
| U5.2 | `transpose/cutile_ops.py`、`reduction/cutile_ops.py` | U：真实转置、Row Sum 与尾块通过 |
| U5.3 | `gemm/cutile_ops.py` | U：本地最小 FP16 Matmul 通过；匹配官方完整教程尚未运行 |
| U5.4 | `cutile_suite.py --scan-tiles` | 第二 Tile 源码已备、未跑；已导出独立 TileIR，无实际 PTX/SASS/资源 |
| 第 10 周 | `commands/compare/run.py` | X：四接口向量/转置、三工具链 GEMM 功能覆盖；CUDA C++/库基线和性能报告未完成 |

### 第 11–12 周：H20

[08-hopper.md](08-hopper.md) 按 **H7.1、H7.2、H7.3、H7.4、H7.5、H7.6、H7.7、H7.8** 分别列出交付、前置条件和同步推导边界。全部保持 **未实现/未验证**：当前只有 SM89，不伪造 H20 运行日志，不用 cp.async 冒充 TMA。

## 下一步的边界

先完成小功能集的学习和复核，再决定是否增加 sanitizer、少量可选 Tile/Stage 验证、库基线或真正性能实验。本轮没有把功能通过自动换算成十二周全部完成。

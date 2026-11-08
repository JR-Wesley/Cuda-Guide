# 01 — E0.1–E0.4：环境、host/kernel 边界与共享校验

本轮按“使用现有环境、小规模、功能优先”执行，没有 setup、pip/uv 安装、升级或大规模扫描。

## E0.1 当前环境

2026-11-05 本次 [host 日志](../../runs/20261105T060430Z-learn-ne9a7tkh/00-E0.1-host.log) 与各 DSL 的 GPU 上下文日志确认：RTX 4060 Laptop、SM89，驱动 KMD 596.21、CUDA UMD 13.2，系统 Toolkit 13.4.92。现有 Python 3.12.15，Triton 3.9.0/Gluon、CuTe 4.8.1、cuTile 1.7.0。

驱动负责执行/JIT，系统 Toolkit 提供 nvcc/cuobjdump 等工具，wheel 可携带自己的编译器，Torch/CuPy runtime 也不是 nvcc 的版本。旧环境记录仍保留，不能用系统 Toolkit 版本推断所有 wheel 的运行时或 PTX 兼容性。没有连接 H20。

## E0.2 原有四个 Vector Add 的逐段边界

原 `examples/` 保持不变，仍是环境冒烟测试；新共享实验在 `experiments/`。

| 文件与行 | 执行侧 | 含义 |
| --- | --- | --- |
| [Triton](../../examples/triton_vector_add.py) L1–3 | host | 导入编译器、语言与校验函数 |
| 同文件 L6–7 | JIT 定义 | `@triton.jit` 不立即执行 GPU；BLOCK 是编译常量，指针和 N 是启动参数 |
| 同文件 L8–12 | kernel | program ID 与逻辑 offset、边界 mask、load/add/store；逻辑元素数不等于 CUDA 线程数 |
| 同文件 L15–16 | host | `ceil(N/256)` grid，第一次调用触发 JIT/加载/启动；本行没有同步 |
| [Gluon](../../examples/gluon_vector_add.py) L1–5 | host | 同一 Triton 环境的显式 layout 前端 |
| 同文件 L8–10 | JIT/kernel | 定义 kernel；`BlockedLayout([1],[32],[4],[0])` 显式分配 lane/warp/元素 |
| 同文件 L11–15 | kernel | mask、两个 global load、加法和 store |
| 同文件 L18–19 | host | grid 和调用；CPU 参考/同步仍在公共 `validate` |
| [CuTe](../../examples/cute_vector_add.py) L1–4 | host | 导入 DSL 和 DLPack 桥接 |
| 同文件 L7–13 | kernel | 实际 thread/block ID，`bid*256+tid`；条件保护尾部 |
| 同文件 L16–18 | JIT host wrapper | 构建并提交 `grid`、`block=(256,1,1)`；区别于 DSL 的逻辑 Tile |
| 同文件 L21–22 | host | 将 torch tensors 转换为 CuTe views，再调用 JIT wrapper；不是 CPU 复制计算 |
| [cuTile](../../examples/cutile_vector_add.py) L1–3 | host | NumPy、CuPy、cuTile 三层 |
| 同文件 L6–12 | kernel | Tile block ID、zero-padding load、加法和带边界 store；Tile=256 不指定实际线程数 |
| 同文件 L15–22 | host | NumPy 输入，CuPy 上传与输出分配 |
| 同文件 L23–27 | host/launch | poison 输出、当前 stream 提交，device synchronize 后下载，与 CPU `a+b` 比较 |
| 同文件 L28 | host | 只在校验成功后报告 PASS |

前三个示例共享 [torch_inputs.py](../../examples/torch_inputs.py)：L8–12 在 CPU 生成参考，L13–14 上传/分配；L17–18 投毒并同步，L19 提交 kernel，L20 等待，L21 下载和校验。原 cuTile 使用 NumPy，前三个使用 Torch，因此原冒烟输入不能直接跨 DSL 比较。新实验统一修复这一差异，不改写历史结果。

## E0.3 公共输入与结果协议

源码：[core.py](../../benchmarks/core.py)、[runner.py](../../benchmarks/runner.py)。

- NumPy PCG64 根据输入族和 shape 确定种子；四个 DSL 只上传同一生成器的数组。SHA256 覆盖 dtype、shape 和逐字节输入；保存 `inputs.npz`、`reference.npy` 和两次 `actual-*.npy`。
- 向量 `1,255,256,4099`；转置 `32×32,37×53`；行归约/Softmax `1×127,3×129,2×4097`；GEMM `32×32×32,37×53×29`。CuTe 最小 MMA 另含 `16×8×16`。不跑计划中的大矩阵和 2^24 长向量。
- Copy/Transpose 精确一致，Add/AXPY/Fusion `rtol=atol=1e-6`，Sum `rtol=2e-5, atol=2e-3`（输入含约 20 倍标准正态幅值及消减），Softmax `2e-5/2e-6`，FP16 输出 GEMM `2e-3/2e-3`。不在失败后放宽容差。
- Sum/Softmax/GEMM 用 CPU FP64 运算构造参考，再转换到输出 dtype；GPU GEMM 为 FP16 输入、FP32 累计、FP16 输出。比较记录最大绝对/相对误差；相对误差分母下限为 `1e-12`。
- 空输入、错误 rank/dtype、非连续输入等不支持；Softmax 只支持有限输入。每个 GPU 输出先填 NaN，两次 launch 后分别检查形状、dtype、连续性、有限性和误差。
- `startup.prepare_ms` 包含分配/上传以及 eager compile，`first_launch_sync_ms` 包含 lazy JIT、提交和同步；两者都不是 kernel 时间，也不是完整应用端到端耗时。
- 默认 `timing.status` 明确写未测量。仅 `--bench` 启用同一当前 stream 的 CUDA Event，预热 10 次、每批 50 次、5 批；保存原始批平均值。输出投毒和校验在事件区间之外。批平均含整条 launch closure，融合基线为两 kernel 链。
- 计时 GPU 路径本轮未运行，仅有 CPU 替身回归测试；尚未做交替配置、温控与重复性能报告，不能据此评价哪种 DSL 更快。

[跨 DSL CPU 审计](../../runs/20261105-functional-comparison/SUMMARY.md) 核对了 15 个共享算子/shape 组的 input ID 与原功能 PASS，未新增 GPU 运行。

## E0.4 指令检查与工具边界

工具版本检查由 [learning_tools.py](../../scripts/learning_tools.py) 留档。初轮功能验证时 `cuobjdump`、`nvdisasm`、Compute Sanitizer 可用，脚本未在当时的可用路径中找到 `ncu`；该历史结果不改写，也不代表此前从未安装或采集过。

**2026-11-06 更新：**用户重新安装后，`ncu` 已能通过 PATH 解析为 `/usr/sbin/ncu`（实际路径 `/usr/lib/nsight-compute/ncu`），`--version` 返回 **2026.3.1.0 / build 38829034，exit_code=0**。证据见 [版本检查 manifest](../../runs/20261106T140618Z-ncu-check-rx5bgfry/manifest.json) 与 [原始日志](../../runs/20261106T140618Z-ncu-check-rx5bgfry/tools.log)，完整说明见 [环境记录](../ENVIRONMENT.md)。本次只检查版本，没有运行新的 Profile、调整性能计数器权限或修改系统配置。

[离线静态报告](06-static.md) 实际反汇编了 19 个 cubin，39 条工具命令均退出 0；保留 PTX/SASS/资源原日志，解析修正另存，不覆盖首次证据。cuTile 仅有单独前端 TileIR 导出，不冒充实际 JIT 机器码。

**指令检查已做，ncu 命令可用；当前采集权限/兼容性尚未验证，本项目完整 kernel Profile、sanitizer 和硬件计数器仍未测量，E0.4 整体验收保持部分完成。**

## 记录与复现

从项目根目录执行 python3 commands/learn/run.py，再选择 cpu/triton/gluon/cute/cutile/all；--ids 可筛选计划项。环境探测另执行 commands/inspect/run.py。

新日志固定在 commands/learn/log/：manifest.json 记录命令与退出码，原始输出在各标签 .log；backend/results.json[l] 记录 case、参数、输入身份、误差与状态，各 case 子目录保存输入、参考、实际输出和可用编译产物，SUMMARY.md 汇总功能结果。

同动作重跑覆盖 log/，不同动作互不覆盖；不复制源码/脚本，不生成制品 SHA256 或随机日期目录。历史 runs/ 及其中的快照、drift 检查结果原样保留，仅描述当时验证，不能视为当前入口的 GPU 复验。直接 suite CLI 仍保留禁止覆盖调用者指定输出的检查；正常动作会先清理自己的 log/ 再调用它。

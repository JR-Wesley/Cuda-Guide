# 底层执行、日志与工具职责

9 个可执行专题有 51 个独立的 `commands/<action>/run.sh`。每个脚本直接列出自己的动作命令；没有实验室总入口，也没有万能动作分派器。先读专题 README 与 `commands/README.md`，不能默认所有专题都支持 `profile/types/jit/debug`。

## 公共函数做什么

- [lab-common.sh](lab-common.sh)：取得专题级 `flock` 锁、设置共享产物目录、将输出写入动作脚本同目录日志、检查必需产物、记录退出码。没有源码扫描、哈希、自动快照或动作分派。依赖 Bash、`flock`、`tee` 和 Python 3。
- [lab_state.py](lab_state.py)：执行一个明确命令，记录步骤输出与退出码；显式编译失败时删除旧输出和部分输出。保留文件名以减少无关移动，但不再管理 state/cache。
- [test_lab_state.py](test_lab_state.py)：使用临时目录和模拟工具检查独立动作流程，不调用已安装 CUDA/GPU 工具。

构建产物为 `build/<variant>/`，日志为 `commands/<action>/<variant>.log`。`compile` 每次显式重建，其余执行/检查动作只复用产物；缺失时提示先编译。修改源码、头文件、工具链或参数后由使用者主动重新编译，成功标记不代表源码版本校验。正负编译探针每次重新执行并核对诊断。完整布局和特殊配置见 [RECORD_LAYOUT.md](../RECORD_LAYOUT.md)。

日志只覆盖当前动作的当前配置，不合并其他动作，不写入已有 `results/*.log`。静态检查需要的临时步骤文件在退出时清理；强制杀死进程可能留下不完整输出，不能算通过。历史证据保持原样，没有本轮 GPU 采集。

## 无 GPU 回归

从工作区根目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s gpu-architecture-lab/tools -p test_lab_state.py
```

测试覆盖 9 个真实编译脚本、缺产物拒绝运行、显式重编译、release/debug 分离、运行/sanitizer/JIT 复用、静态解析失败、编译失败阻止旧产物、日志替换与历史保护、并发锁、类型和指令正负探针。模拟工具只能验证编排与错误传播，不能证明 CUDA 编译、真实调试器停点、驱动 JIT、GPU 正确性或性能。

## 按问题选择工具

| 问题 | 工具/方法 | 不能据此证明 |
| --- | --- | --- |
| 环境和资源条件是什么 | 系统工具、CUDA Runtime 属性、编译器资源报告、occupancy API | 静态上限/预测不是动态利用率 |
| 生成了哪些指令 | NVCC/PTX、cuobjdump/nvdisasm、专题静态解析器 | 静态计数不是执行次数、延迟或缓存命中率 |
| 地址、同步或初始化错误 | CPU 参考、边界校验、Compute Sanitizer | 零报告不是任意 global 跨 block 协议正确性的证明 |
| CPU/GPU 活动是否重叠 | Nsight Systems 实际 kernel/memcpy、NVTX 与原始报告 | event 区间相交不保证实际活动重叠 |
| kernel 资源瓶颈 | Nsight Compute 指标、section/metric 与对照 | 单个 occupancy/stall 比例不能独自给出因果 |
| 错误发生在提交还是执行 | launch 检查、完成点检查、独立 debug/CUDA-GDB | 调试构建时间不是优化基线 |

每次先确认工具版本与能力。缺工具、权限不足或解析失败保存错误与退出码，不修改系统权限绕过限制，不用历史数据补写本次指标。修改内联 PTX 或测量循环后先重新 compile，再 static 确认目标指令和依赖仍存在；解析器失败不能忽略。

后续 benchmark、NCU 和其他测量动作也应各自提供可读脚本与同目录日志，复用共享源码和产物，不扩建万能驱动。参数、计时范围及状态解释见 [实验协议](../EXPERIMENT_PROTOCOL.md)。

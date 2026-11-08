# 独立动作入口

从 [专题目录](../README.md) 执行：

```bash
bash commands/test/run.sh
```

| 动作脚本 | 目的 | 同目录日志 |
| --- | --- | --- |
| [test](test/run.sh) | CPU 地址集合模型的 Python 单元测试 | `test/cpu.log` |

依赖 Bash、Python 3、`flock` 和 `tee`；无第三方 Python 包，不调用 NVCC、CUDA Runtime 或 GPU 工具，不需要先 compile。脚本直接执行测试，复用 `tools/lab-common.sh` 的锁、原始 stdout/stderr、命令和退出状态记录；`build/` 仅用于公共锁与公共层初始化目录，不产生 CUDA 二进制。`CUDA_ARCH=not-applicable` 表示本动作与 GPU 架构无关。

日志记录 Python 版本、实际测试命令、用例结果、各步骤退出码和动作退出码。测试失败返回非零，不能把失败视为跳过。普通重跑覆盖 `test/cpu.log`；若日志已被报告引用或需要保留独立对照，重跑前先保存为有意义的名称并更新引用，不依赖自动归档。测试不写入其他专题日志。

不经日志脚本直接调用 Python 的底层命令见专题 README。两者测试范围相同；只有此动作负责保存日志。结果解释见 [RESULTS.md](../RESULTS.md)，记录约定见 [RECORD_LAYOUT.md](../../../RECORD_LAYOUT.md)。

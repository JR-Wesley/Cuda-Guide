# CUDA 算子目录：本地与第三方边界

本目录混合了独立第三方克隆与本地学习材料，没有统一的本地构建/运行入口。本次只检查自维护脚本与当前说明，不重写第三方工程。

| 子目录 | 来源与入口 | 本次处理 |
| --- | --- | --- |
| [SGEMM_CUDA](SGEMM_CUDA/README.md) | 独立 Git，`siboehm/SGEMM_CUDA`；自带 CMake、Makefile、benchmark/autotuner 脚本 | 属于上游工程，保持不动，不把其脚本误当成本地动作入口 |
| [flash-attention](flash-attention/README.md) | 独立 Git，`Dao-AILab/flash-attention`；依赖和测试按上游文档 | 保持源码、安装和测试脚本不动 |
| [flash-attention-minimal](flash-attention-minimal/README.md) | 独立 Git，`tspeterkim/flash-attention-minimal`；上游使用 `bench.py` | 保持不动；检查时已有 `flash.cu` 删除状态，不恢复、不覆盖，不能据 README 假定可直接运行 |
| [softmax](softmax/online-softmax.md) | 本地 `softmax.cpp`、算法笔记和图片；没有独立 Git 或自维护 shell 入口 | 无脚本需要拆分；已有 `a.out` 保留，不能把它当作当前编译/验证结果 |

## 自维护入口约定

当前没有需要迁移的本地脚本，因此不新增万能 `lab.sh`、换名 dispatcher 或空动作框架。未来为具体练习增加入口时：

- 每个动作分别放 `commands/<action>/run.sh`，命令、输出和退出码写入同目录 `output.log`；重跑只覆盖本动作日志，不自动归档。
- 共享源码与固定 `build/` 放在动作目录之外；编译与运行分别执行，运行缺产物就报错，不隐式重新编译。
- 不生成哈希、自动源码快照、时间戳/history 目录；不改写或删除原有历史记录。
- 第三方仓库继续使用自身构建约定；任何迁移须先明确本地维护边界，不能为了统一目录形式重写上游脚本。

本次没有运行编译、基准或 GPU 测试；只核对目录与 Git 边界。上游文档中的性能数值不是本次环境的实测结论。

# 二进制阅读导航：一次回答一个问题

## 背景与学习目标

编译后需要区分“装了哪些镜像”“生成了哪些指令”“声明了多少资源”。这些都是离线读文件，不能证明程序在 GPU 上算对了或跑得快。本目录只保留导航，旧 `cuobjdump-tour.sh` 已删除；每个动作有自己的脚本、README 和就地日志，没有总调度入口。

前置：认识主机程序、PTX（虚拟指令）和 cubin（目标机器码的设备 ELF）。通用工具关系见 [Toolkit 总导览](../../TOOLKIT_GUIDE.md)，本页只组织实践。

## 1. 显式准备一个小输入

按 [编译专题](../compile/README.md) 的当前独立编译入口生成同一份 `compiler_probe.cu` 的产物：

```text
build/<arch>/compile/artifacts/
  probe.ptx       # 独立 PTX 文本，可直接打开
  probe.cubin     # 独立设备 ELF；nvdisasm 的输入
  probe.fatbin    # 设备镜像容器
  probe.o         # 主机目标文件
  probe           # 主机可执行文件；这里只读，不执行
```

本轮目标为 `sm_89`、Toolkit 13.4。cubin 需保留行信息与重定位信息供后续阅读；这些由编译动作负责，检查动作绝不偷偷编译。`CUDA_ARCH` 只选择共享目录，不把任意输入转换为该架构。

## 2. 先底层命令，再选择一个记录入口

下面路径均按相应动作目录解释。手动进入表中目录后执行命令；每份 README 解释参数、输出和练习。

| 问题 | 独立动作 | 核心底层命令（后接输入文件） | 默认输入 |
| --- | --- | --- | --- |
| 有哪些设备镜像？ | [binary-list](../binary-list/README.md) | `cuobjdump --list-elf` 与 `--list-ptx` 分别调用 | `probe.fatbin` |
| 内嵌 PTX 是什么？ | [binary-ptx](../binary-ptx/README.md) | `cuobjdump --dump-ptx` | `probe.fatbin` |
| 最终机器指令是什么？ | [binary-sass](../binary-sass/README.md) | `cuobjdump --dump-sass` | `probe.cubin` |
| 静态资源是多少？ | [binary-resources](../binary-resources/README.md) | `cuobjdump --dump-resource-usage` | `probe.cubin` |
| 设备节与符号是什么？ | [binary-elf](../binary-elf/README.md) | `cuobjdump --dump-elf` 与 `--dump-elf-symbols` | `probe.cubin` |
| 指令对应哪行源码？ | [nvdisasm-sass](../nvdisasm-sass/README.md) | `nvdisasm --print-code --print-line-info` | `probe.cubin` |
| 控制流怎样连接？ | [nvdisasm-cfg](../nvdisasm-cfg/README.md) | `nvdisasm --output-control-flow-graph-with-basic-blocks` | `probe.cubin` |
| 哪些寄存器值还会用到？ | [nvdisasm-live-ranges](../nvdisasm-live-ranges/README.md) | `nvdisasm --print-code --print-life-ranges` | `probe.cubin` |

例如，当前目录为 `experiments/binary-list/` 时，先理解：

```bash
cuobjdump --list-elf ../../build/sm_89/compile/artifacts/probe.fatbin
cuobjdump --list-ptx ../../build/sm_89/compile/artifacts/probe.fatbin
# 同一动作的记录入口
CUDA_ARCH=sm_89 bash run.sh
```

其他动作也在其自身目录用 `CUDA_ARCH=sm_89 bash run.sh` 调用，不在本导航目录执行。每个脚本至多接受一个文件参数，不接受 artifacts 目录或 `all`。`CUOBJDUMP` / `NVDISASM` 可指定相应工具路径，默认由 `PATH` 查找。

## 3. 日志和证据边界

每个动作复用 `../record.sh`，同名记录重跑覆盖，不自动归档：

- `log/<label>.sh`：实际命令，可在该动作目录重放；重放不更新状态表。
- `log/<label>.log`：该命令 stdout/stderr，`log/tool_version.log` 保留工具版本。
- `log/status.tsv`：本次执行步骤及真实退出码；失败即停止，未列入本次表的旧文件不算新证据。
- CFG 是唯一输出分流特例：stdout 为 `log/cfg.dot`，stderr 为 `log/cfg.log`，状态表仍记录 `nvdisasm` 真实退出码。

缺输入用 `require_file` 报错，缺工具会记录实际命令失败；都不能标成通过。清单为空而退出码为零时，也不能假定镜像存在。PTX dump 含包装信息，不是可直接加载的提取文件。

不恢复 `--extract-elf` / `--extract-ptx`：其逗号参数是名称过滤列表，不是输出文件映射。独立 cubin 已由编译产生，无需提取；不为不提供的功能创建假的执行日志。

## 4. 练习与排查

先预测 `probe.fatbin` 是否同时包含 ELF/PTX，再运行清单动作核对。选择其中一种指令表示阅读，解释为什么“镜像存在”不能证明“驱动已加载它”。失败先看状态表和输入编译记录，不运行 GPU 来修复静态读取问题。

转置程序的可选延伸见 [static 导航](../static/README.md)，默认仍使用小 probe，不批量生成大 dump。

## 本轮状态

已确认 `cuobjdump`、`nvdisasm` 为 13.4.92 并核对选项。共享输入生成后，以 `CUDA_ARCH=sm_89` 实际运行了上述 8 个独立动作，所有记录命令退出码均为 0，各动作 README 已链接本次日志并补充观察。8 份脚本的 `bash -n` 检查通过。镜像清单各有一份 cubin/PTX，资源为 `REG:10`，行信息与寄存器区间均实际输出；CFG DOT 为 987 字节、stderr 为空。`dot` 未找到，渲染未执行、不安装依赖、不生成图片报告；DOT 未经 Graphviz 解析验证。没有 GPU 运行或性能测量，也未运行转置延伸或手动练习。

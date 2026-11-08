# 静态检查导航：从小 probe 延伸到转置程序

## 背景与目标

静态检查只读取编译产物，不启动 kernel。先用一个小函数学会读镜像、指令和资源，再把相同问题应用到已有转置程序，避免一次输出大量信息。本目录只有导航，旧 `run.sh` 已删除，不再把资源、PTX 和 SASS 串成一个大动作。

前置：先按 [二进制导航](../binary/README.md) 完成小 probe 阅读。转置输入由独立 [build 动作](../build/) 显式生成 `build/<arch>/vector_transpose`；检查脚本不编译。

## 1. 先选一个问题

| 目标 | 入口 | 脚本日志 |
| --- | --- | --- |
| 有没有原生/PTX 镜像？ | [binary-list](../binary-list/README.md) | `log/list_elf.log`、`log/list_ptx.log` |
| 看静态资源 | [binary-resources](../binary-resources/README.md) | `log/resources.log` |
| 看内嵌 PTX | [binary-ptx](../binary-ptx/README.md) | `log/ptx.log` |
| 看机器指令 | [binary-sass](../binary-sass/README.md) | `log/sass.log` |

## 2. 在目标动作目录执行底层命令

例如手动进入 `cuda-sdk-lab/experiments/binary-resources/`，只读资源：

```bash
cuobjdump --dump-resource-usage ../../build/sm_89/vector_transpose
# 同一动作的记录脚本；传入文件，不传目录
CUDA_ARCH=sm_89 bash run.sh ../../build/sm_89/vector_transpose
```

`--dump-resource-usage` 输出静态分配，不执行主机程序。`CUDA_ARCH=sm_89` 显式选择本轮目录；它不能改变输入的目标架构。脚本不切目录，输入不存在就提示先构建。其他动作也支持这一单文件参数，但不要为了“完整”而一次跑完全部 dump。

对较大程序只想读某个 kernel 时，先从符号输出确定其真实名称，再手动用 `cuobjdump --function <实际符号名> --dump-sass <输入文件>` 缩小范围；尖括号表示需根据实际输出替换，不是可直接执行的命令。不添加通用参数框架。

`nvdisasm` 的三个动作仍只接收独立 cubin，不能把 `vector_transpose` 主机程序直接交给它们；本轮不增加提取流程。

## 3. 解释输出与记录

`REG` 是每线程寄存器分配数，内存字段以字节计。先按函数区分，再看指令；静态资源不等于 occupancy，存在加载/存储指令也不证明缓存命中率或吞吐。

各动作日志在各自 `log/`，不写到本导航目录。`.sh` 保存命令，`.log` 保存输出，`status.tsv` 保存本次真实退出码；同名文件重跑覆盖。用转置替换 probe 会覆盖该动作的 probe 记录，不自动维护双份报告；需要对照时先由使用者明确保存。

## 4. 练习

只选择资源检查：先预测不同转置 kernel 的静态共享内存是否相同，再用已有程序核对。若不同，回源码解释存储选择；不要直接得出哪个更快的结论，也不要为了本练习启动 GPU。

## 本轮验证

本页是操作导航，不声称转置检查已运行。工具选项已按 13.4.92 核对；实际结果必须以所选动作本次日志为准。

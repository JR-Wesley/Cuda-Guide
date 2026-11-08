# `cuobjdump` 二进制阅读实践索引

工具的整体定位、输入类型与 `nvdisasm` 的区别统一见 [Toolkit 总导览](TOOLKIT_GUIDE.md)。本页只索引已经拆开的实践，不再提供 `cuobjdump-tour.sh` 大动作，也不新增另一份 nvdisasm 总论。

## 学习顺序

先按 [编译专题](experiments/compile/README.md) 显式准备 `build/<arch>/compile/artifacts/` 下的小 probe 产物，再一次选择一个问题：

1. [镜像清单](experiments/binary-list/README.md)：分别列 ELF/PTX，判断容器里有什么。
2. [PTX dump](experiments/binary-ptx/README.md)：读内嵌虚拟指令，不把 dump 日志当成提取文件。
3. [SASS dump](experiments/binary-sass/README.md)：读目标机器指令，对照 PTX。
4. [静态资源](experiments/binary-resources/README.md)：认识寄存器、栈与共享内存字段。
5. [ELF 与符号](experiments/binary-elf/README.md)：定位代码节、函数名，区分设备 ELF 与主机目标文件。

每个动作只读取一个输入文件，有自己的 `run.sh`、README 和 `log/`。具体默认输入、可选文件参数及记录边界见 [二进制导航](experiments/binary/README.md)。扩展到已有 `vector_transpose` 的单项检查见 [static 导航](experiments/static/README.md)；没有静态检查总入口。

## 独立 cubin 的后续阅读

编译直接生成带行信息、保留重定位信息的 `probe.cubin`，无需恢复提取：

- [SASS 与源码行](experiments/nvdisasm-sass/README.md)
- [CFG DOT](experiments/nvdisasm-cfg/README.md)：stdout 与 stderr 分离，保留真实退出码
- [寄存器活跃区间](experiments/nvdisasm-live-ranges/README.md)

## 记录与验证

在所选动作目录运行 `CUDA_ARCH=sm_89 bash run.sh`。架构必须显式指定；脚本不切目录、不隐式编译、不运行 GPU。重跑覆盖本动作同名日志，具体命令和退出码由共享 `experiments/record.sh` 记录，CFG 使用简单的独立重定向保存 DOT。

本轮已确认 `cuobjdump` / `nvdisasm` 13.4.92 的工具选项。共享 probe 产物生成后，已按 `sm_89` 运行上述 8 个离线动作，所有记录命令退出码均为 0；实际日志与观察见各动作 README。8 个脚本的 Bash 语法检查通过。`dot` 未找到，渲染未执行，不安装工具、不创建 SVG/PNG 假报告；DOT 未经 Graphviz 解析验证。没有执行 GPU 程序，静态输出不证明正确性、JIT 成功或运行性能。

官方参数依据：[CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)。

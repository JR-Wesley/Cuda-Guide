# SASS dump：阅读目标机器指令

## 背景与目标

编译器将 PTX 降低为某种 GPU 架构的机器指令；SASS 是这些指令的可读反汇编。本节从小型 `probe.cubin` 找到索引、边界检查和浮点运算，不导出大型应用的全部代码。

前置：按 [编译专题](../compile/README.md) 显式准备 `probe.cubin`；先认识 [PTX](../binary-ptx/README.md) 与最终指令的区别。本动作仅离线读文件。

## 1. 底层命令

在 `cuda-sdk-lab/experiments/binary-sass/` 中：

```bash
cuobjdump --dump-sass ../../build/sm_89/compile/artifacts/probe.cubin
```

`--dump-sass` 解码原生设备代码。cubin 是设备 ELF；它不是主机可执行程序，也不需要 GPU 参与反汇编。

## 2. 记录脚本

```bash
CUDA_ARCH=sm_89 bash run.sh
```

默认读取 `../../build/<CUDA_ARCH>/compile/artifacts/probe.cubin`，可用唯一文件参数替换输入。`CUDA_ARCH` 必填，仅选目录，不改目标架构；工具可由 `CUOBJDUMP` 指定。脚本必须在本目录执行，不切目录、不编译。

`log/sass.log` 保存输出，`log/tool_version.log` 保存版本；各自的 `.sh` 可在本目录重放，但重放不更新 `status.tsv`。脚本将真实退出码写入状态表，失败即停；重跑覆盖本动作同名输出，不读取旧输出来声称成功。

## 3. 阅读输出

先看 `arch` 和 `Function : saxpy_probe`，再看地址偏移及指令。预期可以定位线程索引相关指令、谓词判断、全局内存访问以及浮点乘加；具体助记符由编译器和架构决定，不应预先断言逐行对应 PTX。

谓词是控制指令是否生效的条件位，例如 `@P0` 表示受条件限制。偏移不是执行耗时；一个访存指令也不能证明缓存命中或访问合并。源码行对应关系在 [nvdisasm SASS/行号](../nvdisasm-sass/README.md) 中单独阅读。

## 4. 练习

先在 [ELF/符号](../binary-elf/README.md) 找到真实函数名，再手动缩小范围：

```bash
cuobjdump --function saxpy_probe --dump-sass ../../build/sm_89/compile/artifacts/probe.cubin
```

`--function` 按函数名筛选。解释筛选后为什么仍不能根据指令行数预测 kernel 时间；若输出为空，先检查符号名而不是只看退出码。

## 本轮验证

输入生成后，已用 `CUDA_ARCH=sm_89 bash run.sh`、`cuobjdump` 13.4.92 完成离线检查，见 [SASS](log/sass.log) 和 [退出码 0](log/status.tsv)。目标为 `sm_89`，函数为 `saxpy_probe`；本次 `0x0040` 的 `ISETP.GE.AND` 与 `0x0050` 的 `@P0 EXIT` 实现越界提前退出，`0x00c0` 的 `FFMA` 对应浮点乘加，前后有 `LDG.E` / `STG.E`。这里只确认指令形式，未执行 GPU 程序。

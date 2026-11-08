# 编译、指令检查与运行调试

> 当前复现入口已迁到 `commands/<action>/run.sh`：先按专题 README 显式 compile，再执行相应动作。下文命令是当前复现写法，历史精确命令仍以原日志为准；本轮未重新采集 GPU 数据。

动作入口为 `commands/<action>/run.sh`；先显式执行 `commands/compile/run.sh`，其余执行/检查动作只复用 `build/<variant>/` 的已有产物，缺失即提示先编译，不隐式构建。修改源码、头文件、编译选项或工具链后必须主动重新编译；没有哈希缓存或自动失效判断。

日志位于脚本同目录 `commands/<action>/<variant>.log`，同动作同配置重跑覆盖（包括失败），其他动作日志不变。命令、实际参数、原始输出及退出码均记录；检查器的临时步骤文件在退出后清理。不生成自动快照、时间戳归档或 history。已有 `results/` 原始日志和历史报告保持原样，当前动作不写入。

静态解析器临时读取动作目录中的步骤输出，结果保存在 `commands/static/<variant>.log`；旧 `results/` 日志仅解释历史诊断，本轮未真实编译或运行 GPU。

## 1. 先分清三类失败

| 阶段 | 检查对象 | 本专题提供的证据 |
| --- | --- | --- |
| C++ 编译 | overload 是否存在、参数类型是否匹配 | 当前 `type_probe.cu` 与 types 节的命令、diagnostic、退出码 |
| 静态检查 | C++ 意图是否保留为预期 PTX、是否有实际 SASS 访存 | `static` 的 `probe.ptx`、`sass.log`、`inspect.log` |
| GPU 执行 | 值正确、边界正确、地址和对齐是否合法 | `run/edge` 逐字节验证及 `memcheck` |

历史开发过程中主程序首次构建成功；运行未出现数值或设备访存错误。实际修正的是静态检查器对 packed half 指令类型的假设。下文只保留失败的历史说明；被后续记录取代的首次 build 与失败 static 日志及快照已清理，没有人为构造 GPU 崩溃。现存证据与版本限制见 [记录索引](results/README.md)。

## 2. 真实定位：`__half2` 被误判为缺少指令

首次 `static` 的结果为 16 项检查、1 项失败：

```text
FAIL probe_half2_ldg
missing ('ld', 'global', 'nc', 'u32')
observed: ld.global.nc.b32 %r1, [%rd1];
SASS: LDG.E.CONSTANT R3, [R2.64]
```

以上为历史诊断摘要，不是现存日志链接。原目录 `load_store_intrinsics-20261023T141903Z-SmDvJf` 的 `inspect` 退出码为 1；该记录被后续 static 取代，日志、PTX 和快照已清理，不能作为当前源码验证证据。

定位步骤：

1. 在具名的 `probe_half2_ldg` 函数内查找实际 load，避免把其他 kernel 的 `ld.global.nc` 当作证据。
2. 对比 `sizeof(__half2)=4` 与 PTX `.b32`。这里是 32 位不解释数值含义的搬运，不要求 `.u32`，也不要求 `.f16x2`。
3. 查看同一函数 SASS，存在 `LDG.E.CONSTANT`，说明访存没有丢失。
4. 将该检查的预期类型从 `u32` 修正为实际接口实现的 `b32`，仍要求 `ld/global/nc` 和实际机器 load；没有移除失败项目。
5. 重新生成 PTX/cubin/SASS，再运行全部 16 项检查，结果 `failed=0`。

修正后记录：[配置日志](results/sm_89-release.log) static 节中的 `inspect.log`。这份日志只证明修正后的检查结果，不替代已经清理的首次失败日志。

## 3. 重载错误如何复现与判断

`bash commands/types/run.sh` 为每个候选类型分别编译普通赋值、`__ldg`、`__ldca`、`__stwb`，不会让第一个错误掩盖后续类型。以 `bool` 为例，普通 kernel 能编译，调用 intrinsic 时 nvcc 返回 2，并报告：

```text
no instance of overloaded function "__ldg" matches the argument list
```

原始记录位于 [配置日志](results/sm_89-release.log) types 节的 `type_bool_1.log` 与 `types.tsv` 文件标界内。当前 `check_types.py` 每次重新编译全部正负探针，保留原诊断匹配；日志写入 types 动作目录。历史 types 日志不证明新入口已验证。

探针同时要求“非零退出码”和含目标函数名的 overload diagnostic 才把负向测试记为通过。找不到 nvcc、头文件缺失或不支持目标架构等错误都不能冒充预期的类型拒绝。如果以后 Toolkit 增加了重载，当前预期失败项会报警，应检查新证据并更新矩阵，而不是继续声称不支持。

这种错误应通过调整存储类型或逐成员访问修复；随意强制转换指针会把编译错误变成可能的对齐、越界或类型别名问题。

## 4. GPU 检查的定位粒度

`type_matrix` 对每个 variant 先重置输出，再 launch、`cudaGetLastError()`、`cudaDeviceSynchronize()`、拷回并比较。任何不一致都会报告类型、load/store 名称和首个不一致的元素编号。由于比较包含头尾 guard，不能只看中间有效数据就忽略边界写坏。

`run` 的 n=4099 验证尾 block，`edge` 的 n=1 验证极小输入。`sanitizer` 使用 n=257，仍有多个 block 和不满 block 的尾部，同时运行全部类型、三代 producer/consumer 和 mapped handoff。

```bash
bash commands/sanitizer/run.sh
```

此任务没有 shared-memory 通信协议，采用 memcheck 检查访存合法性即可。没有把 racecheck 或 synccheck 的潜在零报告当作 global-memory 只读约束的证明；只读性由分离 allocation 和明确的 kernel 依赖在代码层保证。

memcheck 通过的含义是这次执行中没有被该工具发现的访存错误。它不证明某次 load 命中了 L1，也不证明把一个合法只读 kernel 改成边写边读之后仍正确。

## 5. 读反汇编时的三处陷阱

- **看到 `LDG` 就认定用了 `__ldg`。** 本机普通 global load 也是 `LDG.E`；只读 probe 是 `LDG.E.CONSTANT`。必须连同 PTX 和目标架构看，不能仅搜索助记符前缀。
- **看到 `CONSTANT` 就认定来自 `__constant__`。** 本例输入来自 `cudaMalloc` 的 global memory。该 SASS 修饰符不是 C++ 对象的地址空间声明。
- **看到 `STRONG.GPU/SYS` 就认定具有 acquire/release。** cache operator 没有把普通数据访问变成原子同步；正确性仍要按照 C++/PTX 内存模型和 API 依赖推导。

现有 [Cache Operators 调试分析](../01_cache_operators/README.md) 还讨论了重复加载被提升到循环外的情况。如果扩展性能循环，应继续检查最终 SASS 的循环结构，而不是只根据源码迭代次数计算访存量。

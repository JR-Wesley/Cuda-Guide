# Thread Scopes 实验调试过程

> 当前复现入口已迁到 `commands/<action>/run.sh`：先按专题 README 显式 compile，再执行相应动作。下文命令是当前复现写法，历史精确命令仍以原日志为准；本轮未重新采集 GPU 数据。

> 记录整理（2026-11-06）：历史证据按真实编译配置合并为 `results/<variant>.log`，各 action 仅保留最新一节；见 [记录索引](results/README.md)。无日期目录或源码/脚本副本。本次未编译、未运行 GPU。下文数值仅对应标有 `HISTORICAL IMPORT` 的原记录；链接进入配置日志后按 `ACTION` 与 `FILE` 标界查找。本轮迁移保留这些历史证据；新日志写入独立动作目录，旧结论不自动套到新日志。

调试顺序为：编译 → 数值及边界 → memcheck → racecheck/synccheck → 静态指令核对 → 有界反例与修复。所有记录来自本次实际执行，日志索引见 [RESULTS.md](RESULTS.md)。

## 1. 模板按引用绑定主机 constexpr

初次构建在 `copies.cu` 的故意字节数错误示例中失败：

```text
error: identifier "Tile" is undefined in device code
```

历史失败对应（`thread_scopes-20261023T140033Z-Eeq9OT` 日志已清理）四参数调用 `cg::memcpy_async(group,tile,in,Tile)`。`Tile` 是头文件中的 namespace-scope constexpr；该模板参数以 `const TySizeT&` 接收，直接绑定名字可能需要主机变量的存储实体，不只是把整数值作为编译期常量使用。

将其改成 `size_t(Tile)` 临时值后编译通过。这里仍故意传错字节数 128，保留下一节的逻辑错误演示；正常路径始终使用 `sizeof(tile)`，即 512 字节。

这与 shared 数组界限中使用 `Tile` 的情况不同：数组界限只需要常量值，不需要对主机对象建立引用。修复没有开启 relaxed constexpr 编译选项，也没有把全局变量改成设备存储。

## 2. 四参数 memcpy_async 的单位错误

复现：`bash commands/debug/run.sh`。`wrong_bytes` 先用 `0xdeadbeef` 初始化整个 shared tile，再 block sync，随后发出四参数异步拷贝并正确等待。拷贝完成后逐元素复制到输出。

它要拷贝 128 个 unsigned，却传入 128 字节。实际只有 32 个元素被覆盖，余下 96 个保持毒化值。本次结果：

```text
EXPECTED_MISMATCH requested_bytes=128 required_bytes=512 mismatches=96 poison=96
PASS byte_count_bug_detected
```

[原始日志](results/sm_89-release.log)。这是一个确定的逻辑错误：所有 shared 元素先初始化，所有访问都有合法边界，拷贝也等待完成。因此 memcheck 不必报告越界，racecheck 也不必报告竞争，数值校验仍能发现少拷贝。

修复方式：四参数 overload 传 `Tile*sizeof(unsigned)` 或 `sizeof(tile)`。正常的 `copy_single<1>` 已用后者，并覆盖 `n=1/127/128/129/1027`。注意五参数 layout overload 有元素数语义，不能将其解释套到四参数版本。

## 3. 超过默认 shared 上限但未 opt-in

同一 `debug` 动作在独立的 carveout 进程中执行一次没有 opt-in 的 64 KiB dynamic shared launch。设备报告默认上限 49152 字节，opt-in 上限 101376 字节。

实测：

```text
EXPECTED_LAUNCH_FAILURE before_optin=cudaErrorInvalidValue
PASS large_shared_requires_optin
```

[原始日志](results/sm_89-release.log)。程序用 `cudaGetLastError()` 获取并清除 launch 错误，只接受 InvalidValue 或 InvalidConfiguration 这两类预期错误，不把任意失败当作正确诊断。

随后设置 `cudaFuncAttributeMaxDynamicSharedMemorySize=65536`，重新运行相同 kernel；四种 carveout 偏好下的所有输出均通过 CPU 校验。它演示的是可恢复的 launch 参数/资源错误，没有故意触发非法地址或破坏 context。

如果设备不存在超过默认上限的可用动态 shared 容量，反例返回 SKIP/77。不能为制造预期错误而硬编码超出设备总能力的大小，再将那种失败解释为缺少 opt-in。

## 4. 消息发布代码审查

附件片段省略 `ready` 初始化，且没有给出多轮复用协议。正常实现采用：

1. 每轮由唯一初始化者重置 mailbox。
2. block/grid 同步，确保 reset 完成。
3. 生产者写 payload，release 发布 flag。
4. 消费者 acquire 读到 flag 后才读 payload。
5. block/grid 同步，保证读完再复用。

同一 block 的通信使用线程 0 和 32，覆盖跨 warp 情形。跨 block 通信只在 cooperative launch 驻留条件检查通过后启动。所有自旋均有预算；超时写出 `0xffffffff` 哨兵，host 必须判错。

block/global 和 block/shared counter 两个合法对照共同说明存储位置不等于 scope。没有执行“跨 block 使用 block 原子”这类未定义行为来寻找某个偶然错误结果。其错误来自 scope 不覆盖参与者，不需要依赖一次运行是否出现旧值。

## 5. Pipeline 与拷贝同步检查

对每种路径检查两个时刻：

- **何时可读：** collective wait、barrier wait，或 thread wait 后补 block sync。
- **何时可覆盖：** 消费完后的 block/barrier 同步，或 pipeline consumer release 协议。

本例读取邻居元素，故意使“只等本线程搬运完成”不足以满足依赖。输入按 tile 补零，输出只写 n 个元素；小于一 tile 与非整除尾部都纳入检查。两阶段 pipeline 的九个 tile 输入经过多次槽复用及最后排空。

正常程序的三种分析器均无错误。racecheck 主要检查 shared hazards，本例 global mailbox 的作用域与 release/acquire 正确性仍依赖内存模型推导，不能声称 racecheck 证明了所有 global 原子协议。工具说明见 [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)。

本专题没有包含上一专题的原生 parity 或 custom completion 测试；本次零报告也不表示此前专题未解决的分析器报告已被修复。

## 6. 能力检查先于高级路径执行

CPU/GPU managed-memory 并发要求 concurrentManagedAccess，本机为 0。因此 `system.cu` 在分配并发 mailbox、启动 kernel 前返回 77；没有把 managed memory 换成普通 host 指针来绕过条件。

cluster 原子独立编译到 SM90，并在主机侧检查 SM 与 clusterLaunch 属性。SM89 不启动该 kernel。编译成功说明工具链接受代码，不证明本机已执行，也不证明其他设备上的驱动 JIT 能支持当前 Toolkit 生成的 PTX。

`system_gpu_only` 则完全不同：所有并发原子访问来自一块 GPU 的 cudaMalloc 内存，CPU 在结束后复制结果。这项正常通过不会抵消 CPU/GPU 并发路径的跳过状态。

## 7. 静态分析如何避免误读

第一次静态检查将四个原子 kernel 判为 FAIL（`thread_scopes-20261023T140534Z-XYvmOD` 日志已清理，此处保留历史原因）。读取 PTX 后确认指令存在，形式是 `atom.add.relaxed.cta.u32` / `.gpu` / `.sys`；旧匹配规则错误地要求 scope 出现在 `add` 前面。

修复为提取每个 kernel 中的完整 atomic opcode，再按点分隔的修饰符集合检查 `add`、`relaxed` 和预期 scope，避免固定修饰符次序假设。同时将 SASS 的 LDGSTS 存在性加入自动检查。当时修正后重新运行 static，没有忽略解析器失败；本次整理只保留修正后的最新 static，旧失败日志已清理，不以新日志替代旧失败证据。

`check_static.py` 在 `count_cta/count_gpu/count_sys/count_builtin` 各自 kernel 内检查原子指令，避免因另一个 kernel 恰好存在某 scope 指令就误判。拷贝 PTX 检查 cp.async、commit、wait；完整 SASS 也保存，可定位 LDGSTS。

一个 API 可以被编译成更强的 scope 或 order；这不扩大程序可依赖的源代码契约。静态指令与数值 PASS 不证明实际 cache 路径、copy/compute 重叠或性能优势。本专题没有通过计时推断这些属性。

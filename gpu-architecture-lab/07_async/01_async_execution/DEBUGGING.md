# 异步执行调试记录与方法

> 当前复现入口已迁到 `commands/<action>/run.sh`：先按专题 README 显式 compile，再执行相应动作。下文命令是当前复现写法，历史精确命令仍以原日志为准；本轮未重新采集 GPU 数据。

以下为历史观测和调试方法，本轮没有运行 GPU 或真实编译。已有 [sm_89-release.log](results/sm_89-release.log) 保持原样。动作入口为 `commands/<action>/run.sh`；先显式执行 `commands/compile/run.sh`，其余执行/检查动作只复用 `build/<variant>/` 的已有产物，缺失即提示先编译，不隐式构建。修改源码、头文件、编译选项或工具链后必须主动重新编译；没有哈希缓存或自动失效判断。日志位于脚本同目录 `commands/<action>/<variant>.log`，同动作同配置重跑覆盖（包括失败），其他动作日志不变。命令、实际参数、原始输出及退出码均记录；检查器的临时步骤文件在退出后清理。不生成自动快照、时间戳归档或 history。已有 `results/` 原始日志和历史报告保持原样，当前动作不写入。

## 1. 分开检查提交失败和执行失败

推荐保留这两处检查：

```cpp
kernel<<<grid, block, 0, stream>>>(...);
CUDA_CHECK(cudaGetLastError());       // 提交/launch 参数检查
CUDA_CHECK(cudaStreamSynchronize(stream)); // 调试或确需结果时检查执行完成
```

第一处成功不表示 kernel 已经成功执行。错误也可能由之后的 CUDA API 报出，因此应检查每个 API 的直接返回值；不要只在最后 peek 全局错误状态。正常性能路径不应在每次 launch 后都加同步，否则会改变想观察的并发。

## 2. 真实发现：非法 launch 的错误码并非预先假设的那个

`errors launch` 使用 `maxThreadsPerBlock + 1` 启动空 kernel。本机 Toolkit 13.4 返回 `cudaErrorInvalidValue`，最初只接受 `cudaErrorInvalidConfiguration` 的测试因此失败。该旧 run（`async_execution-20261023T132327Z-5ZmwAJ`）已被后续完整验收取代，原日志已清理；这里仅保留历史原因。现存证据及源码匹配边界见 [results/README.md](results/README.md)。

修正后的检查接受这两种与非法配置/参数对应的返回，并继续验证：

1. `cudaPeekAtLastError()` 返回非成功状态且不清除。
2. 紧接着的 `cudaGetLastError()` 返回相同状态并清除。
3. 再次 Get 返回 `cudaSuccess`。
4. 一个合法 kernel 可以启动并完成。

这是可恢复的 launch 参数错误。代码没有把任意 CUDA 错误都当作“预期错误”，也没有把清除 launch 错误推广到执行期致命错误。

## 3. 执行期故障与 CUDA_LAUNCH_BLOCKING

`errors fault` 在单独进程运行 `invalid_write`，故意写地址 0。普通运行观察到：

```text
launch_check=cudaSuccess
synchronize=cudaErrorIllegalAddress
```

用 `CUDA_LAUNCH_BLOCKING=1` 运行同样代码，观察到：

```text
launch_check=cudaErrorIllegalAddress
synchronize=cudaErrorIllegalAddress
```

这说明调试开关可以使错误更早暴露在 launch 检查处；它没有修复非法访问。遇到这种执行期错误后，本例结束进程，不通过 GetLastError 清除后继续复用 context。

使用：

```bash
bash commands/debug/run.sh
```

脚本为普通故障与 blocking 故障分别启动进程，最后再用 Compute Sanitizer 定位同一个非法写入。

## 4. Compute Sanitizer 定位

本机故障检查定位到 `invalid_write()` 中 `errors.cu` 的内联 PTX 写入，报告：

```text
Invalid __global__ write of size 4 bytes
Access to 0x0 is out of bounds
```

在插桩环境中，后续同步返回了 `cudaErrorUnknown`，而非未插桩时的 IllegalAddress，故目标程序也返回非零；工具汇总包含非法写入和 API 错误共 2 项。

`debug` 脚本要求“工具非零退出且出现 Invalid global write 诊断”才将这一预期故障阶段记为检测成功。它不会把所有非零退出一概视为成功。原始非零退出码仍保存在配置日志 debug 段内的 `status.tsv` 内容中，正常路径的 sanitizer 检查则必须零错误退出。

## 5. 排查异步程序常见问题

| 现象 | 首先检查 |
| --- | --- |
| 输出偶尔错误 | 是否在 D2H 完成前读取 host 输出；跨流是否缺少 event 依赖 |
| 分块结果被覆盖 | 同一 host/device 区间是否在传输或计算完成前复用 |
| 多流没有加速 | pinned 条件、分块规模、硬件能力、隐式默认流依赖、host function 延迟、实际 trace |
| 一个 event 完成但其他工作未完成 | event 是否覆盖所有 worker；是否遗漏 fan-in；是否在 event 后仍排有工作 |
| 回调死锁或失败 | host function 是否调用 CUDA API，或等待其后排队的工作 |
| Graph 重放数据错误 | capture 时保存的地址是否仍有效；输入和依赖是否与普通路径一致 |
| 故障指向某个同步 API | 在可复现场景用 launch blocking 或缩小同步范围定位真正的 kernel，再用 sanitizer 检查 |

## 6. 对所贴材料的几个纠正和限定

- `cudaMallocHost` 分配的内存应使用 `cudaFreeHost`，不能用材料示例末尾的 `free(h_data)`。
- `cudaStreamDestroy` 不要求调用返回前所有工作已经完成。排队工作与资源清理有自己的完成时机，数据缓冲区不能因此提前释放。
- 检验 Peek 不清除错误时，紧接着的 Get 应返回相同错误；材料中用 `err2 != err` 打印“Peek 未清除”是反向判断。
- Last-error 查询状态与调用的 host thread 相关，不能把它当作所有线程、所有流的完整错误日志。同步 API 的直接返回值必须检查；严重执行错误也不会因为 Get 清除状态就恢复 context。
- `cudaMemcpyAsync` 的 host 同步行为受内存类型、方向和内部资源影响；“Async 一定立即返回”和“pageable 一定完全同步”都过于绝对。
- event 只保证排在它之前的工作完成，并不保证排在它之后的 kernel 尚未开始。
- 新代码使用 `cudaLaunchHostFunc`。材料中简写的 `cudaAddCallback` 不是本例使用的 Runtime API；旧接口的完整名称是 `cudaStreamAddCallback`。

上述同步行为以 [Runtime API synchronization behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html) 和 [Stream Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html) 为准。错误查询作用域与清除语义见 [Runtime Error Handling](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html)。本例的数值、顺序、错误码和时间线证据链接汇总在 [RESULTS.md](RESULTS.md)。

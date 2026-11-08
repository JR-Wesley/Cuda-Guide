# 04 · Synchronization / Atomic / Ordering

对应 M4、M5。目标是区分**功能正确性**与**同步成本**；已有语义/校验专题并不等于已完成同步开销扫描。

## 已有可执行专题

| 入口 | 先回答的问题 | 限制与阅读重点 |
| --- | --- | --- |
| [01_memory_model](01_memory_model/README.md) | 原子性、order 与 scope 如何建立消息发布？ | fence 需要相应原子读值关系；system GPU-only 不等于 CPU/GPU 并发 |
| [02_async_barriers](02_async_barriers/README.md) | arrive/wait 等哪个 phase，何时可读和覆盖？ | drop/completion/双缓冲与 SM90 transaction 路径分开；阅读具体 sanitizer 报告 |
| [03_thread_scopes](03_thread_scopes/README.md) | 通信对象、完成范围、pipeline 所有权怎样对应？ | 含能力门控的 system/cluster 示例及 carveout；scope 不等于缓存层级或驻留保证 |
| [04_warp_functions](04_warp_functions/README.md) | mask、shuffle source、width 与共享存储交接是否合法？ | vote/match/reduce/shuffle 不自动发布内存；CUB 临时存储也有参与与复用契约 |

先学内存模型与 warp 参与集合，再进入 split barrier 和 pipeline。每项构建/运行/静态/调试动作按各自 README 执行，没有主题统一入口；历史环境与未验证项在各自 `RESULTS.md` 中，不把迁入称为本次实测。

## 每个协议都画两条边

第一条边是生产者写 → 发布/完成 → 消费者读；第二条边是消费者读完 → 允许复用 → 下一次覆盖。还要验证初始化可见、参与集合、scope 覆盖双方、phase 计数和进展条件。`__syncwarp` 只协调相应 warp 集合，不能当作跨 warp/barrier；普通 launch 的 device scope 也不能保证全 grid 同时驻留。

Cooperative Groups 的 group sync、cooperative grid 与 cluster 是不同层级；已有 cluster 计数不覆盖所有分布式共享内存或远端 barrier 行为。cluster 要求 CC 9.0+ 和相应启动/资源条件，不宣称历史 `sm_89` 支持。内存模型与正确性推导见 [ARCHITECTURE.md](../ARCHITECTURE.md)；host 侧 stream/event 顺序见 [异步主题](../07_async/README.md)。

## 保留的同步成本规划

| 子题 | 计划回答 |
| --- | --- |
| `barrier` | 哪些线程之间需要同步？去掉或移动 barrier 会产生什么错误？ |
| `atomic` | 冲突度增加后，吞吐与争用如何变化？ |
| `memory_order` | 可见性、顺序和同步范围应如何选择？ |

错误案例需要独立于正常性能基线运行，并保存 Compute Sanitizer 原始报告；有数据竞争或未定义行为的结果不能当作性能优化结论。现有 [转置 kernel](../../cuda-sdk-lab/examples/vector_transpose.cu)含 `__syncthreads()`，检查方法见 [当前边界尺寸校验动作说明](../../cuda-sdk-lab/experiments/validation/README.md)。原 racecheck 记录已不可用，不再引用其退出码或将其作为实测证据；当前 README 是操作说明，不证明检查通过。同步位置和争用程度的单变量实验仍待实施。

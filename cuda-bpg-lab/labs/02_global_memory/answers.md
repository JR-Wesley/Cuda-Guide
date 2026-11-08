# Stage 2 参考答案

1. scalar 对应 4/8/12/32 sectors；起点 `4*lane*stride`，每 lane 覆盖 4 字节。gather 写入仍连续，每完整 warp 4 sectors。这里只统计一次 warp 指令。
2. 读取 byte 4..131，覆盖 5 sectors；相邻 warp 可以复用边缘 sector。请求数不等于 cache miss，更不等于 DRAM bytes，仍需各层指标。
3. Requested=8,388,608 B；input span=`((1048576-1)*8+1)*4=33,554,404 B`；output span=4,194,304 B；前缀=12 B；allocation=37,748,720 B。E2E 传输全部分配，但逻辑有效数据仍是 8N B。
4. 完整 vector 数 1，tail 数 3，工作范围 max(1,1,3)=3 个线程。线程0复制 float4；线程0/1/2 复制第4/5/6个元素。只按完整 vector 数启动会漏掉尾部，测试覆盖了 block=1。
5. float4 warp 处理 128 floats，scalar warp 处理 32；理想 sector 利用率相同。向量化可能减少指令/调度，但资源、grid、尾部与最终指令选择都会影响性能。
6. 没有规定排序。检查原始样本和参数；顺序进程中小幅差异需 A/B/B/A 及工作集对照，不能宣布普遍加速。
7. 地址跨度、cache sector footprint、页/TLB 分布、传输量与访问映射同时变化。可增加固定分配范围内的地址置换实验，或先限制较小工作集再查询 cache/DRAM 指标；每个对照说明其假设。
8. 无计数器仍能确认 CPU 正确性、已测时间、实际访问宽度和地址模型；不能宣称测出了命中率、DRAM 事务或某类 stall。

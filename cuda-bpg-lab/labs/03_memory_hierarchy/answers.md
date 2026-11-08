# Stage 3 参考答案

1. Requested=8,388,608 B；shared=4096/4224 B。Requested 是源级逻辑 I/O，不是 DRAM bytes；静态 shared 可由 ptxas、cuobjdump、Runtime attributes 核对。
2. 固定列c，pitch32时所有 lane 都是 bank c；pitch33时是 `(c+lane)%32`。不同 lane 访问不同 word，故不是同址广播。边缘 partial warp 的冲突程度按有效地址集合另算。
3. 有效 store 条件是原矩阵行 `blockY*32+tx<R` 和列 `blockX*32+ty+j<C`，它们正是对应 shared 元素的有效 load 条件；全 block barrier 在所有 load 之后。提前 return 会让部分线程不参加所需同步，必须避免。
4. 两者 requested 都是 `4*65536*(256+1)=67,371,008 B`，add数=16,777,216；输入唯一字节为65,536/67,108,864，输出均262,144 B。逻辑 AI 相同，约0.249 FLOP/B；E2E 传输量分别327,680/67,371,008 B。不能混用逻辑请求与物理传输。
5. 下一次读地址由索引计算，不依赖上次加载结果；很多线程并行，测的是整体工作完成时间。曲线可能受缓存、带宽、循环/加法依赖、调度影响，不能直接等同单次 load latency。
6. 每轮旋转是0..31上的置换；每个独立值更新一次，值与值之间无依赖。最后按相同顺序求和，不能为方便改变归约次序后仍宣称逐位一致。这里使用显式 FMA 和指定容差。
7. dynamic_index 可能已经在 PTX 声明 local 数组并出现 ld/st.local；register_limited 的索引仍静态，但后端受预算约束可能产生 SASS spill，同时 ptxas 报告 spill loads/stores。stack非零本身不能分辨这两种原因，也不是运行流量。
8. 更紧预算可能引入 local spill 与额外指令/依赖；occupancy 上界不是 achieved 值，也不是吞吐。检查编译资源、local 指令/流量和调度指标，再用独立计时检验。没有指标时只保留有限推断。
9. 没有预设数值。小形状可能被 launch、barrier和不满 tile 主导，大形状的访存/复用更重要；给出原始样本、误差、配置和反例，不能只挑最快结果。

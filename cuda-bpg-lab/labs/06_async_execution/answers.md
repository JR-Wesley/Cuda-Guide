# Stage 6 参考答案

1. 16 chunks；H2D/D2H各4 MiB，合计8 MiB；kernel逻辑读写另为8 MiB；device分配2×65,536×3×4=1,572,864 B=1.5 MiB；FLOP=268,435,456。PCIe传输和kernel设备读写属于不同路径，不能相加伪装为单层带宽。
2. c=0..6的(slot,worker)依次为(0,0)、(1,1)、(0,2)、(1,0)、(0,1)、(1,2)、(0,0)。从c2开始等c−2的D2H后事件；没有等待，新H2D可覆盖前一kernel输入，新kernel可覆盖尚未回传的output。各chunkhost输出虽独立，不能保护共享device slot。
3. 所有worker在自己的全部chunk之后记录done，coordinator等待每个done之后才记录stop，因此stop依赖所有工作。没有join只测到coordinator自己完成记录，可能遗漏worker工作。wait在调用时捕获最近一次record的状态，后来record不会重新定向已排队的wait；这个语义必须查Runtime文档，不能根据某次测试“没挂”推定。
4. ceil(N/chunk)=17，尾部17元素，H2D/D2H各68 B，grid1。整除测试会漏掉越界copy、kernel尾部、未写output以及chunk>N的分配边界。
5. 串行16×80=1280 us；理想流水80+15×35=605 us；模型加速约2.116。提交与事件开销、copy引擎/链路争用、kernel资源竞争、尾块不等长、buffer不足、pageable阻塞、频率/调度漂移等均会偏离。它是带前提的参考模型，不是硬件保证。
6. 报告“观察到时间交集，但普通整流水耗时未改善”，不能撤销真实交集，也不能称为有效加速。可固定其他变量把chunk增大4倍：若API/事件数量下降且端到端明显改善，支持原配置受细分提交成本影响；若copy占满而kernel较短，剩余传输瓶颈可能限制收益。一次变化仍可能同时影响grid与buffer容量，需列派生量。
7. Event包围device队列及可能的host提交空隙；host E2E另测一次完整提交至完成；API只测CPU调用，可能阻塞/同步；device copy是GPU活动记录。样本、范围和插桩不同，不能直接相减。pin/unpin在本实验计时外，没有实测其成本。
8. kernel活跃和20、并集15、kernel-kernel并发5；copy并集[2,7)和[12,20)，与kernel并集[0,15)交集为5+3=8 us。正交集证明时间重叠；并发资源争用、提交/依赖和基线不同，8 us不能直接称节省8 us。
9. [0,1366)、[1366,2732)、[2732,4099)，count1366/1366/1367，grid各6。serial也发3次才能固定launch数与分区形状。N65536、jobs4时每job16384元素=64 blocks，相比N4096、jobs4的4 blocks/job，单job更能覆盖24SM；并发收益可能下降，但不由grid数单独保证。
10. 没有唯一性能答案。合格记录需包含机器/编译/输入、CPU全量验证、原始样本、Min/Median/P95、命令和快照；以API correlation核对多stream提交顺序及计数；单独列普通与NSYS耗时，标记未测计数器。负结果与竞争解释必须保留。可对照本阶段[实际分析](analysis.md)，不能抄实测数作为自己的运行结果。

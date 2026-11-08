# Stage 4 参考答案

1. 4099→11→1；requested=4×(4099+11+11+1)=16488 B；两个partial共88 B，含input总分配16484 B。N=1仍launch一个pass，以输出独立结果并保持计时语义；没有“零kernel”有效样本。
2. I1：16777216→65536→256→1，3 passes、67635204 B；I4：16777216→16384→16→1，3 passes、67240068 B；I16：16777216→4096→1，2 passes、67141636 B。控制I；派生变化包括grid、每线程局部sum长度、partial流量、pass数和计算依赖长度，不能宣称只改变occupancy。
3. interleaved分别t=0,2,4…；0,4,8…；0,8,16…。sequential分别连续t<64、t<32、t<16。地址与活跃mask可从源码计算；真实shared transactions、branch/predicate形式与stall必须检查指令和计数器。源级if不保证生成分支。
4. 所有物理lane仍执行shuffle，越界输入的累加初值为0；full mask描述参与线程。提前return会让mask包含缺席lane，且跨warp shared阶段可能造成barrier不完整，不能用“只读前5个值”消除这些参与要求。
5. raw register上界3/4/5/8，再与thread上界6和block上界24取min，得3/4/5/6；粗略occupancy50%/66.7%/83.3%/100%。真实分配粒度或其他kernel限制可能降低结果，API值也只是资源允许量，不是运行时achieved值。
6. 保留结果，否定“occupancy越高一定越快”。静态：ptxas的spill/stack、SASS的local load/store及指令结构；运行期：local/DRAM流量、eligible warps/stall（另需普通重复计时和状态稳定性）。更高理论占用可以与更多spill、较低ILP同时存在；没有计数器不进行唯一归因。
7. reference接近0时相对误差失去良好尺度；sumabs相关界能描述不同加法顺序的最坏舍入。它可能太宽，漏算或错误索引也可能偶然落在界内；ones的整数可表示性给零容差对照，再结合多种非整形状、不同输入、sanitizer及算法证明，增强验证证据。
8. 140 us包含launch与pass之间的间隙，三pass活跃时间的和只有103 us；若这些数来自同一个op，可以检查剩余的时间线区间。不同运行或不同统计量的Median不宜直接相减作精确分解。先核对kernel数、时间戳与API/stream关系。
9. 报告每轮原始样本、顺序、同配置漂移和结果范围，降级为本次观测；不要筛选低噪声的有利轮次。下一轮固定工作量与环境，明确预先判定的稳定性标准，再调查较大的差异或计数器。交错顺序只能减少一部分混杂。

10. d=2*stride是正的2次幂，无符号t有`t%d=t&(d−1)`；因此条件、树和地址不变。静态序列消失只能证明编译结果改变，总耗时还包含加载、同步及launch间隙，需未插桩同工作量对照。

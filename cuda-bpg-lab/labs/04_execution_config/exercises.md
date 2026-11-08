# Stage 4 练习

先完成推导，再看[独立答案](answers.md)。需要运行的题目请另建记录，不能将下列问题中的数值假设当作实测。

1. N=4099、B=128、I=3时，计算各pass输入/输出、requested bytes、两个partial缓冲区的总分配。为什么至少有一个pass？
2. N=2^24、B=256，I分别1、4、16时计算pass数与requested bytes。哪个变化是控制量，哪些是派生变化？
3. B=128，画interleaved前三层的活跃线程编号；与sequential相比，活跃lane如何分布？不能只靠图证明哪项性能？
4. 只剩5个有效输入的最后一个warp，为什么仍可使用full shuffle mask？若越界线程提前return会发生什么问题？
5. 假定SM有65536 registers、1536 threads、最多24 blocks，B=256，R=71/64/48/32。计算粗略block上界与warp占用；为何仍要调用occupancy API？
6. 限制预算后occupancy从50%升到100%，但时间增加5倍，应否删除负面结果？列出两项静态证据和两项运行期指标以区分解释。
7. 为什么对强抵消归约只用`rtol*abs(reference)`可能不合适？本实验为何还需要ones精确检查，而不能只依赖gamma上界？
8. NSYS看到三种pass耗时1、2、100 us，而Event op为140 us，能否声称算术耗时140 us？下一项应检查什么？
9. 固定N/B/I，对三variant交错重复并保存顺序。若差异小于同配置漂移，如何写结论和下一轮假设？本题为扩展，未执行时明确标记。

10. 证明interleaved_mask与interleaved的活跃线程相同；SASS中MUFU.RCP消失是否足以证明端到端加速？

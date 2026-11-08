# Stage 5 练习

先按[分析流程](../../docs/analysis-workflow.md)写预测，再看[答案](answers.md)。扩展实验未执行时不得填入预期性能。

1. 对unsigned x=0、7、8、UINT_MAX计算x/8与x>>3；对signed x=−7、−8、INT_MIN说明为什么直接移位或先取signed绝对值可能错误。
2. 为什么constant7必须与runtime d7比较？若改成d8而结果改变，能否仅以速度更快称为优化？
3. 解释`state=mix(state)^q`怎样避免商快速归零与死代码；仍需哪些PTX/SASS检查？
4. N=65536、rounds128、chains4，计算update数、requested bytes；如何与chains1保持update总数相同？这样还改变了哪些因素？
5. rounds分别7、8、9时，u8为什么都需验证？静态多了指令是否意味着每线程执行更多指令？
6. N=33/64/65时，uniform/interleaved各有多少mixed warp、path B任务？给出重排后的前几项。
7. 短body SASS如果有SEL而没有输入flag控制的BRA，应该怎样修正“分支导致串行”解释？
8. 长body交错layout若约慢1.6倍，为何不能反推出branch efficiency=1/1.6或断言理应正好慢2倍？
9. u8提高整体update/s但使用更多register，是否与S4矛盾？列出应保留的指标及待测项。
10. 扩展：固定输入、编译和总工作量，交错重复runtime/constant，保留顺序；随后改变N，判断收益在哪类规模上消失。写明预先的反证标准。

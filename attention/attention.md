flash attention 论文:https://arxiv.org/pdf/2205.14135

以 flash attention 论文中的GPT2 模型数据来计算。即代码中的 *𝑁 = 1024* 和 *𝑑 = 64* 。

硬件GPU以A100为例：
A100 的 Shared Memory 大小为192KB=196608 B，那么可以计算出这里Flash Attention的分块大小：Bc=M/4/64=768，Br=min(768,64)=64，Tr=1024/64=16，Tc=1024/768=2

### standard attention
### flash attention
### flash attention v2
### flash attention v3
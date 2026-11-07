# Log-Probability 的计算

目标 token 的 log-prob 等于目标 logit 减去全词表的 log-sum-exp。从数值稳定的公式出发，分别对照 Megatron Core 的词表并行、vLLM 的推理评分，以及 Liger-Kernel 的分块计算与反向传播。

# 目标 token 的 log-prob

设模型已经读取前缀 $`x_{\le t}`$，最后一层隐藏状态为 $`h_t\in\mathbb R^D`$，词表大小为 $`V`$。LM head 的权重为 $`W\in\mathbb R^{V\times D}`$，第 $`j`$ 行对应词表中的第 $`j`$ 个 token：

```math
z_t=Wh_t+b\in\mathbb R^V,\qquad
z_{t,j}=w_j^\top h_t+b_j.
```

$`z_t`$ 是整个词表的 **logits**，尚未归一化；$`y_t=x_{t+1}`$ 是需要评分的目标 token ID。以下使用自然对数，先讨论温度为 1、没有额外 logits processor 的情况。

由 softmax 的定义，目标概率及其对数为：

```math
\begin{aligned}
p_t(y_t)
&=\frac{\exp(z_{t,y_t})}{\sum_{j=0}^{V-1}\exp(z_{t,j})},\\[4pt]
\ell_t=\log p_t(y_t)
&=\underbrace{z_{t,y_t}}_{\text{目标 logit}}
-\underbrace{\log\sum_{j=0}^{V-1}\exp(z_{t,j})}_{\text{全词表归一化项}}.
\end{aligned}
```

直接计算指数可能溢出。令 $`m_t=\max_j z_{t,j}`$，从求和项中提出 $`e^{m_t}`$，得到稳定形式：

```math
\begin{aligned}
\operatorname{LSE}(z_t)
&=m_t+\log\sum_{j=0}^{V-1}\exp(z_{t,j}-m_t),\\[4pt]
\ell_t
&=\boxed{z_{t,y_t}-m_t-\log\sum_{j=0}^{V-1}\exp(z_{t,j}-m_t).}
\end{aligned}
```

减去最大值后，指数的输入不大于零。**分子只取一个词表位置，分母仍需覆盖整个词表**；只返回一个 log-prob，并不意味着只计算一个 token 的 logit。

例如 $`z=[2,1,0]`$，目标 ID 为 1，则 $`m=2`$、$`s=1+e^{-1}+e^{-2}`$，得到 $`\log p(1)=1-2-\log s\approx-1.407606`$。把全部 logits 同时加上一个常数，概率和 log-prob 都保持不变。

对应的 PyTorch 基准实现可以直接写成 `gather − logsumexp`：

```python 目标值与归一化项 · 基准实现
import torch


def selected_token_logprobs(logits, target_ids):
    # logits: [..., V]；target_ids: [...]，必须是合法词表 ID
    z = logits.float()
    target_z = z.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    log_z = torch.logsumexp(z, dim=-1)
    return target_z - log_z
```

它与 `logits.float().log_softmax(-1).gather(...)` 在数学上等价，但不显式生成完整的 log-softmax 输出。这里仍然接收完整 logits；低精度输入执行 `.float()` 时也可能产生一份 FP32 副本。

# 已知序列为什么可以并行评分

## 因果关系与位置对齐

生成时，下一个 token 尚未确定，必须先采样才能继续。评分时，序列已经给定，可以把所有位置一起送入模型；因果 attention mask 保证位置 $`t`$ 只能读取 $`x_{\le t}`$，不会提前看到目标 $`x_{t+1}`$。

同一层中的各位置可以并行计算，模型各层仍依次执行。长序列还可以按调度和显存预算分块。已知输入消除了逐 token 采样的依赖，并未消除 Transformer 的层间依赖。

```text 预测位置与目标位置
输入位置      已知前缀             该位置输出评分的目标
   0          x₀                        x₁
   1          x₀, x₁                    x₂
   2          x₀, x₁, x₂                x₃

logits[:, :-1, :]  ←→  input_ids[:, 1:]
     [B, T−1, V]          [B, T−1]
```

序列第一个 token 需要额外的起始上下文，例如 BOS，才能取得对应条件概率。最后一个输入位置的 logits 预测序列之外的下一个 token，不应拿来评分该位置自身。

## 回答区间与 loss mask

若只评价回答部分，应在**目标位置**构造 mask，再与移位后的 logits 对齐。prompt、padding 和不参与目标的工具返回内容需要排除；packing 还需要排除不同样本之间的跨边界预测。

```python 右移目标并选择评分区间 · 示例
# outputs.logits: [B, T, V]
# score_mask: [B, T]，标记需要评分的目标 token
target_ids = input_ids[:, 1:]
target_mask = (
    score_mask[:, 1:].bool()
    & attention_mask[:, 1:].bool()
    & attention_mask[:, :-1].bool()
)
token_logprobs = selected_token_logprobs(
    outputs.logits[:, :-1, :], target_ids
)
sequence_logprob = token_logprobs.masked_fill(~target_mask, 0).sum(-1)
```

这里假设每行是一条未 packing 的序列。若用带有 `-100` 的 `labels` 代替 `input_ids`，需要先把忽略位置替换成合法 ID，再执行 `gather`，最后用 mask 排除；`-100` 本身不能作为词表索引。

回答序列的条件 log-prob 是有效位置之和。按有效 token 数求平均得到长度归一化分数，两者对应不同的评分方式。训练时保留计算图，冻结教师或离线评分时可以在外层使用 `torch.no_grad()`。

# 训练引擎：Megatron Core 的词表并行

Megatron Core 的 `vocab_parallel_cross_entropy` 接收已经沿词表切分的 logits。下面对应 [固定版本 `37cb4f2` 的实现](https://github.com/NVIDIA/Megatron-LM/blob/37cb4f261f103a7c882b5e310082fd0b7dbb226f/megatron/core/tensor_parallel/cross_entropy.py#L159)，讨论 `label_smoothing=0`。

## 三个统计量如何跨卡合并

把完整词表 $`\mathcal V`$ 切为 $`R`$ 个互不重叠的子集 $`\mathcal V_r`$。每个 TP rank 只持有本地 logits，形状为 `[T, B, V_local]`，目标仍使用**全局 token ID**。

第一步，每张卡先求本地最大值，再做 `AllReduce(MAX)`：

```math
m_t^{(r)}=\max_{j\in\mathcal V_r}z_{t,j},\qquad
m_t=\max_r m_t^{(r)}.
```

第二步，所有 rank 使用同一个全局最大值 $`m_t`$，计算本地指数和；通过 `AllReduce(SUM)` 得到全词表分母：

```math
s_t^{(r)}=\sum_{j\in\mathcal V_r}\exp(z_{t,j}-m_t),\qquad
s_t=\sum_r s_t^{(r)}.
```

第三步，只有持有目标 token 的 rank 提供移位后的目标 logit，其余 rank 提供零，再做一次 `AllReduce(SUM)`：

```math
a_t^{(r)}=
\begin{cases}
z_{t,y_t}-m_t,& y_t\in\mathcal V_r,\\
0,& y_t\notin\mathcal V_r,
\end{cases}
\qquad
a_t=\sum_r a_t^{(r)}.
```

因此 $`\ell_t=a_t-\log s_t`$。源码在全局最大值就绪后，先归约目标值，再归约指数和；这两个 SUM 的作用分别是取回目标分子和合并归一化分母。

注意，本地指数和使用的是**全局**最大值。如果先按各自的 $`m_t^{(r)}`$ 求和，则合并前必须乘以 $`\exp(m_t^{(r)}-m_t)`$，不能直接相加。

## 从交叉熵取得 log-prob

`calculate_cross_entropy_loss()` 的核心为：

```python Megatron Core · 交叉熵归约
loss = torch.log(sum_exp_logits) - predicted_logits
```

其中 `predicted_logits` 已经减去全局最大值，对应 $`a_t`$；`sum_exp_logits` 对应 $`s_t`$。普通硬标签交叉熵满足：

```math
\mathrm{CE}_t=\log s_t-a_t=-\ell_t.
```

直接取已有 autograd 算子的负值即可：

```python Megatron Core · 调用示例
from megatron.core.tensor_parallel.cross_entropy import (
    vocab_parallel_cross_entropy,
)

# 已初始化 TP 进程组，且已经完成预测位置与目标的对齐
# sharded_logits: [T, B, V_local]；target_ids: [T, B]
token_logprobs = -vocab_parallel_cross_entropy(
    vocab_parallel_logits=sharded_logits,
    target=target_ids,
    label_smoothing=0.0,
    tp_group=tp_group,
)
```

这条路径传输的是每个位置的统计量，无需 all-gather 完整词表 logits。反向传播由自定义 autograd 实现处理：它保存本地 softmax 分片，再计算本地 $`p-\mathrm{onehot}(y)`$。因此，**避免聚合完整词表**与**不保存本地词表激活**是两件事；这里的普通实现仍会保存本地 softmax 张量。

# 推理引擎：vLLM 的 log-prob 路径

以下对照 [vLLM 固定版本 `dfebb42`](https://github.com/vllm-project/vllm/tree/dfebb42d096d5c6a04aa5a1722195e37cb8e0b2d)。不同 worker 路径的中间张量不同，不能仅凭接口都叫 log-prob 就推断它们采用同一套融合方式。

## 常规 sampler：log-softmax 后取目标值

[`Sampler.compute_logprobs()` 与 `gather_logprobs()`](https://github.com/vllm-project/vllm/blob/dfebb42d096d5c6a04aa5a1722195e37cb8e0b2d/vllm/v1/sample/sampler.py#L369) 先得到全词表 log-probs，再提取实际 token。省略 top-k、rank 和返回结构后，计算为：

```python vLLM · 常规 sampler 核心摘录
logprobs = logits.log_softmax(dim=-1, dtype=torch.float32)
token_ids = token_ids.unsqueeze(-1)
token_logprobs = logprobs.gather(-1, token_ids)
```

输入为 `[N, V]` logits，中间会产生 `[N, V]` 的 FP32 log-probs，最后只保留所请求的值。这里 $`N`$ 表示本次参与评分的位置数。

## 指定 token：融合归约与 gather

GPU worker 中的 [`compute_token_logprobs()`](https://github.com/vllm-project/vllm/blob/dfebb42d096d5c6a04aa5a1722195e37cb8e0b2d/vllm/v1/worker/gpu/sample/logprob.py#L94) 接收 logits 和 `[N, K]` 个指定 ID，调用 `_topk_log_softmax_kernel`。它并不在这些 ID 内重新归一化，而是扫描整个词表：

1. 第一遍按块加载 logits，得到行最大值 `max_val`。
2. 第二遍把加载的值转为 FP32，累计 `exp(logits - max_val)` 的和 `se`。
3. 再加载指定 token 的 logits，只写出所请求的 $`K`$ 个 log-prob。

```python vLLM · 词表归约后的输出步骤
lse = tl.log(se)

# 此处 logits 是重新加载的指定 token 的值
logits = logits.to(tl.float32)
o = logits - max_val - lse
```

源码变量 `lse` 是 $`\log s_t`$，完整的 $`\operatorname{LSE}(z_t)`$ 应为 `max_val + lse`。该 kernel 省去了完整 `[N, V]` **log-probs 输出**的物化，但其输入仍是已经计算好的 logits。

## Prompt log-prob：先按位置分块计算 LM head

已知序列的评分还要控制 LM head 的峰值内存。[`compute_prompt_logprobs_with_chunking()`](https://github.com/vllm-project/vllm/blob/dfebb42d096d5c6a04aa5a1722195e37cb8e0b2d/vllm/v1/worker/gpu/sample/prompt_logprob.py#L290) 先切分 hidden states，调用 `logits_fn` 得到当前块的词表 logits，再计算目标值、top-k 和 rank。

```text vLLM · prompt 评分的数据流
已知序列 → 因果前向 → hidden states
                         │ 按 token 位置分块
                         ▼
                    logits_fn
                         │ 当前块 [C, V] logits
                         ▼
             全词表归约 + 指定 token 输出
                         │ 当前块 [C, K] log-probs
                         ▼
                   拼接评分结果
```

这条路径避免一次保留全序列 logits，仍会物化当前块的 `[C, V]` logits。源码还明确指出 `logits_fn` 可能涉及 all-gather，所以不能把 Megatron 的“仅归约统计量”直接套用到这条推理路径。

若请求整个词表的 log-probs，输出本身仍有 $`O(NV)`$ 大小；指定少量 token 才能把返回结果缩小到 $`O(NK)`$。

# Liger-Kernel：分块计算与训练反向

Liger-Kernel 中存在多种实现。这里重点看与目标 token 评分直接对应的 [PPO 分块 log-prob 路径，固定版本 `4dce24b`](https://github.com/linkedin/Liger-Kernel/blob/4dce24bbb70d8bda18e373b5f0f469a51ab77a2c/src/liger_kernel/chunked_loss/fused_linear_ppo.py#L18)，再与 fused linear cross-entropy 区分。

## 同时切 token 位置和词表

`chunk_forward()` 把 `[B, T, D]` 隐藏状态展平为 `[N, D]`，其中 $`N=BT`$，然后交给 `_ChunkedSelectiveLogProbFunction`。目标 ID 由调用方传入，这个函数本身不负责 causal shift。

`_selective_logprob_forward()` 再按两个维度循环：当前版本内部的位置块大小为 2048，词表块大小为 4096。设两者为 $`C_N`$ 与 $`C_V`$，每次只投影当前子块：

```math
Z_{\mathrm{chunk}}
=H_{\mathrm{chunk}}W_{\mathrm{chunk}}^\top
\in\mathbb R^{C_N\times C_V}.
```

下面是省略 bias 和温度处理后的源码核心：

```python Liger-Kernel · 当前词表块与归一化状态
logits_chunk = (hidden_chunk @ weight_chunk.to(hidden.dtype).t()).float()

chunk_max = logits_chunk.amax(dim=-1)
max_new = torch.maximum(max_old, chunk_max)
rescale = torch.exp(max_old - max_new)
chunk_exp = torch.exp(logits_chunk - max_new.unsqueeze(-1))

sum_exp = sum_exp * rescale + chunk_exp.sum(dim=-1)
max_old = max_new
```

这里的矩阵乘法采用 `hidden.dtype`，之后才转为 FP32 做归约；不能把它描述为 FP32 LM head。它生成实际的 PyTorch 分块张量，限制的是临时张量规模，**并不保证块内 logits 只存在于 SRAM**。

## Online log-sum-exp 为什么需要重缩放

将词表分成若干块 $`\mathcal C_k`$。对一个固定 token，处理到第 $`k`$ 块后，维护已访问 logits 的最大值 $`m^{(k)}`$，以及相对于这个最大值的指数和 $`s^{(k)}`$：

```math
\begin{aligned}
m^{(k)}
&=\max\left(m^{(k-1)},\max_{j\in\mathcal C_k}z_j\right),\\[4pt]
s^{(k)}
&=s^{(k-1)}\exp\left(m^{(k-1)}-m^{(k)}\right)
+\sum_{j\in\mathcal C_k}\exp\left(z_j-m^{(k)}\right).
\end{aligned}
```

初始状态为 $`m^{(0)}=-\infty`$、$`s^{(0)}=0`$。旧累积和以旧最大值为基准；当最大值增大时，每一项都需要乘上 $`\exp(m^{(k-1)}-m^{(k)})`$。因此无需重新读取旧块，就能把新旧两部分放到同一个尺度上。

这对应 [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) 的在线归一化思路。数学上等价于全词表归约，实际浮点结果仍可能因分块顺序而有舍入差异。

目标 logit 也在扫描过程中取得：目标落在当前词表块时，使用全局 ID 减去块起点得到局部索引，累积该位置的值。所有词表块处理完后：

```python Liger-Kernel · 完成当前位置块的评分
log_z_chunk = max_old + torch.log(sum_exp)
log_z[seq_start:seq_end] = log_z_chunk
logprobs[seq_start:seq_end] = target_logit - log_z_chunk
```

返回值是目标 log-probs；`log_z` 则保存每个位置的完整 LSE，供反向使用。没有保存整张 `[N, V]` logits 或 softmax，并不意味着只保留这两个输出：隐藏状态、LM head 权重、目标 ID 等仍是反向需要的输入。

## 只取一个 log-prob，梯度为什么仍遍历词表

对 $`\ell_t=z_{t,y_t}-\operatorname{LSE}(z_t)`$ 求导：

```math
\frac{\partial\ell_t}{\partial z_{t,j}}
=\mathbf1[j=y_t]-p_t(j).
```

设上游损失传来的梯度为 $`g_t=\partial L/\partial\ell_t`$，则：

```math
G_{t,j}=\frac{\partial L}{\partial z_{t,j}}
=g_t\left(\mathbf1[j=y_t]-\exp(z_{t,j}-\operatorname{LSE}(z_t))\right).
```

因此，只要保存每个位置的 LSE，就能在反向时重新计算当前块的 logits，恢复该块的概率和梯度。[`_selective_logprob_backward()`](https://github.com/linkedin/Liger-Kernel/blob/4dce24bbb70d8bda18e373b5f0f469a51ab77a2c/src/liger_kernel/chunked_loss/fused_linear_ppo.py#L73) 正是沿位置和词表再次分块：先形成 $`-g_t p_t(j)`$，再在目标位置加上 $`g_t`$。

若忽略 bias、温度为 1，对 $`Z=HW^\top`$ 有：

```math
\nabla_H L=GW,\qquad
\nabla_W L=G^\top H.
```

代码把每一块对隐藏状态和权重的梯度累积到对应缓冲区；LM head 权重被冻结时，可以跳过权重梯度。这里节省的是全词表中间激活，模型参数和需要训练的参数梯度仍然占用内存。

普通交叉熵对应 $`g_t=-1`$，恢复 $`p_t-\mathrm{onehot}(y_t)`$。对于简化的策略梯度目标 $`L_t=-\operatorname{stopgrad}(A_t)\ell_t`$，有 $`g_t=-A_t`$；PPO、GRPO 中的比例、裁剪和 KL 项则通过各自的上游梯度进入，不能一概写成负 log-prob。

## Fused linear cross-entropy 是另一条实现

[`ops/fused_linear_cross_entropy.py`](https://github.com/linkedin/Liger-Kernel/blob/4dce24bbb70d8bda18e373b5f0f469a51ab77a2c/src/liger_kernel/ops/fused_linear_cross_entropy.py#L146) 主要沿 token 位置分块。每块先计算 `[C_N, V]` logits，再交给 Triton cross-entropy kernel 完成归约；这与上面的“双维分块 + 反向重计算”不能混为一谈。

当关闭 label smoothing、类别权重、额外 z-loss 等修改，且采用 `reduction="none"` 时，有效 token 的 CE 取负可对应它所评分分布的 log-prob；若还要求原始 logits 分布，需要保持温度为 1，并关闭 softcap 等变换。

# 三种实现分别省掉了什么

令 $`N`$ 为评分位置数，$`K`$ 为每个位置请求的 token 数，$`R`$ 为 TP rank 数。下表描述本文核对的具体路径，不代表每个框架的全部后端。

| 路径 | 主要词表中间张量 | 优化位置 |
| --- | --- | --- |
| Megatron 普通词表并行 CE | 每卡 `[N, V/R]` logits / softmax | 归约统计量，避免 all-gather 完整词表 |
| vLLM 常规 sampler | `[N, V]` logits 与 log-probs | 计算后选取目标值 |
| vLLM GPU prompt 分块评分 | 当前块 `[C_N, V]` logits | 按位置限制峰值，融合归约后只输出请求值 |
| Liger PPO selective log-prob | 当前块 `[C_N, C_V]` logits 等临时张量 | 双维分块，保存 LSE，反向重计算 |
| Liger fused linear CE | 当前块 `[C_N, V]` logits | 位置分块与 CE kernel 融合 |

以 $`N=4096`$、$`V=152064`$ 为例，一张 BF16 logits 张量占用 $`4096\times152064\times2=1{,}245{,}708{,}288`$ 字节，约 1.16 GiB；同形状的 FP32 log-probs 约 2.32 GiB。分块和融合减少大张量的同时驻留及读写，但完整、精确的 softmax 归一化仍需遍历全词表。具体耗时还取决于矩阵乘法、带宽、通信、分块大小和后端，不能仅由最终输出大小推断加速比例。

训练与推理对齐时，还要确保评分的是同一种分布：温度 $`\tau>0`$ 对应 $`\operatorname{logsoftmax}(z/\tau)`$；top-k、top-p、惩罚或约束处理后的采样分布，需要按其处理后的 logits 重新归一化。原始模型的 log-prob 与采样分布的 log-prob 应分别核对。

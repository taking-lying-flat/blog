# Log-Probability 的计算

# Log-Probability Computation

设训练序列为 $`x=(x_0,\ldots,x_T)`$，$`x_0`$ 为起始 token。自回归预训练监督后续所有 token；仅监督回答的 SFT 中，$`x_{1:M}`$ 为提示词，$`x_{M+1:T}`$ 为参考回答。两者使用相同的负对数似然目标，区别在于参与监督的位置：

```math
\begin{aligned}
\mathcal L_{\mathrm{PT}}(\theta)
&=-\frac{1}{T}\sum_{t=0}^{T-1}\ln P_\theta(x_{t+1}\mid x_{\le t}),\\[4pt]
\mathcal L_{\mathrm{SFT}}(\theta)
&=-\frac{1}{T-M}\sum_{t=M}^{T-1}\ln P_\theta(x_{t+1}\mid x_{\le t}).
\end{aligned}\tag{1}
```

对单个位置，硬标签交叉熵（CE）与负对数似然（NLL）相同。令 $`z_t\in\mathbb R^V`$ 为该位置的词表 logits，$`q_{t,j}=\mathbf 1\{j=x_{t+1}\}`$ 为目标 token 的 one-hot 标签，则

```math
\begin{aligned}
\ell_t
&=\operatorname{CE}(q_t,\operatorname{softmax}(z_t))\\[4pt]
&=-\sum_{j=0}^{V-1}q_{t,j}\ln\operatorname{softmax}(z_t)_j\\[4pt]
&=-\ln P_\theta(x_{t+1}\mid x_{\le t})
=\ell_t^{\mathrm{NLL}}.
\end{aligned}\tag{2}
```

隐藏状态 $`h_t`$ 经 LM head 得到 $`z_t=Wh_t+b`$。式 (2) 中的 log-prob 可由目标 logit 与全词表 log-sum-exp 计算：

```math
\begin{aligned}
\ln P_\theta(x_{t+1}\mid x_{\le t})
&=\operatorname{logsoftmax}(z_t)_{x_{t+1}}\\[4pt]
&=\ln\frac{\exp(z_{t,x_{t+1}})}
{\sum_{j=0}^{V-1}\exp(z_{t,j})}\\[4pt]
&=z_{t,x_{t+1}}-\ln\sum_{j=0}^{V-1}\exp(z_{t,j})\\[4pt]
&=z_{t,x_{t+1}}-\ln\left[
\exp(m_t)\sum_{j=0}^{V-1}\exp(z_{t,j}-m_t)\right]\\[4pt]
&=z_{t,x_{t+1}}-
\left[m_t+\ln\sum_{j=0}^{V-1}\exp(z_{t,j}-m_t)\right],
\qquad m_t=\max_j z_{t,j}.
\end{aligned}\tag{3}
```

式 (3) 将 softmax 的分子与分母转为两项之差，再从指数和中提出 $`\exp(m_t)`$，得到避免指数上溢的计算形式。对结果取负，即为式 (2) 的逐 token 损失。

**教师强制（teacher forcing）**将整段真实序列作为 `input_ids` 输入。`input_ids` 经 embedding 和带因果掩码的 Transformer 得到隐藏状态；位置 $`t`$ 只能访问真实前缀 $`x_{\le t}`$，其 logits 监督 $`x_{t+1}`$。各位置的损失在一次前向中并行计算，模型预测不会替换后续位置的输入。预训练与 SFT 都可使用这一过程。

以单轮提示与回答为例，`labels` 初始复制 `input_ids`。仅监督回答时，将提示词对应的标签置为 `-100`；提示词仍完整保留在 `input_ids` 中。

```python Transformers · 预训练与 SFT 调用示例
# 示意 token ID：[BOS, p1, p2, a1, a2, EOS]
input_ids = torch.tensor([[1, 11, 12, 21, 22, 2]], device=model.device)
labels = input_ids.clone()

# 预训练保留完整 labels；仅回答 SFT 增加这一行
labels[:, :3] = -100

loss = model(input_ids=input_ids, labels=labels, use_cache=False).loss
loss.backward()
```

Transformers [`LlamaForCausalLM.forward`（v4.44.2）](https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/llama/modeling_llama.py#L1208) 的 LM head 与损失分支如下。标签错位在模型内部完成，传入的 `labels` 应与 `input_ids` 保持原始位置对应：

```python Transformers · LlamaForCausalLM.forward
logits = self.lm_head(hidden_states)
logits = logits.float()

if labels is not None:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, self.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
```

`logits[..., :-1, :]` 与 `labels[..., 1:]` 将当前位置的预测对齐到下一 token。上述 SFT 示例中的目标依次为 `[-100, -100, a1, a2, EOS]`：`p2` 位置预测 `a1`，`a1` 位置预测 `a2`，`a2` 位置预测 `EOS`。`CrossEntropyLoss` 忽略 `-100`，并在内部完成式 (2) 的 log-softmax 与目标 token 取值。

**Megatron Core**：[词表并行交叉熵](https://github.com/NVIDIA/Megatron-LM/blob/37cb4f261f103a7c882b5e310082fd0b7dbb226f/megatron/core/tensor_parallel/cross_entropy.py#L159)将式 (3) 的计算分布到各个 tensor parallel rank。MAX 归约得到全词表最大值，SUM 归约合并目标 logit 与指数和。核心前向取 `label_smoothing=0`：

```python Megatron Core · vocab_parallel_cross_entropy
import torch.distributed as dist

z = vocab_parallel_logits.float()                 # [..., V_local]
m = z.amax(dim=-1)
dist.all_reduce(m, op=dist.ReduceOp.MAX, group=tp_group)
z = z - m.unsqueeze(-1)

outside = (target < vocab_start_index) | (target >= vocab_end_index)
local_id = (target - vocab_start_index).masked_fill(outside, 0)
target_z = z.gather(-1, local_id.unsqueeze(-1)).squeeze(-1)
target_z = target_z.masked_fill(outside, 0)
sum_exp = z.exp().sum(dim=-1)

dist.all_reduce(target_z, op=dist.ReduceOp.SUM, group=tp_group)
dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=tp_group)
log_prob = target_z - sum_exp.log()               # 等于 -CE
```

`target` 为全局词表 ID，只有持有该 token 的 rank 提供目标值。`target_z` 已减去全局最大值 $`m_t`$，最终减去 `sum_exp.log()` 得到 log-prob，即逐 token 交叉熵的相反数。

**vLLM**：[指定 token 的 kernel](https://github.com/vllm-project/vllm/blob/dfebb42d096d5c6a04aa5a1722195e37cb8e0b2d/vllm/v1/worker/gpu/sample/logprob.py#L17)在同一个 kernel 中完成全词表归约和目标值提取。`row_ptr` 指向当前行，`topk_ids` 为请求评分的词表 ID：

```python vLLM · _topk_log_softmax_kernel
max_val = float("-inf")
for i in range(0, vocab_size, BLOCK_SIZE):
    block = i + tl.arange(0, BLOCK_SIZE)
    z = tl.load(row_ptr + block, mask=block < vocab_size, other=-float("inf"))
    max_val = tl.max(tl.maximum(z, max_val))
max_val = max_val.to(tl.float32)

sum_exp = 0.0
for i in range(0, vocab_size, BLOCK_SIZE):
    block = i + tl.arange(0, BLOCK_SIZE)
    z = tl.load(row_ptr + block, mask=block < vocab_size, other=0.0)
    e = tl.exp(z.to(tl.float32) - max_val)
    sum_exp += tl.sum(tl.where(block < vocab_size, e, 0.0))

target_z = tl.load(row_ptr + topk_ids).to(tl.float32)
log_prob = target_z - max_val - tl.log(sum_exp)
```

两次遍历分别计算整行最大值与 FP32 指数和。`target_z` 为原始 logit，因此同时减去 `max_val` 和 `log(sum_exp)`；分母仍覆盖完整词表。

**Liger-Kernel**：[`_selective_logprob_forward`](https://github.com/linkedin/Liger-Kernel/blob/4dce24bbb70d8bda18e373b5f0f469a51ab77a2c/src/liger_kernel/chunked_loss/fused_linear_ppo.py#L18)逐块计算词表投影，并在线更新最大值和指数和。核心计算省略外层位置分块，取 `bias=None`：

```python Liger-Kernel · _selective_logprob_forward
# h: 当前块隐藏状态 [C, D]；targets: [C]；weight: [V, D]
m = torch.full((h.shape[0],), -float("inf"), device=h.device, dtype=torch.float32)
sum_exp = torch.zeros_like(m)
target_z = torch.zeros_like(m)

for start in range(0, weight.shape[0], vocab_chunk_size):
    end = min(start + vocab_chunk_size, weight.shape[0])
    z = (h @ weight[start:end].to(h.dtype).T).float()

    m_new = torch.maximum(m, z.amax(dim=-1))
    sum_exp = sum_exp * torch.exp(m - m_new) + (
        z - m_new.unsqueeze(-1)
    ).exp().sum(dim=-1)
    m = m_new

    in_chunk = (targets >= start) & (targets < end)
    local_id = (targets - start).clamp(0, end - start - 1)
    target_z += z.gather(-1, local_id.unsqueeze(-1)).squeeze(-1) * in_chunk

log_z = m + sum_exp.log()
log_prob = target_z - log_z
```

最大值由 `m` 更新为 `m_new` 时，旧累积和乘以 `exp(m - m_new)`，再与新块的指数和相加。目标 token 所在块提供 `target_z`；遍历结束后的 `log_z` 为全词表 log-sum-exp，并保存供反向重计算使用。

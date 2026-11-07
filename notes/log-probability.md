# Log-Probability 的计算

# Log-Probability Computation

监督微调（SFT）通常在给定真实前缀的条件下，提高模型对参考答案中下一个 token 的预测概率。以 Hugging Face TRL 的 [`SFTTrainer.compute_loss`](https://github.com/huggingface/trl/blob/5873c558c26b1d63960262750ede00cd34369510/trl/trainer/sft_trainer.py#L1657) 为例，下面节选 `loss_type="nll"`、未启用标签平滑时的核心计算，省略其他损失分支与分布式归约修正：

```python TRL · SFTTrainer.compute_loss
outputs = model(**inputs, fused_lm_head=True)
per_token_loss = -outputs.log_probs  # zero on ignored positions
num_tokens = outputs.label_mask.sum() if num_items_in_batch is None else num_items_in_batch
loss = per_token_loss.sum() / torch.as_tensor(num_tokens).clamp(min=1)
```

`outputs.log_probs` 已经是与目标标签对齐的 token 对数概率；被忽略的位置为零。第一步取负，得到逐 token 的负对数似然（negative log-likelihood，NLL）；第二步对有效位置求和，并除以有效 token 数。`num_items_in_batch` 可传入训练器统计的归一化计数；未传入时，使用当前批次的 `label_mask.sum()`。`clamp(min=1)` 则使没有有效标签的批次返回零损失。

令 $`\mathcal I`$ 为一个批次中参与监督的位置集合，$`N=|\mathcal I|>0`$，$`b`$ 表示样本，$`t`$ 表示前缀末尾的位置。按有效 token 平均的 SFT 目标写为

```math
\begin{aligned}
\mathcal L_{\mathrm{SFT}}(\theta)
&=\frac{1}{N}\sum_{(b,t)\in\mathcal I}\ell_{b,t}^{\mathrm{NLL}}\\[4pt]
&=-\frac{1}{N}\sum_{(b,t)\in\mathcal I}
\ln P_\theta\!\left(x_{t+1}^{(b)}\mid x_{\le t}^{(b)}\right).
\end{aligned}\tag{1}
```

这里使用自然对数 $`\ln`$。它作用于模型给目标 token 分配的条件概率；代码中的负号将 log-prob 转为 NLL，最终的 `loss` 再按式 (1) 聚合。padding 和其他被屏蔽的标签不属于 $`\mathcal I`$；例如只监督回答时，提示词仍参与上下文计算，但不计入损失。

NLL 与训练代码中常见的交叉熵（cross-entropy，CE）在硬标签、无标签平滑且无类别加权时等价。令词表大小为 $`V`$，记 $`p_{b,t,j}=P_\theta(j\mid x_{\le t}^{(b)})`$，目标分布 $`q_{b,t,j}=\mathbf 1\{j=x_{t+1}^{(b)}\}`$ 为 one-hot 分布，则词表上的交叉熵求和只保留目标 token 对应的一项：

```math
\begin{aligned}
\ell_{b,t}^{\mathrm{CE}}
&=-\sum_{j=0}^{V-1}q_{b,t,j}\ln p_{b,t,j}\\[4pt]
&=-\ln p_{b,t,x_{t+1}^{(b)}}
=\ell_{b,t}^{\mathrm{NLL}}.
\end{aligned}\tag{2}
```

因此，`cross_entropy(logits, labels)` 中虽然没有显式写出 `ln`，却已经包含了对数归一化：它在数学上等价于先做 `log_softmax`，再对目标标签计算 `nll_loss`。[PyTorch 文档](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html)明确给出了这一等价关系。另一种常见实现是 Transformers 的 [`ForCausalLMLoss`](https://github.com/huggingface/transformers/blob/0f53828e83f2212060233d2ea552cc76617b8712/src/transformers/loss/loss_utils.py#L54)：先将标签与下一 token 对齐，再由 `fixed_cross_entropy` 调用 `nn.functional.cross_entropy`，并通过 `ignore_index=-100` 排除无效标签。它与上面 TRL 直接归约 log-prob 的路径表达同一个 NLL 目标。

接下来展开式 (1) 中的对数概率。省略样本下标 $`b`$，自回归模型读取前缀 $`x_{\le t}`$ 后，得到隐藏状态 $`h_t`$，再经 LM head 投影为全词表 logits：$`z_t=Wh_t+b_{\mathrm{LM}}\in\mathbb R^V`$。其中 $`V`$ 是词表大小，$`z_{t,j}`$ 是第 $`j`$ 个 token 的未归一化得分；当前位置 $`t`$ 的输出用于评分下一个 token $`x_{t+1}`$。取温度为 1，可将 softmax 定义逐步改写为数值稳定的统一计算式：

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

先按 softmax 定义，把目标 token 的指数得分除以整个词表的指数和，得到条件概率。取对数后，分式变为两项之差：分子的 $`\ln\exp(z_{t,x_{t+1}})`$ 还原为目标 logit，分母则留下全词表的 log-sum-exp。因此，计算一个目标 token 的 log-prob，需要取得它自身的得分，以及所有候选共同决定的归一化项。

最后两行处理数值稳定性。将每一项写成 $`\exp(z_{t,j})=\exp(m_t)\exp(z_{t,j}-m_t)`$，从求和中提出公共因子 $`\exp(m_t)`$，再利用 $`\ln\exp(m_t)=m_t`$，便得到最终形式。实现时直接采用最后一行，只计算移位后的指数。减去最大值后，指数的输入都不大于零，至少有一项为零；这既保留了原来的概率，也避免直接对较大的 logits 取指数造成上溢。对式 (3) 取负并按式 (1) 归约，就得到上述 SFT 损失；下面三个实现中的 `log()` 与 `tl.log()` 均表示自然对数。

**Megatron Core**：词表沿 tensor parallel rank 切分，各卡持有自己的 logits 分片。MAX 归约给出全词表最大值，随后各卡在同一尺度上计算目标值和指数和，再通过 SUM 归约合并。下面合并了[源码](https://github.com/NVIDIA/Megatron-LM/blob/37cb4f261f103a7c882b5e310082fd0b7dbb226f/megatron/core/tensor_parallel/cross_entropy.py#L159)中的前向计算，省略 autograd 封装，取 `label_smoothing=0`。

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

`target` 使用全局词表 ID，减去当前分片起点后得到本地索引。只有持有目标 token 的 rank 提供目标值，其余 rank 将这一项置零。这里的 `target_z` 已经减过全局最大值 $`m_t`$，所以最后直接减去 `sum_exp.log()`，便与式 (3) 对应；按式 (2)，它也等于该 token 的交叉熵损失取负。

**vLLM**：[指定 token 的 kernel](https://github.com/vllm-project/vllm/blob/dfebb42d096d5c6a04aa5a1722195e37cb8e0b2d/vllm/v1/worker/gpu/sample/logprob.py#L17) 在同一个 kernel 中完成全词表归约和目标值提取。下面省略指针步长与输出存储；`row_ptr` 指向当前行，`topk_ids` 是请求评分的词表 ID。

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

第一次遍历得到整行最大值 `max_val`，第二次以它为基准，用 FP32 累计指数和。最后加载的 `target_z` 是目标 token 的原始 logit，因此要同时减去 `max_val` 和 `log(sum_exp)`。即使只请求少量 token，归一化项仍由整个词表计算，输出阶段才选取指定位置。

**Liger-Kernel**：[`_selective_logprob_forward`](https://github.com/linkedin/Liger-Kernel/blob/4dce24bbb70d8bda18e373b5f0f469a51ab77a2c/src/liger_kernel/chunked_loss/fused_linear_ppo.py#L18) 从隐藏状态出发，逐块计算词表投影，并在线更新最大值和指数和。每处理一个词表块，就把它的统计量并入已有结果。下面保留核心计算，省略外层位置分块，取 `bias=None`。

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

循环中的 `sum_exp` 始终以当前最大值 `m` 为基准。当新块使最大值增大到 `m_new` 时，旧累积和乘以 `exp(m - m_new)`，便转换到新的尺度，再与当前块的指数和相加。这样可以逐块完成全词表归一化。

目标 token 所在的块同时提供 `target_z`。遍历完成后，`m + log(sum_exp)` 就是完整的 log-sum-exp，最后相减得到 log-prob。该实现会保存 `log_z`，在反向重算当前块的 logits 时恢复其概率。

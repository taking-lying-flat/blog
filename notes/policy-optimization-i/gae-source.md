### 源码：OpenRLHF 如何反向计算 GAE

`get_advantages_and_returns` 接收形状为 `[batch_size, response_length]` 的 `values`、`rewards` 和有效 token 掩码。下面是函数的计算主体；外层使用 `@torch.no_grad()`，优势估计不参与反向传播。

```python get_advantages_and_returns · 计算主体 | OpenRLHF/OpenRLHF · dc2a7ad
lastgaelam = 0
advantages_reversed = []
response_length = rewards.size(1)

# Mask invalid responses
if action_mask is not None:
    values = action_mask * values
    rewards = action_mask * rewards

for t in reversed(range(response_length)):
    nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
    delta = rewards[:, t] + gamma * nextvalues - values[:, t]
    lastgaelam = delta + gamma * lambd * lastgaelam
    advantages_reversed.append(lastgaelam)
advantages = torch.stack(advantages_reversed[::-1], dim=1)
returns = advantages + values
return advantages.detach(), returns
```

[对应源码](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/trainer/ppo_utils/experience_maker.py#L371-L387)。

`delta` 对应 $`\delta_t=r_t+\gamma V_{t+1}-V_t`$；倒序循环中的 `lastgaelam` 对应 $`\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}`$。循环结束后把时间顺序翻回来，并用 `advantages + values` 构造价值函数的 λ-return 目标。它在 `λ=1` 时才退化为对应的 Monte Carlo 回报。

这段实现把 response 末端的 `nextvalues` 设为 `0.0`，并在递推前将无效 token 的奖励和价值置零。对于完整响应，这是它采用的边界处理；若把任意环境轨迹截成片段，还需要区分真正终止与时间截断，后者应使用末端价值进行 bootstrap。`rewards` 可以已经包含 KL 惩罚，不一定只有终止时的奖励模型分数。

源码：OpenRLHF，按 [Apache License 2.0](assets/licenses/OpenRLHF-LICENSE.txt) 摘录。

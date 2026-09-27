### 源码：OpenRLHF 的 PPO 截断损失

下面选取 `PolicyLoss.forward` 的标准 PPO 路径：`policy_loss_type="ppo"`、`dual_clip=None`，且未启用训练策略与 rollout 策略之间的额外重要性修正。省略其他分支，保留原计算及最终的有效 token 聚合。

```python PolicyLoss.forward · 标准 PPO 路径 | OpenRLHF/OpenRLHF · dc2a7ad
raw_policy_log_ratio = log_probs - old_log_probs
if self.policy_loss_type == "ppo":
    policy_log_ratio = raw_policy_log_ratio.clamp(min=-20.0, max=20.0)
    ratio = policy_log_ratio.exp()
surr1 = ratio * advantages
surr2 = ratio.clamp(1 - self.clip_eps_low, 1 + self.clip_eps_high) * advantages

if self.dual_clip is None:
    # Standard PPO
    loss = -torch.min(surr1, surr2)
loss = aggregate_loss(
    loss,
    action_mask,
    token_level_loss=self.token_level_loss,
    dp_size=dp_size,
    batch_num_tokens=batch_num_tokens,
    global_batch_size=global_batch_size,
)
```

[对应源码](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/models/loss.py#L185-L256)。

`exp(log_probs - old_log_probs)` 对应概率比 $`r_t(\theta)`$。`surr1` 和 `surr2` 分别是原代理目标和截断代理目标，`-torch.min(...)` 把最大化目标转成优化器可最小化的损失。`aggregate_loss` 再按 `action_mask` 和批次设置聚合有效 token；源码允许上下两侧使用不同的截断阈值。

`policy_log_ratio.clamp(-20, 20)` 是指数计算前的数值保护，`ratio.clamp(1-ε, 1+ε)` 才是 PPO 的截断项；二者作用不同。后者仍只是目标函数的一部分，不保证更新后的实际策略概率比落在该区间。

源码：OpenRLHF，按 [Apache License 2.0](assets/licenses/OpenRLHF-LICENSE.txt) 摘录。

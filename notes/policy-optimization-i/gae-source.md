```python GAE · 计算优势与回报 | OpenRLHF · PyTorch
@torch.no_grad()
def get_advantages_and_returns(
    self,
    values,
    rewards,
    action_mask,
    gamma,
    lambd,
):
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

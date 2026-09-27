```python PPO · 策略梯度目标（含 Dual-clip） | verl · 节选
def compute_policy_loss_vanilla(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    loss_agg_mode="token-mean",
    config=None,
    rollout_is_weights=None,
):
    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get(
        "clip_ratio_c", 3.0
    )

    negative_approx_kl = log_prob - old_log_prob

    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - clip_ratio_low, 1 + clip_ratio_high
    )
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    return pg_loss
```

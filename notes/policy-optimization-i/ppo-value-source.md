```python PPO · 价值函数目标 | verl · 节选
def compute_value_loss(
    vpreds,
    returns,
    values,
    response_mask,
    cliprange_value,
    loss_agg_mode="token-mean",
    dp_size=1,
    batch_num_tokens=None,
    global_batch_size=None,
    loss_scale_factor=None,
):
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(
        loss_mat=clipped_vf_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        dp_size=dp_size,
        batch_num_tokens=batch_num_tokens,
        global_batch_size=global_batch_size,
        loss_scale_factor=loss_scale_factor,
    )
    return vf_loss
```

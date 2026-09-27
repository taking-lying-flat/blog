```python PPO · 策略与价值网络共用的梯度更新 | verl
def train_batch(self, data, loss_function):
    maybe_fix_3d_position_ids(data)

    self.optimizer_zero_grad()
    outputs = self.forward_backward_batch(data, loss_function, forward_only=False)
    grad_norm = self.optimizer_step()
    if self.is_mp_src_rank_with_outputs():
        assert "grad_norm" not in outputs["metrics"]
        outputs["metrics"]["grad_norm"] = grad_norm
    return outputs
```

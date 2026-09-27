```python PPO · 策略与价值网络共用的梯度更新 | verl · 节选
def train_batch(self, data, loss_function):
    self.optimizer_zero_grad()
    self.forward_backward_batch(data, loss_function, forward_only=False)
    self.optimizer_step()
```

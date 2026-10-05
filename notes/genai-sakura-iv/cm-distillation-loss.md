<div class="lake-derivation" style="--derivation-border:#a597c6;--derivation-background:#f8f5fc">
<p class="lake-derivation-title">一致性蒸馏通过约束相邻轨迹点的预测一致来训练模型</p>
<div class="lake-derivation-body">

```math
\mathcal L_{\mathrm{CD}}^N(\theta,\theta^-;\phi)
=\mathbb E\!\left[\lambda(t_n)d\!\left(f_\theta(x_{t_{n+1}},t_{n+1}),f_{\theta^-}(\hat x^\phi_{t_n},t_n)\right)\right]\tag{12}
```

- 期望对 $`x\sim p_{\mathrm{data}}`$、$`n\sim\operatorname{Uniform}\{1,\ldots,N-1\}`$ 和 $`x_{t_{n+1}}\sim\mathcal N(x,t_{n+1}^2I)`$ 计算，$`\hat x^\phi_{t_n}`$ 由式（10）得到
- $`\lambda(t_n)>0`$ 为权重，$`\theta^-`$ 为目标网络参数；距离函数满足 $`d(x,y)\geq0`$，且仅当 $`x=y`$ 时取 0

</div>
</div>

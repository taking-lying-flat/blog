<div class="lake-derivation" style="--derivation-border:#98b9ac;--derivation-background:#f3f9f6">
<p class="lake-derivation-title">定理 2：一致性训练与蒸馏的渐近关系</p>
<div class="lake-derivation-body">

沿用 $`\Delta t=\max_{1\leq n<N}|t_{n+1}-t_n|`$，在下列条件下，一致性训练损失与蒸馏损失具有渐近关系

- $`d`$ 与 $`f_{\theta^-}`$ 二阶连续可微且二阶导数有界，$`\lambda`$ 有界，并满足 $`\mathbb E[\|\nabla\log p_{t_n}(x_{t_n})\|_2^2]<\infty`$
- 采用 Euler 求解器，且对所有 $`t\in[\epsilon,T]`$，分数模型精确满足 $`s_\phi(x,t)=\nabla\log p_t(x)`$

```math
\mathcal L_{\mathrm{CD}}^N(\theta,\theta^-;\phi)
=\mathcal L_{\mathrm{CT}}^N(\theta,\theta^-)+o(\Delta t)\tag{16}
```

其中，一致性训练目标定义为

```math
\mathcal L_{\mathrm{CT}}^N(\theta,\theta^-)
=\mathbb E\!\left[\lambda(t_n)d\!\left(f_\theta(x+t_{n+1}z,t_{n+1}),f_{\theta^-}(x+t_nz,t_n)\right)\right]\tag{17}
```

- 期望对 $`x\sim p_{\mathrm{data}}`$、$`n\sim\operatorname{Uniform}\{1,\ldots,N-1\}`$ 和 $`z\sim\mathcal N(0,I)`$ 计算，两侧加噪输入使用同一个噪声样本 $`z`$

</div>
</div>

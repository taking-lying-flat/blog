设 $`x_1\sim q(x_1)`$，其中 $`q`$ 是未知的数据分布，只能从中获得样本。取连接简单先验与数据分布的概率路径 $`p_t`$，使 $`p_0=p`$（例如 $`p(x)=\mathcal N(x\mid 0,I)`$），且 $`p_1\approx q`$。给定这条路径及生成它的目标向量场 $`u_t(x)`$，**Flow Matching（FM）通过回归目标向量场训练 CNF**，其目标为

```math
\mathcal{L}_{\mathrm{FM}}(\theta)=\mathbb{E}_{t,\,p_t(x)}\|v_t(x)-u_t(x)\|^2
```

- 其中，$`\theta`$ 是网络 $`v_t`$ 的可学习参数，$`t\sim\mathcal U[0,1]`$，$`x\sim p_t(x)`$。当损失为零时，$`v_t`$ 生成目标概率路径 $`p_t`$。但满足 $`p_1\approx q`$ 的路径并不唯一，生成指定路径的 $`u_t`$ 通常也没有已知的闭式表达，因此还需构造可计算的训练目标

**从单个数据样本对应的条件路径出发，可以构造边缘概率路径及其向量场。** 给定 $`x_1`$，选择条件概率路径 $`p_t(x\mid x_1)`$，使 $`p_0(x\mid x_1)=p(x)`$，并使终点分布集中在 $`x_1`$ 附近，例如 $`p_1(x\mid x_1)=\mathcal N(x\mid x_1,\sigma^2I)`$，其中 $`\sigma>0`$ 足够小。对 $`x_1\sim q(x_1)`$ 边缘化，得到

```math
p_t(x)=\int p_t(x\mid x_1)q(x_1)\,dx_1
```

在 $`t=1`$ 时，这一混合分布近似数据分布，即

```math
p_1(x)=\int p_1(x\mid x_1)q(x_1)\,dx_1 \approx q(x).
```

令 $`u_t(\cdot\mid x_1):\mathbb R^d\to\mathbb R^d`$ 为生成 $`p_t(\cdot\mid x_1)`$ 的条件向量场。假设对所有 $`t`$ 和 $`x`$ 均有 $`p_t(x)>0`$，则可按如下权重聚合条件向量场

```math
u_t(x)=\int u_t(x\mid x_1)\frac{p_t(x\mid x_1)q(x_1)}{p_t(x)}\,dx_1
```

**Theorem 1.** 若 $`u_t(x\mid x_1)`$ 生成条件概率路径 $`p_t(x\mid x_1)`$，则对任意数据分布 $`q(x_1)`$，式（7）的边缘向量场 $`u_t`$ 生成式（5）的边缘概率路径 $`p_t`$，即二者满足连续性方程 $`\frac{d}{dt}p_t(x)+\operatorname{div}\!\left(p_t(x)u_t(x)\right)=0`$

**边缘路径和边缘向量场仍包含难以计算的积分，因此改用 Conditional Flow Matching（CFM）目标，直接回归条件向量场**

```math
\mathcal{L}_{\mathrm{CFM}}(\theta) = \mathbb{E}_{t,\;q(x_1),\;p_t(x\mid x_1)} \left\|v_t(x)-u_t(x\mid x_1)\right\|^2
```

- 其中，$`t\sim\mathcal U[0,1]`$，$`x_1\sim q(x_1)`$，$`x\sim p_t(x\mid x_1)`$。只需从条件路径采样并计算 $`u_t(x\mid x_1)`$，即可构造该目标的无偏估计。**FM 与 CFM 关于 $`\theta`$ 的梯度相同**，因此可通过条件路径和条件向量场训练 CNF，无需显式计算边缘密度 $`p_t`$ 或边缘向量场 $`u_t`$

**Theorem 2.** 假设对所有 $`x\in\mathbb R^d`$ 和 $`t\in[0,1]`$ 均有 $`p_t(x)>0`$，则 $`\mathcal L_{\mathrm{CFM}}`$ 与 $`\mathcal L_{\mathrm{FM}}`$ 仅相差一个与 $`\theta`$ 无关的常数，因而 $`\nabla_\theta\mathcal L_{\mathrm{FM}}(\theta)=\nabla_\theta\mathcal L_{\mathrm{CFM}}(\theta)`$

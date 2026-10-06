流匹配模型中的迭代去噪过程可以形式化为马尔可夫决策过程 $`(\mathcal S,\mathcal A,\rho_0,P,R)`$。此处采用离散噪声层级 $`t=T,T-1,\ldots,1`$，其中 $`T`$ 为去噪步数，与流方程中的连续时间区分。状态由文本条件、噪声层级与当前样本组成，动作是模型生成的下一层级去噪样本，相应定义为

```math
\begin{aligned}
s_t&\triangleq(c,t,x_t),\qquad a_t\triangleq x_{t-1},\\[4pt]
\pi(a_t\mid s_t)&\triangleq p_\theta(x_{t-1}\mid x_t,c),\\[4pt]
P(s_{t-1}\mid s_t,a_t)&\triangleq\left(\delta_c,\delta_{t-1},\delta_{x_{t-1}}\right),\\[4pt]
\rho_0(s_T)&\triangleq\left(p(c),\delta_T,\mathcal N(0,I)\right).
\end{aligned}\tag{3}
```

- $`\pi`$ 表示去噪策略，$`P`$ 表示确定性状态转移，$`\rho_0`$ 是初始状态分布；$`\delta_y`$ 表示以 $`y`$ 为中心的狄拉克分布
- 奖励仅在最后一步给出：当 $`t=1`$、动作生成最终样本时，$`R(s_t,a_t)\triangleq r(x_0,c)`$；其余步骤的奖励均为 0

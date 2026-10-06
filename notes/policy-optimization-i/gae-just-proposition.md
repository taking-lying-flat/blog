<div class="policy-theorem" id="gae-just-proposition">

**命题 1（γ-just 的充分条件）。** 假设估计器可以写为

```math
\hat A_t(s_{0:\infty},a_{0:\infty})
=Q_t(s_{t:\infty},a_{t:\infty})-b_t(s_{0:t},a_{0:t-1}),
```

且对所有 $`(s_t,a_t)`$ 都满足

```math
\mathbb E_{s_{t+1:\infty},\,a_{t+1:\infty}\mid s_t,a_t}
\left[Q_t(s_{t:\infty},a_{t:\infty})\right]=Q^{\pi,\gamma}(s_t,a_t).
```

则 $`\hat A_t`$ 是 $`\gamma`$-just 估计器

</div>

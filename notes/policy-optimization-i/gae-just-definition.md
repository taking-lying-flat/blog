<div class="policy-theorem" id="gae-just-definition">

**定义（γ-just 估计器）。** 若估计器 $`\hat A_t`$ 满足下式，则称其为 $`\gamma`$-just

```math
\mathbb E_{\substack{s_{0:\infty}\\a_{0:\infty}}}
\left[\hat A_t(s_{0:\infty},a_{0:\infty})\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right]
=\mathbb E_{\substack{s_{0:\infty}\\a_{0:\infty}}}
\left[A^{\pi,\gamma}(s_t,a_t)\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right].
```

</div>

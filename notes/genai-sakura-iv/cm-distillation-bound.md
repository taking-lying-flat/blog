<div class="lake-derivation" style="--derivation-border:#94adc8;--derivation-background:#f3f7fc">
<p class="lake-derivation-title">定理 1：一致性蒸馏的误差界</p>
<div class="lake-derivation-body">

记 $`\Delta t=\max_{1\leq n<N}|t_{n+1}-t_n|`$，$`f(\cdot,\cdot;\phi)`$ 为式（8）所定义经验概率流 ODE 的一致性函数

- 假设 $`f_\theta`$ 对输入满足一致的 Lipschitz 条件：存在 $`L>0`$，使任意 $`t\in[\epsilon,T]`$、$`x`$ 和 $`y`$ 均满足 $`\|f_\theta(x,t)-f_\theta(y,t)\|_2\leq L\|x-y\|_2`$
- 假设 ODE 求解器在每个区间上的局部误差一致有界于 $`O((t_{n+1}-t_n)^{p+1})`$，其中 $`p\geq1`$；若 $`\mathcal L_{\mathrm{CD}}^N(\theta,\theta;\phi)=0`$，则有

```math
\sup_{n,x}\|f_\theta(x,t_n)-f(x,t_n;\phi)\|_2=O((\Delta t)^p)\tag{14}
```

</div>
</div>

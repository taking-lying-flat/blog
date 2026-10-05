- $`\lambda(t)`$ 严格单调递减，记其反函数为 $`t_\lambda(\cdot)`$。令 $`\hat{x}_\lambda:=x_{t_\lambda(\lambda)}`$，并记 $`\hat{\epsilon}_\theta(\hat{x}_\lambda,\lambda):=\epsilon_\theta(x_{t_\lambda(\lambda)},t_\lambda(\lambda))`$。将积分变量由时间换为 $`\lambda`$，利用 $`\sigma_t/\alpha_t=e^{-\lambda_t}`$，即可得到扩散 ODE 的精确解

```math
x_t=\frac{\alpha_t}{\alpha_s}x_s-\alpha_t\int_{\lambda_s}^{\lambda_t}e^{-\lambda}\hat{\epsilon}_\theta(\hat{x}_\lambda,\lambda)\,\mathrm d\lambda
```

- 线性部分由比例因子 $`\alpha_t/\alpha_s`$ 精确计算，非线性部分则化为噪声预测模型的**指数加权积分**。DPM-Solver 对这一积分进行数值近似，进而构造不同阶数的求解器

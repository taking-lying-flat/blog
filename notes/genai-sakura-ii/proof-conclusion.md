```math
\begin{aligned}
J_\sigma(\epsilon_\theta)
&=\sum_{t=1}^{T}\frac{\kappa_t^2(1-\alpha_t)}{2\sigma_t^2\alpha_t}\,\mathbb E_{x_0,\epsilon_t}\!\left[\left\|\epsilon_\theta^{(t)}(x_t)-\epsilon_t\right\|_2^2\right]+C
=L_\gamma(\epsilon_\theta)+C\\[6pt]
&\text{where}\quad\gamma_t=\frac{\kappa_t^2(1-\alpha_t)}{2\sigma_t^2\alpha_t},\quad\kappa_1:=1,\quad t=1,\ldots,T
\end{aligned}
```

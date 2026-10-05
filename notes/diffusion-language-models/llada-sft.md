- $`L'`$ 表示响应长度，算法中的损失按 $`L'`$ 归一化，提示始终保持未掩蔽

<figure class="llada-algorithm" aria-labelledby="llada-algorithm-2">
<figcaption id="llada-algorithm-2"><strong>Algorithm 2</strong> Supervised Fine-Tuning of LLaDA</figcaption>

```math
\begin{array}{r l}
&\textbf{Require:}\ \text{mask predictor }p_\theta,\ \text{pair data distribution }p_{\mathrm{data}}\\
1:&\textbf{repeat}\\
2:&\quad(p_0,r_0)\sim p_{\mathrm{data}}\\
3:&\quad t\sim U(0,1]\\
4:&\quad r_t\sim q_{t\mid 0}(r_t\mid r_0)\\
5:&\quad\displaystyle\mathcal L\gets-\frac{1}{tL'}\sum_{i=1}^{L'}\mathbf 1[r_t^i=\mathrm M]\log p_\theta(r_0^i\mid p_0,r_t)\\
6:&\quad\text{Calculate }\nabla_\theta\mathcal L\text{ and run optimizer}\\
7:&\textbf{until }\text{converged}\\
8:&\textbf{return }p_\theta
\end{array}
```

</figure>

- $`q_{t\mid 0}`$ 表示上述独立掩码过程，算法中的损失按输入序列长度 $`L`$ 归一化

<figure class="llada-algorithm" aria-labelledby="llada-algorithm-1">
<figcaption id="llada-algorithm-1"><strong>Algorithm 1</strong> Pre-training of LLaDA</figcaption>

```math
\begin{array}{r l}
&\textbf{Require:}\ \text{mask predictor }p_\theta,\ \text{data distribution }p_{\mathrm{data}}\\
1:&\textbf{repeat}\\
2:&\quad x_0\sim p_{\mathrm{data}}\\
3:&\quad t\sim U(0,1]\\
4:&\quad x_t\sim q_{t\mid 0}(x_t\mid x_0)\\
5:&\quad \displaystyle\mathcal L\gets-\frac{1}{tL}\sum_{i=1}^{L}\mathbf 1[x_t^i=\mathrm M]\log p_\theta(x_0^i\mid x_t)\\
6:&\quad\text{Calculate }\nabla_\theta\mathcal L\text{ and run optimizer}\\
7:&\textbf{until }\text{converged}\\
8:&\textbf{return }p_\theta
\end{array}
```

</figure>

- 固定提示与响应，算法通过 $`n_{mc}`$ 次蒙特卡洛估计，得到与式（6）对应的对数似然下界评分

<figure class="llada-algorithm" aria-labelledby="llada-algorithm-3">
<figcaption id="llada-algorithm-3"><strong>Algorithm 3</strong> Conditional Log-likelihood Evaluation of LLaDA</figcaption>

```math
\begin{array}{r l}
&\textbf{Require:}\ p_\theta,\ \text{prompt }p_0,\ \text{response }r_0,\ n_{mc}\\[-0.25em]
1:&\mathrm{log\_likelihood}\gets 0\\[-0.25em]
2:&\textbf{for }i\gets 1\textbf{ to }n_{mc}\textbf{ do}\\[-0.25em]
3:&\quad l\sim\operatorname{Uniform}\{1,2,\ldots,L\}\\[-0.25em]
4:&\quad\text{Obtain }r_l\text{ by uniformly masking }l\text{ positions of }r_0\text{ without replacement}\\[-0.25em]
5:&\quad\displaystyle\mathrm{log\_likelihood}\gets\mathrm{log\_likelihood}+\frac{L}{l}\sum_{i=1}^{L}\mathbf 1[r_l^i=\mathrm M]\log p_\theta(r_0^i\mid p_0,r_l)\\[-0.25em]
6:&\textbf{end for}\\[-0.25em]
7:&\mathrm{log\_likelihood}\gets\mathrm{log\_likelihood}/n_{mc}\\[-0.25em]
8:&\textbf{return }\mathrm{log\_likelihood}
\end{array}
```

</figure>

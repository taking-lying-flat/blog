<figure class="generative-algorithm" aria-labelledby="ncsn-algorithm-1">
<figcaption id="ncsn-algorithm-1"><strong>Algorithm 1</strong> Annealed Langevin Dynamics</figcaption>

```math
\begin{array}{r l}
&\textbf{Require:}\ \{\sigma_i\}_{i=1}^{L},\epsilon,T\\[-0.2em]
1:&\text{Initialize }\tilde{\mathbf x}_0\\[-0.2em]
2:&\textbf{for }i\gets 1\textbf{ to }L\textbf{ do}\\[-0.2em]
3:&\quad\alpha_i\gets\epsilon\cdot\sigma_i^2/\sigma_L^2\\[-0.2em]
4:&\quad\textbf{for }t\gets 1\textbf{ to }T\textbf{ do}\\[-0.2em]
5:&\qquad\text{Draw }\mathbf z_t\sim\mathcal N(0,I)\\[-0.2em]
6:&\qquad\displaystyle\tilde{\mathbf x}_t\gets\tilde{\mathbf x}_{t-1}+\frac{\alpha_i}{2}s_\theta(\tilde{\mathbf x}_{t-1},\sigma_i)+\sqrt{\alpha_i}\,\mathbf z_t\\[-0.2em]
7:&\quad\textbf{end for}\\[-0.2em]
8:&\quad\tilde{\mathbf x}_0\gets\tilde{\mathbf x}_T\\[-0.2em]
9:&\textbf{end for}\\[-0.2em]
10:&\textbf{return }\tilde{\mathbf x}_T
\end{array}
```

</figure>

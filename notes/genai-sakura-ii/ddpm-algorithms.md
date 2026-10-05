<div class="ddpm-algorithms">
<figure class="ddpm-algorithm" aria-labelledby="ddpm-algorithm-1">
<figcaption id="ddpm-algorithm-1"><strong>Algorithm 1</strong> Training</figcaption>

```math
\begin{array}{r l}
1:&\textbf{repeat}\\[-0.2em]
2:&\quad\mathbf{x}_0\sim q(\mathbf{x}_0)\\[-0.2em]
3:&\quad t\sim\operatorname{Uniform}(\{1,\ldots,T\})\\[-0.2em]
4:&\quad\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})\\[-0.2em]
5:&\quad\text{Take gradient descent step on}\\[-0.2em]
&\qquad\displaystyle\nabla_\theta\left\|\boldsymbol{\epsilon}-\boldsymbol{\epsilon}_\theta\!\left(\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon},t\right)\right\|^2\\[-0.2em]
6:&\textbf{until }\text{converged}
\end{array}
```

</figure>
<figure class="ddpm-algorithm" aria-labelledby="ddpm-algorithm-2">
<figcaption id="ddpm-algorithm-2"><strong>Algorithm 2</strong> Sampling</figcaption>

```math
\begin{array}{r l}
1:&\mathbf{x}_T\sim\mathcal{N}(\mathbf{0},\mathbf{I})\\[-0.2em]
2:&\textbf{for }t=T,\ldots,1\textbf{ do}\\[-0.2em]
3:&\quad\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I})\text{ if }t>1\text{, else }\mathbf{z}=\mathbf{0}\\[-0.2em]
4:&\quad\displaystyle\mathbf{x}_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)\right)+\sigma_t\mathbf{z}\\[-0.2em]
5:&\textbf{end for}\\[-0.2em]
6:&\textbf{return }\mathbf{x}_0
\end{array}
```

</figure>
</div>

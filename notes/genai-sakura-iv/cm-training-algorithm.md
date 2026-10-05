<figure class="cm-algorithm" aria-labelledby="cm-algorithm-3">
<figcaption id="cm-algorithm-3"><strong>Algorithm 3</strong> Consistency Training (CT)</figcaption>

**Input:** dataset $`\mathcal D`$, initial model parameter $`\theta`$, learning rate $`\eta`$, step schedule $`N(\cdot)`$, EMA decay rate schedule $`\mu(\cdot)`$, $`d(\cdot,\cdot)`$, and $`\lambda(\cdot)`$

```math
\begin{array}{l}
\theta^-\leftarrow\theta\text{ and }k\leftarrow0\\[-0.2em]
\textbf{repeat}\\[-0.2em]
\quad\text{Sample }\mathbf x\sim\mathcal D,\text{ and }n\sim\mathcal U[\![1,N(k)-1]\!]\\[-0.2em]
\quad\text{Sample }\mathbf z\sim\mathcal N(\mathbf0,\mathbf I)\\[-0.2em]
\quad\mathcal L(\theta,\theta^-)\leftarrow\lambda(t_n)\,d\!\left(f_\theta(\mathbf x+t_{n+1}\mathbf z,t_{n+1}),f_{\theta^-}(\mathbf x+t_n\mathbf z,t_n)\right)\\[-0.2em]
\quad\theta\leftarrow\theta-\eta\nabla_\theta\mathcal L(\theta,\theta^-)\\[-0.2em]
\quad\theta^-\leftarrow\operatorname{stopgrad}\!\left(\mu(k)\theta^-+(1-\mu(k))\theta\right)\\[-0.2em]
\quad k\leftarrow k+1\\[-0.2em]
\textbf{until }\text{convergence}
\end{array}
```

</figure>

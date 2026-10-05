<figure class="cm-algorithm" aria-labelledby="cm-algorithm-2">
<figcaption id="cm-algorithm-2"><strong>Algorithm 2</strong> Consistency Distillation (CD)</figcaption>

**Input:** dataset $`\mathcal D`$, initial model parameter $`\theta`$, learning rate $`\eta`$, ODE solver $`\Phi(\cdot,\cdot;\phi)`$, $`d(\cdot,\cdot)`$, $`\lambda(\cdot)`$, and $`\mu`$

```math
\begin{array}{l}
\theta^-\leftarrow\theta\\[-0.2em]
\textbf{repeat}\\[-0.2em]
\quad\text{Sample }\mathbf x\sim\mathcal D\text{ and }n\sim\mathcal U[\![1,N-1]\!]\\[-0.2em]
\quad\text{Sample }\mathbf x_{t_{n+1}}\sim\mathcal N(\mathbf x;t_{n+1}^2\mathbf I)\\[-0.2em]
\quad\hat{\mathbf x}_{t_n}^\phi\leftarrow\mathbf x_{t_{n+1}}+(t_n-t_{n+1})\Phi(\mathbf x_{t_{n+1}},t_{n+1};\phi)\\[-0.2em]
\quad\mathcal L(\theta,\theta^-;\phi)\leftarrow\lambda(t_n)\,d\!\left(f_\theta(\mathbf x_{t_{n+1}},t_{n+1}),f_{\theta^-}(\hat{\mathbf x}_{t_n}^\phi,t_n)\right)\\[-0.2em]
\quad\theta\leftarrow\theta-\eta\nabla_\theta\mathcal L(\theta,\theta^-;\phi)\\[-0.2em]
\quad\theta^-\leftarrow\operatorname{stopgrad}\!\left(\mu\theta^-+(1-\mu)\theta\right)\\[-0.2em]
\textbf{until }\text{convergence}
\end{array}
```

</figure>

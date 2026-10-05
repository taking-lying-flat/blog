<figure class="cm-algorithm" aria-labelledby="cm-algorithm-3">
<figcaption id="cm-algorithm-3"><strong>Algorithm 3</strong> Consistency Training (CT)</figcaption>

```math
\begin{array}{l}
\textbf{Input: }\text{dataset }\mathcal D,\text{ initial model parameter }\theta,\text{ learning rate }\eta,\\[-0.2em]
\text{step schedule }N(\cdot),\text{ EMA decay rate schedule }\mu(\cdot),\,d(\cdot,\cdot),\text{ and }\lambda(\cdot)\\[-0.2em]
\theta^-\leftarrow\theta\text{ and }k\leftarrow0\\[-0.2em]
\textbf{repeat}\\[-0.2em]
\quad\text{Sample }\mathbf x\sim\mathcal D,\text{ and }n\sim\mathcal U[\![1,N(k)-1]\!]\\[-0.2em]
\quad\text{Sample }\mathbf z\sim\mathcal N(\mathbf0,\mathbf I)\\[-0.2em]
\quad\mathcal L(\theta,\theta^-)\leftarrow\\[-0.2em]
\qquad\lambda(t_n)\,d\!\left(f_\theta(\mathbf x+t_{n+1}\mathbf z,t_{n+1}),f_{\theta^-}(\mathbf x+t_n\mathbf z,t_n)\right)\\[-0.2em]
\quad\theta\leftarrow\theta-\eta\nabla_\theta\mathcal L(\theta,\theta^-)\\[-0.2em]
\quad\theta^-\leftarrow\operatorname{stopgrad}\!\left(\mu(k)\theta^-+(1-\mu(k))\theta\right)\\[-0.2em]
\quad k\leftarrow k+1\\[-0.2em]
\textbf{until }\text{convergence}
\end{array}
```

</figure>

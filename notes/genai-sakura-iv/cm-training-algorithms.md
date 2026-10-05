<div class="cm-training-algorithms">
<figure class="cm-algorithm" aria-labelledby="cm-algorithm-2">
<figcaption id="cm-algorithm-2"><strong>Algorithm 2</strong> Consistency Distillation (CD)</figcaption>

```math
\begin{array}{l}
\textbf{Input: }\text{dataset }\mathcal D,\text{ initial model parameter }\theta,\\[-0.2em]
\text{learning rate }\eta,\text{ ODE solver }\Phi(\cdot,\cdot;\phi),\,d(\cdot,\cdot),\,\lambda(\cdot),\text{ and }\mu\\[-0.2em]
\theta^-\leftarrow\theta\\[-0.2em]
\textbf{repeat}\\[-0.2em]
\quad\text{Sample }\mathbf x\sim\mathcal D\text{ and }n\sim\mathcal U[\![1,N-1]\!]\\[-0.2em]
\quad\text{Sample }\mathbf x_{t_{n+1}}\sim\mathcal N(\mathbf x;t_{n+1}^2\mathbf I)\\[-0.2em]
\quad\hat{\mathbf x}_{t_n}^\phi\leftarrow\mathbf x_{t_{n+1}}+(t_n-t_{n+1})\Phi(\mathbf x_{t_{n+1}},t_{n+1};\phi)\\[-0.2em]
\quad\mathcal L(\theta,\theta^-;\phi)\leftarrow\\[-0.2em]
\qquad\lambda(t_n)\,d\!\left(f_\theta(\mathbf x_{t_{n+1}},t_{n+1}),f_{\theta^-}(\hat{\mathbf x}_{t_n}^\phi,t_n)\right)\\[-0.2em]
\quad\theta\leftarrow\theta-\eta\nabla_\theta\mathcal L(\theta,\theta^-;\phi)\\[-0.2em]
\quad\theta^-\leftarrow\operatorname{stopgrad}\!\left(\mu\theta^-+(1-\mu)\theta\right)\\[-0.2em]
\textbf{until }\text{convergence}
\end{array}
```

</figure>
<figure class="cm-algorithm" aria-labelledby="cm-algorithm-3">
<figcaption id="cm-algorithm-3"><strong>Algorithm 3</strong> Consistency Training (CT)</figcaption>

```math
\begin{array}{l}
\textbf{Input: }\text{dataset }\mathcal D,\text{ initial model parameter }\theta,\\[-0.2em]
\text{learning rate }\eta,\text{ step schedule }N(\cdot),\text{ EMA decay rate}\\[-0.2em]
\text{schedule }\mu(\cdot),\,d(\cdot,\cdot),\text{ and }\lambda(\cdot)\\[-0.2em]
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
</div>

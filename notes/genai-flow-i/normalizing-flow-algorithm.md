<figure class="flow-algorithm" aria-labelledby="normalizing-flow-algorithm-1">
<figcaption id="normalizing-flow-algorithm-1"><strong>Algorithm 1</strong> Variational Inference With Normalizing Flows</figcaption>

```math
\begin{array}{l}
\text{Parameters: }\phi\text{ variational, }\theta\text{ generative}\\[-0.2em]
\textbf{while }\text{not converged}\textbf{ do}\\[-0.2em]
\quad\mathbf{x}\leftarrow\{\text{Get mini-batch}\}\\[-0.2em]
\quad\mathbf{z}_0\sim q_0(\bullet\mid\mathbf{x})\\[-0.2em]
\quad\mathbf{z}_K\leftarrow f_K\circ f_{K-1}\circ\cdots\circ f_1(\mathbf{z}_0)\\[-0.2em]
\quad\mathcal{F}(\mathbf{x})\approx\mathcal{F}(\mathbf{x},\mathbf{z}_K)\\[-0.2em]
\quad\Delta\theta\propto-\nabla_\theta\mathcal{F}(\mathbf{x})\\[-0.2em]
\quad\Delta\phi\propto-\nabla_\phi\mathcal{F}(\mathbf{x})\\[-0.2em]
\textbf{end while}
\end{array}
```

</figure>

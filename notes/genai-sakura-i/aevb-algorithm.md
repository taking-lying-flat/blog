<figure class="generative-algorithm" aria-labelledby="aevb-algorithm-1">
<figcaption id="aevb-algorithm-1"><strong>Algorithm 1</strong> Minibatch Auto-Encoding Variational Bayes</figcaption>

```math
\begin{array}{r l}
1:&\theta,\phi\gets\text{Initialize parameters}\\[-0.2em]
2:&\textbf{repeat}\\[-0.2em]
3:&\quad\mathbf X^M\gets\text{Random minibatch of }M\text{ datapoints from the full dataset}\\[-0.2em]
4:&\quad\boldsymbol\epsilon\gets\text{Random samples from noise distribution }p(\boldsymbol\epsilon)\\[-0.2em]
5:&\quad\mathbf g\gets\nabla_{\theta,\phi}\widetilde{\mathcal L}^{M}(\theta,\phi;\mathbf X^M,\boldsymbol\epsilon)\\[-0.2em]
6:&\quad\theta,\phi\gets\text{Update parameters using gradients }\mathbf g\text{ (e.g. SGD or Adagrad)}\\[-0.2em]
7:&\textbf{until }\text{convergence of parameters }(\theta,\phi)\\[-0.2em]
8:&\textbf{return }\theta,\phi
\end{array}
```

</figure>

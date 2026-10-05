<figure class="edm-algorithm" aria-labelledby="edm-algorithm-1">
<figcaption id="edm-algorithm-1"><strong>Algorithm 1</strong> Deterministic Sampling Using Heun’s Second-Order Method</figcaption>

```math
\begin{array}{r l}
1:&\textbf{procedure }\operatorname{HeunSampler}\!\left(D_\theta(x;\sigma),\,\sigma(t),\,s(t),\,t_{i\in\{0,\ldots,N\}}\right)\\[-0.2em]
2:&\quad\textbf{sample }\mathbf{x}_0\sim\mathcal N\!\left(\mathbf{0},\,\sigma^2(t_0)s^2(t_0)\mathbf I\right)\\[-0.2em]
3:&\quad\textbf{for }i\in\{0,\ldots,N-1\}\textbf{ do}\\[-0.2em]
4:&\qquad\displaystyle\mathbf d_i\leftarrow\left(\frac{\dot\sigma(t_i)}{\sigma(t_i)}+\frac{\dot s(t_i)}{s(t_i)}\right)\mathbf x_i-\frac{\dot\sigma(t_i)s(t_i)}{\sigma(t_i)}D_\theta\!\left(\frac{\mathbf x_i}{s(t_i)};\sigma(t_i)\right)\\[-0.2em]
5:&\qquad\mathbf x_{i+1}\leftarrow\mathbf x_i+(t_{i+1}-t_i)\mathbf d_i\\[-0.2em]
6:&\qquad\textbf{if }\sigma(t_{i+1})\ne0\textbf{ then}\\[-0.2em]
7:&\qquad\quad\displaystyle\mathbf d'_i\leftarrow\left(\frac{\dot\sigma(t_{i+1})}{\sigma(t_{i+1})}+\frac{\dot s(t_{i+1})}{s(t_{i+1})}\right)\mathbf x_{i+1}-\frac{\dot\sigma(t_{i+1})s(t_{i+1})}{\sigma(t_{i+1})}D_\theta\!\left(\frac{\mathbf x_{i+1}}{s(t_{i+1})};\sigma(t_{i+1})\right)\\[-0.2em]
8:&\qquad\quad\mathbf x_{i+1}\leftarrow\mathbf x_i+(t_{i+1}-t_i)\left(\tfrac12\mathbf d_i+\tfrac12\mathbf d'_i\right)\\[-0.2em]
9:&\quad\textbf{return }\mathbf x_N
\end{array}
```

</figure>

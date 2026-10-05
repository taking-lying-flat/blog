<figure class="edm-algorithm" aria-labelledby="edm-algorithm-2">
<figcaption id="edm-algorithm-2"><strong>Algorithm 2</strong> Stochastic Sampling With <i>σ(t)</i> = <i>t</i> And <i>s(t)</i> = 1</figcaption>

```math
\begin{array}{r l}
1:&\textbf{procedure }\operatorname{StochasticSampler}\!\left(D_\theta(x;\sigma),\,t_{i\in\{0,\ldots,N\}},\,\gamma_{i\in\{0,\ldots,N-1\}},\,S_{\mathrm{noise}}\right)\\[-0.2em]
2:&\quad\textbf{sample }\mathbf x_0\sim\mathcal N(\mathbf0,t_0^2\mathbf I)\\[-0.2em]
3:&\quad\textbf{for }i\in\{0,\ldots,N-1\}\textbf{ do}\\[-0.2em]
&\qquad\gamma_i=\begin{cases}\min\!\left(S_{\mathrm{churn}}/N,\sqrt2-1\right)&\text{if }t_i\in[S_{\mathrm{tmin}},S_{\mathrm{tmax}}]\\0&\text{otherwise}\end{cases}\\[-0.2em]
4:&\qquad\textbf{sample }\boldsymbol\epsilon_i\sim\mathcal N(\mathbf0,S_{\mathrm{noise}}^2\mathbf I)\\[-0.2em]
5:&\qquad\hat t_i\leftarrow t_i+\gamma_i t_i\\[-0.2em]
6:&\qquad\hat{\mathbf x}_i\leftarrow\mathbf x_i+\sqrt{\hat t_i^2-t_i^2}\,\boldsymbol\epsilon_i\\[-0.2em]
7:&\qquad\mathbf d_i\leftarrow\left(\hat{\mathbf x}_i-D_\theta(\hat{\mathbf x}_i;\hat t_i)\right)/\hat t_i\\[-0.2em]
8:&\qquad\mathbf x_{i+1}\leftarrow\hat{\mathbf x}_i+(t_{i+1}-\hat t_i)\mathbf d_i\\[-0.2em]
9:&\qquad\textbf{if }t_{i+1}\ne0\textbf{ then}\\[-0.2em]
10:&\qquad\quad\mathbf d'_i\leftarrow\left(\mathbf x_{i+1}-D_\theta(\mathbf x_{i+1};t_{i+1})\right)/t_{i+1}\\[-0.2em]
11:&\qquad\quad\mathbf x_{i+1}\leftarrow\hat{\mathbf x}_i+(t_{i+1}-\hat t_i)\left(\tfrac12\mathbf d_i+\tfrac12\mathbf d'_i\right)\\[-0.2em]
12:&\quad\textbf{return }\mathbf x_N
\end{array}
```

</figure>

<!-- Transcribed from the supplied Block Diffusion paper, Algorithms 1–2.
Keep the paper's notation and update order; line numbers are presentation only. -->

## 🧮 Training and Sampling Algorithms

<figure class="paper-algorithm" aria-labelledby="block-diffusion-algorithm-1">
<figcaption id="block-diffusion-algorithm-1"><strong>Algorithm 1</strong> Block Diffusion Training</figcaption>

```math
\begin{array}{r l}
&\textbf{Input:}\ \text{datapoint }\mathbf{x},\ \text{number of blocks }B,\ \text{forward noise process }q_t(\cdot\mid\mathbf{x}),\\[-0.25em]
&\qquad\text{model }\mathbf{x}_\theta,\ \text{loss }\mathcal{L}_{\mathrm{BD}}\\[-0.25em]
1:&\textbf{repeat}\\[-0.25em]
2:&\quad\text{Sample }t_1,\ldots,t_B\sim\mathcal{U}[0,1]\\[-0.25em]
3:&\quad\forall b\in\{1,\ldots,B\}:\ \mathbf{x}_{t_b}^{b}\sim q_{t_b}(\cdot\mid\mathbf{x}^{b})\\[-0.25em]
4:&\quad(\varnothing,\mathbf{K}^{1:B},\mathbf{V}^{1:B})\gets\mathbf{x}_\theta(\mathbf{x})\\[-0.25em]
5:&\quad\forall b:\ (\mathbf{x}_{\mathrm{logit}}^{b},\varnothing,\varnothing)\gets\mathbf{x}_\theta^{b}(\mathbf{x}_{t_b}^{b},\mathbf{K}^{1:b-1},\mathbf{V}^{1:b-1})\\[-0.25em]
6:&\quad\mathbf{x}_{\mathrm{logit}}\gets\mathbf{x}_{\mathrm{logit}}^{1}\oplus\cdots\oplus\mathbf{x}_{\mathrm{logit}}^{B}\\[-0.25em]
7:&\quad\text{Take gradient step on }\nabla_\theta\mathcal{L}_{\mathrm{BD}}(\mathbf{x}_{\mathrm{logit}};\theta)\\[-0.25em]
8:&\textbf{until }\text{converged}
\end{array}
```

</figure>

<figure class="paper-algorithm" aria-labelledby="block-diffusion-algorithm-2">
<figcaption id="block-diffusion-algorithm-2"><strong>Algorithm 2</strong> Block Diffusion Sampling</figcaption>

```math
\begin{array}{r l}
&\textbf{Input:}\ \text{number of blocks }B,\ \text{model }\mathbf{x}_\theta,\ \text{diffusion sampling algorithm }\operatorname{SAMPLE}\\[-0.25em]
1:&\mathbf{x},\mathbf{K},\mathbf{V}\gets\varnothing\\[-0.25em]
2:&\textbf{for }b=1\textbf{ to }B\textbf{ do}\\[-0.25em]
3:&\quad\mathbf{x}^{b}\gets\operatorname{SAMPLE}(\mathbf{x}_\theta^{b},\mathbf{K}^{1:b-1},\mathbf{V}^{1:b-1})\\[-0.25em]
4:&\quad(\varnothing,\mathbf{K}^{b},\mathbf{V}^{b})\gets\mathbf{x}_\theta^{b}(\mathbf{x}^{b})\\[-0.25em]
5:&\quad\mathbf{x}\gets\mathbf{x}^{1:b-1}\oplus\mathbf{x}^{b}\\[-0.25em]
6:&\quad(\mathbf{K},\mathbf{V})\gets(\mathbf{K}^{1:b-1}\oplus\mathbf{K}^{b},\mathbf{V}^{1:b-1}\oplus\mathbf{V}^{b})\\[-0.25em]
7:&\textbf{end for}\\[-0.25em]
8:&\textbf{return }\mathbf{x}
\end{array}
```

</figure>

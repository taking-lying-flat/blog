<!-- Source: arXiv:2302.01318v1, Algorithm 2 and Supplementary Materials / Proofs. -->
<figure class="spec-algorithm" aria-labelledby="spec-sampling-algorithm-title">
<figcaption id="spec-sampling-algorithm-title"><strong>Algorithm 2</strong> Speculative Sampling (SpS) with Auto-Regressive Target and Draft Models</figcaption>

```math
\begin{array}{l}
\text{Given lookahead }K\text{ and minimum target sequence length }T \\
\text{Given auto-regressive target model }q(\cdot\mid\cdot)\text{, and auto-regressive draft model }p(\cdot\mid\cdot)\text{,} \\
\text{initial prompt sequence }x_0,\ldots,x_t \\
\text{Initialise }n\gets t \\
\textbf{while }n<T\textbf{ do} \\
\quad\textbf{for }t=1:K\textbf{ do} \\
\qquad\text{Sample draft auto-regressively }\tilde{x}_t\sim p(x\mid x_1,\ldots,x_n,\tilde{x}_1,\ldots,\tilde{x}_{t-1}) \\
\quad\textbf{end for} \\
\quad\text{In parallel, compute }K+1\text{ sets of logits from drafts }\tilde{x}_1,\ldots,\tilde{x}_K:\\
\qquad q(x\mid x_1,\ldots,x_n),\ q(x\mid x_1,\ldots,x_n,\tilde{x}_1),\ldots,\ q(x\mid x_1,\ldots,x_n,\tilde{x}_1,\ldots,\tilde{x}_K) \\[0.2em]
\quad\textbf{for }t=1:K\textbf{ do} \\
\qquad\text{Sample }r\sim U[0,1]\text{ from a uniform distribution} \\
\qquad\textbf{if }r<\min\!\left(1,\frac{q(x\mid x_1,\ldots,x_{n+t-1})}{p(x\mid x_1,\ldots,x_{n+t-1})}\right)\textbf{ then} \\
\qquad\quad\text{Set }x_{n+t}\gets\tilde{x}_t\text{ and }n\gets n+1 \\
\qquad\textbf{else} \\
\qquad\quad\text{sample }x_{n+t}\sim\bigl(q(x\mid x_1,\ldots,x_{n+t-1})-p(x\mid x_1,\ldots,x_{n+t-1})\bigr)_+\text{ and exit for loop} \\
\qquad\textbf{end if} \\
\quad\textbf{end for} \\
\quad\text{If all tokens }x_{n+1},\ldots,x_{n+K}\text{ are accepted, sample extra token} \\
\quad x_{n+K+1}\sim q(x\mid x_1,\ldots,x_n,x_{n+K})\text{ and set }n\gets n+1 \\
\textbf{end while}
\end{array}
```

</figure>

## 🍄 Proofs

- 本篇论文中，**草稿模型（draft）的离散分布记为 $`p`$，目标模型（target）的离散分布记为 $`q`$**，与第一篇论文的符号约定相反。给定单个草稿样本 $`\tilde{x}\sim p`$，记最终得到的样本为 $`X`$。若 $`X=x`$，则或者先采样得到 $`\tilde{x}=x`$ 并接受它，或者在任意取值的 $`\tilde{x}`$ 被拒绝后，重新采样得到 $`x`$，因此

```math
\mathbb{P}(X=x)
=\mathbb{P}(\tilde{x}=x)\mathbb{P}(\tilde{x}\text{ accepted}\mid\tilde{x}=x)
+\mathbb{P}(\tilde{x}\text{ rejected})\mathbb{P}(X=x\mid\tilde{x}\text{ rejected})\tag{14}
```

- 对于第一项，代入接受规则，得到

```math
\mathbb{P}(\tilde{x}=x)\mathbb{P}(\tilde{x}\text{ accepted}\mid\tilde{x}=x)
=p(x)\min\!\left(1,\frac{q(x)}{p(x)}\right)=\min(p(x),q(x))\tag{15}
```

- 对于第二项中的条件概率，代入重新采样规则，得到

```math
\mathbb{P}(X=x\mid\tilde{x}\text{ rejected})=(q(x)-p(x))_+\tag{16}
```

- 其中，$`(\cdot)_+`$ 的归一化方式见式（13）。拒绝概率为

```math
\begin{aligned}
\mathbb{P}(\tilde{x}\text{ rejected})
&=1-\mathbb{P}(\tilde{x}\text{ accepted})\\
&=1-\sum_{x'}\mathbb{P}(X=x',\tilde{x}\text{ accepted})
=1-\sum_{x'}\min(p(x'),q(x'))\\
&=\sum_{x'}\max(0,q(x')-p(x'))
=\sum_{x'}\bigl(q(x')-\min(p(x'),q(x'))\bigr)\\
&=\sum_{x'}\max(0,q(x')-p(x'))
\end{aligned}\tag{17}
```

- 这与 $`(q(x)-p(x))_+`$ 的分母相同，因此

```math
\mathbb{P}(\tilde{x}\text{ rejected})\mathbb{P}(X=x\mid\tilde{x}\text{ rejected})
=\max(0,q(x)-p(x))\tag{18}
```

- 合并两部分，得到目标分布

```math
\mathbb{P}(X=x)=\min(p(x),q(x))+\max(0,q(x)-p(x))=q(x)\tag{19}
```

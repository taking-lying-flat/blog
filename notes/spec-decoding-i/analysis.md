<figure class="spec-algorithm" aria-labelledby="spec-algorithm-title">
<figcaption id="spec-algorithm-title"><strong>Algorithm 1</strong> SpeculativeDecodingStep</figcaption>

```math
\begin{array}{l}
\textbf{Inputs:}\ M_p,\ M_q,\ \mathrm{prefix} \\
\triangleright\ \textit{Sample }\gamma\textit{ guesses }x_1,\ldots,x_\gamma\textit{ from }M_q\textit{ autoregressively} \\
\textbf{for }i=1\textbf{ to }\gamma\textbf{ do} \\
\qquad q_i(x)\gets M_q(\mathrm{prefix}+[x_1,\ldots,x_{i-1}]) \\
\qquad x_i\sim q_i(x) \\
\textbf{end for} \\[0.2em]
\triangleright\ \textit{Run }M_p\textit{ in parallel} \\
p_1(x),\ldots,p_{\gamma+1}(x)\gets
M_p(\mathrm{prefix}),\ldots,M_p(\mathrm{prefix}+[x_1,\ldots,x_\gamma]) \\[0.2em]
\triangleright\ \textit{Determine the number of accepted guesses }n \\
r_1\sim U(0,1),\ldots,r_\gamma\sim U(0,1) \\
n\gets\min\!\left(\left\{i-1\ \middle|\ 1\le i\le\gamma,\ r_i>\frac{p_i(x_i)}{q_i(x_i)}\right\}\cup\{\gamma\}\right) \\[0.2em]
\triangleright\ \textit{Adjust the distribution from }M_p\textit{ if needed} \\
p'(x)\gets p_{n+1}(x) \\
\textbf{if }n<\gamma\textbf{ then} \\
\qquad p'(x)\gets\operatorname{norm}\!\left(\max\!\left(0,p_{n+1}(x)-q_{n+1}(x)\right)\right) \\
\textbf{end if} \\[0.2em]
\triangleright\ \textit{Return one token from }M_p\textit{, and }n\textit{ tokens from }M_q \\
t\sim p'(x) \\
\textbf{return }\mathrm{prefix}+[x_1,\ldots,x_n,t]
\end{array}
```

</figure>

## 🍄 Analysis

**期望生成量。** 给定前缀 $`x_{<t}`$，记 $`\beta_{x_{<t}}`$ 为候选 $`x_t\sim q(x_t\mid x_{<t})`$ 被接受的概率，其平均值 $`\alpha=\mathbb{E}[\beta]`$ 衡量草稿模型 $`M_q`$ 对目标模型 $`M_p`$ 的近似程度。沿用各位置接受率独立同分布的简化假设，每轮先接受一段连续候选，再生成一个修正或额外的 token，因此单轮输出数量 $`N`$ 服从以拒绝概率 $`1-\alpha`$ 为终止概率、上限为 $`\gamma+1`$ 的截顶几何分布，其期望为

```math
\mathbb{E}[N]=\sum_{j=0}^{\gamma}\alpha^j=\frac{1-\alpha^{\gamma+1}}{1-\alpha}\tag{1}
```

当 $`0\le\alpha<1`$ 时，上式的分式形式成立；当 $`\alpha=1`$ 时，全部候选均被接受，每轮固定生成 $`\gamma+1`$ 个 token

**分布差异与接受率。** 为计算 $`\alpha`$，先考察固定前缀下的两个条件分布 $`p(x)`$ 与 $`q(x)`$，并以二者的中点分布 $`M(x)`$ 定义距离

```math
M(x)=\frac{p(x)+q(x)}{2},\qquad
D_{\mathrm{LK}}(p,q)=\sum_x\lvert p(x)-M(x)\rvert=\sum_x\lvert q(x)-M(x)\rvert\tag{2}
```

由 $`\min(p(x),q(x))=(p(x)+q(x)-\lvert p(x)-q(x)\rvert)/2`$，结合概率分布的归一化条件，可将该距离改写为

```math
D_{\mathrm{LK}}(p,q)=\frac{1}{2}\sum_x\lvert p(x)-q(x)\rvert
=1-\sum_x\min(p(x),q(x))\tag{3}
```

因此，$`D_{\mathrm{LK}}`$ 是取值范围为 $`[0,1]`$ 的对称距离：$`D_{\mathrm{LK}}=0`$ 当且仅当 $`p=q`$；$`D_{\mathrm{LK}}=1`$ 当且仅当二者的支撑集互不相交

对于从 $`q`$ 中采样的候选 $`x`$，投机采样以 $`\min(1,p(x)/q(x))`$ 的概率接受它。对所有候选求期望，即得到当前前缀下的接受率

```math
\beta=\mathbb{E}_{x\sim q}\!\left[\min\!\left(1,\frac{p(x)}{q(x)}\right)\right]
=\sum_x\min(p(x),q(x))=1-D_{\mathrm{LK}}(p,q)\tag{4}
```

再对生成过程中遇到的前缀取期望，得到平均接受率

```math
\alpha=\mathbb{E}[\beta]
=1-\mathbb{E}[D_{\mathrm{LK}}(p,q)]
=\mathbb{E}\!\left[\sum_x\min(p(x),q(x))\right]\tag{5}
```

两个分布的重叠程度越高，平均接受率越大；将其代入式（1），即可得到每轮生成 token 数量的期望

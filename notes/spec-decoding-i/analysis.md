[作者伪代码 · Algorithm 1](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf#page=3)

```text SpeculativeDecodingStep
Inputs: M_p, M_q, prefix
▷ Sample γ guesses x_1, ..., x_γ from M_q autoregressively.
for i = 1 to γ do
    q_i(x) ← M_q(prefix + [x_1, ..., x_{i−1}])
    x_i ∼ q_i(x)
end for
▷ Run M_p in parallel.
p_1(x), ..., p_{γ+1}(x) ←
    M_p(prefix), ..., M_p(prefix + [x_1, ..., x_γ])
▷ Determine the number of accepted guesses n.
r_1 ∼ U(0, 1), ..., r_γ ∼ U(0, 1)
n ← min({i − 1 | 1 ≤ i ≤ γ, r_i > p_i(x_i) / q_i(x_i)} ∪ {γ})
▷ Adjust the distribution from M_p if needed.
p′(x) ← p_{n+1}(x)
if n < γ then
    p′(x) ← norm(max(0, p_{n+1}(x) − q_{n+1}(x)))
end if
▷ Return one token from M_p, and n tokens from M_q.
t ∼ p′(x)
return prefix + [x_1, ..., x_n, t]
```

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

**算术运算与访存开销。** 每轮需要串行执行 $`\gamma`$ 次草稿模型计算，并并行评估目标模型在 $`\gamma+1`$ 个前缀下的分布。遇到拒绝时，被拒绝候选之后的目标模型预测会被丢弃，因此总算术运算量可能高于标准解码

记 $`\hat T`$ 为标准解码中目标模型每生成一个 token 的算术运算量，$`\hat c`$ 为草稿模型与目标模型每个 token 所需算术运算量之比，则每轮的总算术运算量为

```math
C_{\mathrm{iter}}=\hat T\hat c\gamma+\hat T(\gamma+1)
=\hat T(\gamma\hat c+\gamma+1)\tag{6}
```

按平均生成量摊销后，每生成一个 token 的算术运算量相对于标准解码的倍数为

```math
R_{\mathrm{ops}}=\frac{C_{\mathrm{iter}}}{\hat T\,\mathbb{E}[N]}
=\frac{\gamma\hat c+\gamma+1}{\mathbb{E}[N]}
=\frac{(1-\alpha)(\gamma\hat c+\gamma+1)}{1-\alpha^{\gamma+1}}\tag{7}
```

在草稿长度和模型计算量固定时，接受率越低，摊销后的运算开销越大；当 $`\alpha=1`$ 时，上述倍数为 $`1+\gamma\hat c/(\gamma+1)`$，此时额外运算仅来自草稿模型。对于 Transformer 解码器，不计草稿模型的计算，单轮目标模型的算术运算量可由相同规模 Transformer 编码器处理同一序列的一次前向计算给出上界

目标模型的权重和已有 KV 缓存可在一轮并行验证中复用，因而每轮只需读取一次。按生成 token 数量摊销，读取这些数据所需的内存访问量可降至标准解码的 $`1/\mathbb{E}[N]`$

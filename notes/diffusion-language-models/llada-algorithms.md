<!-- Transcribed from the user-supplied NeurIPS 2025 paper, pp. 25–27, Algorithms 1–5.
Algorithm 5 corrects the paper's Lowest-n_un typo and selects positions only from
the currently masked set, consistent with ML-GSAI/LLaDA generate.py.
The paper's floor schedule is retained; this is not a literal code transcription. -->

## 🧮 Training and Evaluation Algorithms

**预训练与监督微调均通过随机掩码构造预测任务，区别在于监督微调始终保留提示，仅对响应施加掩码**

- $`q_{t\mid 0}`$ 表示前文的独立掩码过程，每个位置以概率 $`t`$ 变为 $`\mathrm M`$，以概率 $`1-t`$ 保留原 token
- 算法 1 与算法 2 分别按输入长度 $`L`$ 和响应长度 $`L'`$ 对损失归一化；预训练中有 1% 的样本使用从 $`1,\ldots,4096`$ 均匀采样的序列长度

<figure class="llada-algorithm" aria-labelledby="llada-algorithm-1">
<figcaption id="llada-algorithm-1"><strong>Algorithm 1</strong> Pre-training of LLaDA</figcaption>

```math
\begin{array}{r l}
&\textbf{Require:}\ \text{mask predictor }p_\theta,\ \text{data distribution }p_{\mathrm{data}}\\
1:&\textbf{repeat}\\
2:&\quad x_0\sim p_{\mathrm{data}}\\
3:&\quad t\sim U(0,1]\\
4:&\quad x_t\sim q_{t\mid 0}(x_t\mid x_0)\\
5:&\quad \displaystyle\mathcal L\gets-\frac{1}{tL}\sum_{i=1}^{L}\mathbf 1[x_t^i=\mathrm M]\log p_\theta(x_0^i\mid x_t)\\
6:&\quad\text{Calculate }\nabla_\theta\mathcal L\text{ and run optimizer}\\
7:&\textbf{until }\text{converged}\\
8:&\textbf{return }p_\theta
\end{array}
```

</figure>

<figure class="llada-algorithm" aria-labelledby="llada-algorithm-2">
<figcaption id="llada-algorithm-2"><strong>Algorithm 2</strong> Supervised Fine-Tuning of LLaDA</figcaption>

```math
\begin{array}{r l}
&\textbf{Require:}\ \text{mask predictor }p_\theta,\ \text{pair data distribution }p_{\mathrm{data}}\\
1:&\textbf{repeat}\\
2:&\quad(p_0,r_0)\sim p_{\mathrm{data}}\\
3:&\quad t\sim U(0,1]\\
4:&\quad r_t\sim q_{t\mid 0}(r_t\mid r_0)\\
5:&\quad\displaystyle\mathcal L\gets-\frac{1}{tL'}\sum_{i=1}^{L'}\mathbf 1[r_t^i=\mathrm M]\log p_\theta(r_0^i\mid p_0,r_t)\\
6:&\quad\text{Calculate }\nabla_\theta\mathcal L\text{ and run optimizer}\\
7:&\textbf{until }\text{converged}\\
8:&\textbf{return }p_\theta
\end{array}
```

</figure>

**条件似然评估通过重复掩码估计与式（6）对应的对数似然下界，并以此为候选响应评分**

- 固定提示 $`p_0`$ 与响应 $`r_0`$，每次先均匀采样掩码数量 $`l`$，再无放回地选择掩码位置；$`L`$ 为响应长度，$`n_{mc}`$ 为蒙特卡洛估计次数

<figure class="llada-algorithm" aria-labelledby="llada-algorithm-3">
<figcaption id="llada-algorithm-3"><strong>Algorithm 3</strong> Conditional Log-likelihood Evaluation of LLaDA</figcaption>

```math
\begin{array}{r l}
&\textbf{Require:}\ p_\theta,\ \text{prompt }p_0,\ \text{response }r_0,\ n_{mc}\\
1:&\mathrm{log\_likelihood}\gets 0\\
2:&\textbf{for }i\gets 1\textbf{ to }n_{mc}\textbf{ do}\\
3:&\quad l\sim\operatorname{Uniform}\{1,2,\ldots,L\}\\
4:&\quad\text{Obtain }r_l\text{ by uniformly masking }l\text{ positions of }r_0\text{ without replacement}\\
5:&\quad\displaystyle\mathrm{log\_likelihood}\gets\mathrm{log\_likelihood}+\frac{L}{l}\sum_{i=1}^{L}\mathbf 1[r_l^i=\mathrm M]\log p_\theta(r_0^i\mid p_0,r_l)\\
6:&\textbf{end for}\\
7:&\mathrm{log\_likelihood}\gets\mathrm{log\_likelihood}/n_{mc}\\
8:&\textbf{return }\mathrm{log\_likelihood}
\end{array}
```

</figure>

## 🎲 Remasking Algorithms

**两种重掩码策略均并行预测当前掩码位置，保持已经确定的 token 不变，并逐步减少掩码数量**

- $`L`$ 为响应长度，$`N`$ 为采样步数，时间从 $`t=1`$ 逐步递减至 0。附录算法采用逐位置贪心预测，二者的区别在于重掩码位置的选择：随机策略以概率 $`s/t`$ 重新掩蔽预测结果，低置信度策略则优先保留高置信度预测

<figure class="llada-algorithm" aria-labelledby="llada-algorithm-4">
<figcaption id="llada-algorithm-4"><strong>Algorithm 4</strong> Random Remasking Strategy of LLaDA</figcaption>

```math
\begin{array}{r l}
&\textbf{Require:}\ p_\theta,\ \text{prompt }p_0,\ \text{answer length }L,\ \text{sampling steps }N\\
1:&r_1\gets[\mathrm M,\ldots,\mathrm M]\quad\text{(length }L\text{)}\\
2:&\textbf{for }t\gets 1\textbf{ down to }1/N\textbf{ step }1/N\textbf{ do}\\
3:&\quad s\gets t-1/N\\
4:&\quad r_0\gets\arg\max_{r_0}p_\theta(r_0\mid p_0,r_t)\\
5:&\quad\textbf{for }i\gets 1\textbf{ to }L\textbf{ do}\\
6:&\qquad\textbf{if }r_t^i\ne\mathrm M\textbf{ then}\\
7:&\qquad\quad r_0^i\gets r_t^i\\
8:&\qquad\textbf{else}\\
9:&\qquad\quad\text{With probability }s/t,\text{ set }r_0^i\gets\mathrm M\\
10:&\qquad\textbf{end if}\\
11:&\quad\textbf{end for}\\
12:&\quad r_s\gets r_0\\
13:&\textbf{end for}\\
14:&\textbf{return }r_0
\end{array}
```

</figure>

**低置信度重掩码按目标保留数量筛选预测，使置信度较低的位置在后续步骤中继续更新**

- $`c^i`$ 为位置 $`i`$ 上预测 token 的概率，$`n_{\mathrm{un}}=\lfloor L(1-s)\rfloor`$ 表示时间 $`s`$ 应保留的 token 数量，因此需要重新掩蔽 $`L-n_{\mathrm{un}}`$ 个位置。原附录将二者混写，下面据作者实现校正计数，并限定从当前掩码位置中选择
- $`\operatorname{Lowest}_k`$ 返回置信度最低的 $`k`$ 个位置索引，置信度相同时固定择序；$`k=0`$ 时返回空集

<figure class="llada-algorithm" aria-labelledby="llada-algorithm-5">
<figcaption id="llada-algorithm-5"><strong>Algorithm 5</strong> Low-confidence Remasking Strategy of LLaDA</figcaption>

```math
\begin{array}{r l}
&\textbf{Require:}\ p_\theta,\ \text{prompt }p_0,\ \text{answer length }L,\ \text{sampling steps }N\\
1:&r_1\gets[\mathrm M,\ldots,\mathrm M]\quad\text{(length }L\text{)}\\
2:&\textbf{for }t\gets 1\textbf{ down to }1/N\textbf{ step }1/N\textbf{ do}\\
3:&\quad s\gets t-1/N\\
4:&\quad\textbf{for }i\gets 1\textbf{ to }L\textbf{ do}\\
5:&\qquad\textbf{if }r_t^i\ne\mathrm M\textbf{ then}\\
6:&\qquad\quad r_0^i\gets r_t^i,\quad c^i\gets 1\\
7:&\qquad\textbf{else}\\
8:&\qquad\quad r_0^i\gets\arg\max_{r_0^i}p_\theta(r_0^i\mid p_0,r_t)\\
9:&\qquad\quad c^i\gets p_\theta(r_0^i\mid p_0,r_t)\\
10:&\qquad\textbf{end if}\\
11:&\quad\textbf{end for}\\
12:&\quad n_{\mathrm{un}}\gets\lfloor L(1-s)\rfloor\\
13:&\quad\textbf{for }i\gets 1\textbf{ to }L\textbf{ do}\\
14:&\qquad\textbf{if }i\in\operatorname{Lowest}_{L-n_{\mathrm{un}}}\!\left(\{(j,c^j):r_t^j=\mathrm M\}\right)\textbf{ then}\\
15:&\qquad\quad r_0^i\gets\mathrm M\\
16:&\qquad\textbf{end if}\\
17:&\quad\textbf{end for}\\
18:&\quad r_s\gets r_0\\
19:&\textbf{end for}\\
20:&\textbf{return }r_0
\end{array}
```

</figure>

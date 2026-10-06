<div class="block-diffusion-note">

# 🧱 Block Diffusion

**Block Diffusion 在块间采用自回归分解，在块内通过离散扩散生成 token。** 已完成的块作为后续生成的条件，其键值缓存可以复用；当前块中的多个位置则共同参与去噪。块长度决定自回归与扩散的结合方式，使模型能够逐块延长输出，并在块内保留并行生成的能力。

论文 [Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models](https://proceedings.iclr.cc/paper_files/paper/2025/file/7ede97c3e082c6df10a8d6103a2eebd2-Paper-Conference.pdf)（ICLR 2025）将这一框架称为 **Block Discrete Denoising Diffusion Language Models（BD3-LMs）**。除模型分解之外，论文依次讨论如何高效计算训练目标，以及如何通过降低梯度估计方差缩小扩散模型与自回归模型之间的困惑度差距。

## 🐻 Background: Language Modeling Paradigms

设 $`x=(x^1,\ldots,x^L)`$ 是来自数据分布 $`q(x)`$ 的 token 序列，每个 token 用词表上的 one-hot 向量表示，$`m`$ 表示特殊的 `[MASK]` token。自回归模型将序列概率分解为逐 token 的条件概率，因此可以并行计算训练损失，但生成时需要依次确定各个位置

```math
\log p_\theta(x)=\sum_{\ell=1}^{L}\log p_\theta(x^\ell\mid x^{<\ell})\tag{1}
```

离散扩散模型从干净序列 $`x`$ 构造受扰动序列 $`x_t`$，再学习从较大噪声时间 $`t`$ 到较小噪声时间 $`s`$ 的逆向转移。D3PM 的去噪网络先预测干净 token，再结合已知后验构造逆过程；各位置在给定当前受噪序列后分别采样，但网络能够读取整个序列的上下文

```math
p_\theta(x_s\mid x_t)
=\prod_{\ell=1}^{L}p_\theta(x_s^\ell\mid x_t),
\qquad
p_\theta(x_s^\ell\mid x_t)
=\sum_{x^\ell}q(x_s^\ell\mid x_t^\ell,x^\ell)\,p_\theta(x^\ell\mid x_t)\tag{2}
```

- 扩散模型通常通过负证据下界（NELBO）训练，该目标给出负对数似然的上界。Block Diffusion 将同一建模方式用于每个块的条件分布，从而把逐 token 的自回归分解推广到逐块分解

## 🦊 Block Diffusion Distributions And Model Architectures

将长度为 $`L`$ 的序列划分为 $`B=L/L'`$ 个等长块，其中 $`L'`$ 是块长度。记第 $`b`$ 块为 $`x^b`$，之前的所有块为 $`x^{<b}`$，模型的对数似然为

```math
\log p_\theta(x)=\sum_{b=1}^{B}\log p_\theta(x^b\mid x^{<b})\tag{3}
```

- 每个条件分布都由一个离散扩散过程定义。去噪仅发生在当前块，前缀 $`x^{<b}`$ 始终保持干净；将各块的 NELBO 相加，即得整条序列的训练目标

```math
\begin{aligned}
p_\theta(x_s^b\mid x_t^b,x^{<b})
&=\sum_{x^b}q(x_s^b\mid x_t^b,x^b)\,p_\theta(x^b\mid x_t^b,x^{<b}),\\[5pt]
-\log p_\theta(x)
&\leq\mathcal L_{\mathrm{BD}}(x;\theta)
:=\sum_{b=1}^{B}\mathcal L(x^b,x^{<b};\theta).
\end{aligned}\tag{4}
```

**所有块共享同一个 Transformer 去噪网络，注意力在块内双向、在块间因果。** 当前块可以读取自身的全部受噪 token 与之前的干净块，不能读取后续块。$`L'=L`$ 对应整段序列上的扩散建模；$`L'=1`$ 对应逐 token 的分解，后文将说明其掩码扩散目标与自回归负对数似然的关系。

- 记 $`x_\theta^b`$ 为共享网络在第 $`b`$ 块上的输出，$`K^{1:b-1},V^{1:b-1}`$ 为干净前缀的键值缓存，则网络接口可写为

```math
x_{\mathrm{logits}}^b,K^b,V^b
\leftarrow x_\theta^b\!\left(x_t^b,K^{1:b-1},V^{1:b-1}\right)
:=x_\theta^b(x_t^b,x^{<b})\tag{5}
```

这里的 $`x_{\mathrm{logits}}^b`$ 用于预测当前块的干净 token。块内去噪会改变当前输入，因此当前块的中间表示仍需更新；已经完成的前缀不再改变，能够在后续步骤中复用缓存。

## 🐇 Efficient Training And Sampling Algorithms

**训练既需要受噪块的预测，也需要干净块的上下文表示。** 同一个块在计算自身损失时作为受噪输入，在计算后续块损失时又作为干净前缀，因此仅对原长度序列执行一次普通前向计算，无法同时满足这两种需求。论文先给出两次前向计算的形式：第一次处理完整干净序列并生成各块的键值表示，第二次让每个受噪块读取对应的干净前缀，再汇总所有块的损失。

<figure class="llada-algorithm bd3-algorithm" aria-labelledby="bd3-algorithm-1">
<figcaption id="bd3-algorithm-1"><strong>Algorithm 1</strong> Block Diffusion Training</figcaption>

```math
\begin{array}{r l}
&\textbf{Input:}\ x,\ B,\ q_t(\cdot\mid x),\ x_\theta,\ \mathcal L_{\mathrm{BD}}\\[-0.15em]
1:&\textbf{repeat}\\[-0.15em]
2:&\quad t_1,\ldots,t_B\sim U[0,1]\\[-0.15em]
3:&\quad x_{t_b}^b\sim q_{t_b}(\cdot\mid x^b),\qquad b=1,\ldots,B\\[-0.15em]
4:&\quad \varnothing,K^{1:B},V^{1:B}\leftarrow x_\theta(x)\\[-0.15em]
5:&\quad x_{\mathrm{logits}}^b,\varnothing,\varnothing\leftarrow x_\theta^b(x_{t_b}^b,K^{1:b-1},V^{1:b-1}),\qquad b=1,\ldots,B\\[-0.15em]
6:&\quad x_{\mathrm{logits}}\leftarrow x_{\mathrm{logits}}^1\oplus\cdots\oplus x_{\mathrm{logits}}^B\\[-0.15em]
7:&\quad\text{Take a gradient step on }\nabla_\theta\mathcal L_{\mathrm{BD}}(x_{\mathrm{logits}};\theta)\\[-0.15em]
8:&\textbf{until }\text{converged}
\end{array}
```

</figure>

**向量化实现将受噪序列与干净序列拼接，在一次前向计算中完成上述两类计算。** 令 $`x_{\mathrm{noisy}}=x_{t_1}^1\oplus\cdots\oplus x_{t_B}^B`$，以 $`x_{\mathrm{noisy}}\oplus x`$ 作为长度为 $`2L`$ 的输入。附录 B.6 使用以下注意力掩码，其中 1 表示允许读取对应的键值

```math
\mathcal M_{\mathrm{full}}
=\begin{bmatrix}
\mathcal M_{\mathrm{BD}}&\mathcal M_{\mathrm{OBC}}\\
0&\mathcal M_{\mathrm{BC}}
\end{bmatrix}\tag{6}
```

- $`\mathcal M_{\mathrm{BD}}`$ 只允许受噪 token 读取同一受噪块；$`\mathcal M_{\mathrm{OBC}}`$ 允许受噪块读取严格位于其之前的干净块，不能读取当前块的干净答案
- $`\mathcal M_{\mathrm{BC}}`$ 允许干净 token 读取自身及之前的干净块；左下角为零，保证干净前缀不受受噪序列影响。这样，所有块的条件预测可以同时计算，并保持与式（4）一致的条件依赖关系

**采样按块顺序推进，每个新块都在已生成前缀的条件下执行扩散去噪。** 一个块生成完成后，以该块的干净 token 计算缓存，再将其追加到前缀中。下面保留论文中抽象的 $`\operatorname{Sample}`$ 接口，它表示从当前块的条件扩散模型采样；$`\oplus`$ 表示序列或缓存的拼接。

<figure class="llada-algorithm bd3-algorithm" aria-labelledby="bd3-algorithm-2">
<figcaption id="bd3-algorithm-2"><strong>Algorithm 2</strong> Block Diffusion Sampling</figcaption>

```math
\begin{array}{r l}
&\textbf{Input:}\ B,\ x_\theta,\ \operatorname{Sample}\\[-0.15em]
1:&x,K,V\leftarrow\varnothing\\[-0.15em]
2:&\textbf{for }b\leftarrow1\textbf{ to }B\textbf{ do}\\[-0.15em]
3:&\quad x^b\leftarrow\operatorname{Sample}(x_\theta^b,K^{1:b-1},V^{1:b-1})\\[-0.15em]
4:&\quad\varnothing,K^b,V^b\leftarrow x_\theta^b(x^b,K^{1:b-1},V^{1:b-1})\\[-0.15em]
5:&\quad x\leftarrow x^{1:b-1}\oplus x^b\\[-0.15em]
6:&\quad(K,V)\leftarrow(K^{1:b-1}\oplus K^b,\ V^{1:b-1}\oplus V^b)\\[-0.15em]
7:&\textbf{end for}\\[-0.15em]
8:&\textbf{return }x
\end{array}
```

</figure>

- 算法以给定块数表示采样过程；变长生成时可以继续追加块，直至满足停止条件。块间仍然顺序生成，块内则可以同时更新多个掩码位置。论文的长文本实验在生成 `[EOS]` 或最近 256 个 token 的平均熵低于 4 时停止

## 🦋 Understanding Likelihood Gaps Between Diffusion And AR Models

**Masked BD3-LMs 使用逐 token 的吸收态掩码过程，并沿用 MDLM 的简化参数化。** 令 $`\alpha_t`$ 为随时间严格递减的保留概率，掩码概率为 $`1-\alpha_t`$；线性调度取 $`\alpha_t=1-t`$，每个位置的前向边缘分布为

```math
q(x_t^\ell\mid x^\ell)
=\operatorname{Cat}\!\left(x_t^\ell;\alpha_t x^\ell+(1-\alpha_t)m\right),
\qquad\alpha_0=1,\quad\alpha_1=0\tag{7}
```

- 去噪网络对干净输出赋予 `[MASK]` 零概率，并直接保留输入中已经可见的 token。因此，未掩码位置的对数预测概率为零，只有掩码位置贡献交叉熵。在连续时间极限下，重构项与先验项均为零，块扩散的 NELBO 简化为

```math
-\log p_\theta(x)\leq\mathcal L_{\mathrm{BD}}(x;\theta)
=\sum_{b=1}^{B}\mathbb E_{t\sim U[0,1]}
\mathbb E_{q(x_t^b\mid x^b)}
\left[\frac{\alpha'_t}{1-\alpha_t}
\log p_\theta(x^b\mid x_t^b,x^{<b})\right]\tag{8}
```

其中 $`\alpha'_t=d\alpha_t/dt<0`$，与非正的对数概率相乘后得到非负损失；块内的对数预测概率是各位置对数概率之和。该式与每个 token 都参与训练的自回归交叉熵，在随机估计方式上存在差异。

**当块长度为 1 时，掩码扩散目标在期望上等于自回归负对数似然。** 此时一个块只有“保留原 token”与“掩码”两种状态。在线性调度下，前者的损失为零，后者以概率 $`t`$ 出现，恰好抵消损失中的 $`1/t`$ 权重。对论文采用的不显式依赖时间的去噪网络，附录 B.4 的推导可以合并为

```math
\begin{aligned}
\mathcal L_{\mathrm{BD}}(x;\theta)
&=-\sum_{b=1}^{L}\mathbb E_t
\left[\frac{q(x_t^b=m\mid x^b)}{t}
\log p_\theta(x^b\mid m,x^{<b})\right]\\[4pt]
&=-\sum_{b=1}^{L}\log p_\theta(x^b\mid m,x^{<b})
=-\log p_\theta(x).
\end{aligned}\tag{9}
```

- 期望目标相同并不保证有限计算预算下的训练结果相同。均匀采样时间时，平均只有一半位置被掩码，梯度来自更少且数量随机的有效 token；这一随机性会增大训练方差。论文在单 token 情形中改用全掩码，使每个位置都提供训练信号，从而恢复与自回归训练一致的目标

在 LM1B 上训练 16B token 后，自回归模型的测试 PPL 为 **22.88**，原始 $`L'=1`$ 块扩散模型报告的 PPL 上界为 **25.56**；采用全掩码调度后，PPL 同样为 **22.88**。这一对照说明，单 token 情形的差距可以来自随机优化，而非概率分解本身。在更大块长度下，模型还需学习块内依赖，不能直接沿用全掩码策略。

**噪声调度不改变完整连续时间 NELBO 的积分值，但会改变蒙特卡洛估计及其梯度的方差。** 设一个批次包含 $`K`$ 条序列，各序列的每个块采样时间 $`t(k,b)`$ 并独立扰动，训练使用的损失估计为

```math
l(X;\theta)=\frac1K\sum_{k=1}^{K}\sum_{b=1}^{B}
\frac{\alpha'_{t(k,b)}}{1-\alpha_{t(k,b)}}
\log p_\theta\!\left(x^{(k),b}\mid x_{t(k,b)}^{(k),b},x^{(k),<b}\right)\tag{10}
```

- 固定模型参数，对 $`M`$ 个批次及其扰动分别计算梯度。论文以梯度向量相对均值的平方距离衡量总方差，其估计式为

```math
\widehat{\operatorname{Var}}[\nabla_\theta l]
=\frac1{M-1}\sum_{m=1}^{M}\left\|g_m-\bar g\right\|_2^2,
\qquad g_m=\nabla_\theta l(X_m;\theta),
\quad\bar g=\frac1M\sum_{m=1}^{M}g_m\tag{11}
```

## 🌿 Low-Variance Noise Schedules For BD3-LMs

**训练调度应覆盖有用的掩码比例，同时减少低信息量扰动带来的梯度波动。** 掩码比例过低时，模型只需重建少量 token；比例过高时，块内上下文几乎消失，预测更多依赖给定前缀后的各位置边缘分布。两种情形都可能提供低信息量的学习信号，因此论文提出限制掩码比例范围的 clipped schedules。

令 $`0\leq\beta<\omega\leq1`$，在区间 $`[\beta,\omega]`$ 内均匀采样掩码率。附录 C.4 通过对均匀采样的时间作线性映射实现这一操作；范围参数按训练损失估计的方差选择

```math
\begin{aligned}
1-\alpha_t&=\beta+(\omega-\beta)t,\qquad t\sim U[0,1],\\[4pt]
(\beta^*,\omega^*)&=\arg\min_{\beta,\omega}
\operatorname{Var}_{X,t}\!\left[\mathcal L_{\mathrm{BD}}(X;\theta,\beta,\omega)\right].
\end{aligned}\tag{12}
```

- 直接反复估计全部参数的梯度方差开销较高，因此论文使用损失估计的方差作为代理，在训练期间通过网格搜索调整 $`\beta,\omega`$。实际实验约每 5K 次梯度更新进行一次搜索；不同块长度使用各自的掩码区间
- 第 5.2 节将截断采样解释为一种在有效区间外接近端点的连续调度近似。附录 C.4 中的线性映射描述的是训练时使用的掩码率范围；测试似然时仍在完整的 $`[0,1]`$ 区间采样，按式（8）评估 NELBO

原文表 8 在相同的 3B token 微调设置下比较不同调度。小块与较大块偏好的区间不同，训练方差较低的设置也获得更好的困惑度；表中的 PPL 均由 NELBO 估计，对应真实困惑度的上界。

| 块长度 | 掩码率分布 | PPL 上界 ↓ | NELBO 方差 ↓ |
| --- | --- | ---: | ---: |
| $`L'=4`$ | $`U[0.45,0.95]`$ | **29.21** | **6.24** |
| $`L'=4`$ | $`U[0.3,0.8]`$ | 29.38 | 10.33 |
| $`L'=4`$ | $`U[0,1]`$ | 30.18 | 23.45 |
| $`L'=16`$ | $`U[0.45,0.95]`$ | 31.42 | 3.60 |
| $`L'=16`$ | $`U[0.3,0.8]`$ | **31.12** | **3.58** |
| $`L'=16`$ | $`U[0,1]`$ | 31.72 | 7.62 |

## 🧪 Experiments

**主实验先训练整段扩散模型，再切换为较小块长度，并同时优化噪声调度。** 模型先以 $`L'=L`$ 训练 850K 步，再以选定块长度微调 150K 步。LM1B 的训练上下文为 128 token，总训练量为 65B token；OpenWebText（OWT）的训练上下文为 1024 token，总训练量为 524B token。AR、SEDD、MDLM 与 BD3-LMs 使用相同规模的 110M 参数 Transformer，包含 12 层、768 维隐藏状态和 12 个注意力头。

- 似然评估使用覆盖完整噪声区间的低差异时间采样。下表整理原文表 3、4，扩散模型一栏保留原论文的上界符号；BD3-LMs 在两个数据集上均改善扩散基线，但仍与 AR 存在差距

| 模型 | LM1B PPL ↓ | OWT PPL ↓ |
| --- | ---: | ---: |
| AR | 22.83 | 17.54 |
| SEDD | ≤ 32.68 | ≤ 24.10 |
| MDLM | ≤ 31.78 | ≤ 22.98 |
| BD3-LMs，$`L'=16`$ | ≤ 30.60 | ≤ 22.27 |
| BD3-LMs，$`L'=8`$ | ≤ 29.83 | ≤ 21.68 |
| BD3-LMs，$`L'=4`$ | **≤ 28.23** | **≤ 20.73** |

**变长生成由逐块条件分布支持，输出可以超过训练时的上下文长度。** 在 500 条 OWT 生成样本中，$`L'=16`$ 的 BD3-LM 输出长度中位数为 798 token，最大长度为 9982 token；对应 SEDD 的最大长度为 1024 token。该结果说明逐块采样可以延长生成序列，质量仍受停止条件与误差累积影响。

- 论文另用 GPT2-Large 评估生成文本的困惑度。原文表 7 中，生成 1024 token 时，AR、MDLM 与 $`L'=4`$ 的 BD3-LM 分别为 **14.1、46.8、25.7**；生成 2048 token 时分别为 **13.2、41.3、23.6**。这些数值衡量生成样本在外部语言模型下的概率，与上表测试数据的似然指标含义不同
- 表 7 中 BD3-LMs 与 MDLM 采用 5K 个扩散时间步，并缓存输入未改变时的去噪预测，使实际网络评估次数随 token 解掩码事件增加；报告的 1024 与 2048 token 生成预算分别约为 1K 与 2K 次网络评估。该实验比较接近网络评估预算下的样本质量，端到端速度还取决于块长度与采样实现

**向量化训练的收益来自同时计算干净前缀与受噪块，性能改善则同时涉及块分解和噪声调度。** 相较显式执行两次前向计算，论文的向量化实现获得约 20%–25% 的训练加速。块间生成仍然依赖先前结果，较小块长度会增加顺序依赖；块长度的选择需要结合似然、生成效率与任务需求。

</div>

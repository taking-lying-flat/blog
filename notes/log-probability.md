# Cross-Entropy

# 🦊 Binary logistic regression, binary cross-entropy loss

考虑包含 $`m`$ 个样本的二分类数据集，第 $`i`$ 个样本的输入为 $`x^{(i)}`$，标签为 $`y^{(i)}\in\{0,1\}`$。模型通过 logistic 函数将得分映射为类别概率：

```math
\hat y^{(i)}=h\!\left(f(x^{(i)};\theta)\right)
=\frac{1}{1+e^{-f(x^{(i)};\theta)}}.\tag{1}
```

其中，$`f(x;\theta)`$ 为参数化得分函数，$`\theta`$ 为模型参数，$`h`$ 为 logistic 函数；式 (1) 对应 PyTorch 的 `torch.sigmoid(model(x))`。输出 $`\hat y^{(i)}`$ 表示给定输入时标签取 1 的概率，相应的伯努利条件分布为

```math
P(y^{(i)}=1\mid x^{(i)};\theta)=\hat y^{(i)},\qquad P(y^{(i)}=0\mid x^{(i)};\theta)=1-\hat y^{(i)}.\tag{2}
```

```math
P(y^{(i)}\mid x^{(i)};\theta)
=\bigl(\hat y^{(i)}\bigr)^{y^{(i)}}
\bigl(1-\hat y^{(i)}\bigr)^{1-y^{(i)}}
=\begin{cases}
\hat y^{(i)},&y^{(i)}=1,\\
1-\hat y^{(i)},&y^{(i)}=0.
\end{cases}
\tag{3}
```

最大似然估计通过最大化观测标签的联合条件概率确定参数 $`\theta`$。给定输入后，若各样本的标签条件独立，则数据集 $`X,y`$ 的似然函数为

```math
\mathcal L(\theta;X,y)
=\prod_{i=1}^{m}P(y^{(i)}\mid x^{(i)};\theta)=\prod_{i=1}^{m}
\bigl(\hat y^{(i)}\bigr)^{y^{(i)}}
\bigl(1-\hat y^{(i)}\bigr)^{1-y^{(i)}}.
\tag{4}
```

```math
\theta_{\mathrm{MLE}}
=\underset{\theta}{\operatorname{argmax}}\,
\mathcal L(\theta;X,y).\tag{5}
```

对数函数在正数域上严格递增，因此最大化似然与最大化对数似然具有相同的最优解：

```math
\theta_{\mathrm{MLE}}
=\underset{\theta}{\operatorname{argmax}}\,\mathcal L
=\underset{\theta}{\operatorname{argmax}}\,\log\mathcal L.
\tag{6}
```

记对数似然为 $`\ell(\theta)=\log\mathcal L(\theta)`$，展开乘积可得

```math
\begin{aligned}
\ell(\theta)
&=\log\mathcal L(\theta)
=\log\prod_{i=1}^{m}
\bigl(\hat y^{(i)}\bigr)^{y^{(i)}}
\bigl(1-\hat y^{(i)}\bigr)^{1-y^{(i)}} =\sum_{i=1}^{m}\log\left[
\bigl(\hat y^{(i)}\bigr)^{y^{(i)}}
\bigl(1-\hat y^{(i)}\bigr)^{1-y^{(i)}}\right]\\[4pt]
&=\sum_{i=1}^{m}\left[
y^{(i)}\log\hat y^{(i)}
+(1-y^{(i)})\log(1-\hat y^{(i)})\right].
\end{aligned}\tag{7}
```

负对数似然即为按样本求和的二元交叉熵损失：

```math
J(\theta)=-\ell(\theta)
=-\sum_{i=1}^{m}\left[
y^{(i)}\log\hat y^{(i)}
+(1-y^{(i)})\log(1-\hat y^{(i)})\right].\tag{8}
```

多分类任务的标签空间包含 $`C`$ 个类别，自回归语言建模则在词表 $`V`$ 上估计下一 token 的条件概率 $`P(w_i\mid x)`$。记各类别得分为 $`f_j(x;\theta_j)`$。类别概率之和必须为 1，对各类别独立应用 sigmoid 无法保证这一约束；softmax 将得分向量 $`t=(t_1,\ldots,t_C)`$ 归一化为概率向量：

```math
\operatorname{softmax}(t_1,\ldots,t_C)
=\left[
\frac{e^{t_1}}{\sum_{j=1}^{C}e^{t_j}},\ldots,
\frac{e^{t_C}}{\sum_{j=1}^{C}e^{t_j}}
\right]^{\mathsf T}.\tag{9}
```

当得分均为有限值时，softmax 的各分量严格为正且总和为 1。多分类概率模型及其负对数似然为

```math
\begin{gathered}
\hat y^{(i)}
=\left[\hat P(y^{(i)}=1\mid x^{(i)};\theta),\ldots,
\hat P(y^{(i)}=C\mid x^{(i)};\theta)\right]^{\mathsf T} =\operatorname{softmax}\!\left(f_1(x^{(i)};\theta_1),\ldots,f_C(x^{(i)};\theta_C)\right)\\[4pt]
=\left[
\frac{\exp(f_1(x^{(i)};\theta_1))}{\sum_{j=1}^{C}\exp(f_j(x^{(i)};\theta_j))},\ldots,
\frac{\exp(f_C(x^{(i)};\theta_C))}{\sum_{j=1}^{C}\exp(f_j(x^{(i)};\theta_j))}
\right]^{\mathsf T}.
\end{gathered}\tag{10}
```

```math
\begin{aligned}
J(\theta)=-\ell(\theta)
&=-\log\prod_{i=1}^{m}\hat P(y^{(i)}\mid x^{(i)};\theta)
=-\log\prod_{i=1}^{m}\hat y^{(i)}_{y^{(i)}}\\[4pt]
&=-\log\prod_{i=1}^{m}
\operatorname{softmax}\!\left(
f_1(x^{(i)};\theta_1),\ldots,f_C(x^{(i)};\theta_C)
\right)_{y^{(i)}} =-\sum_{i=1}^{m}\log
\frac{\exp(f_{y^{(i)}}(x^{(i)};\theta_{y^{(i)}}))}
{\sum_{j=1}^{C}\exp(f_j(x^{(i)};\theta_j))}.
\end{aligned}\tag{11}
```

神经网络的分类头通常为输出维度等于类别数 $`C`$ 的线性层，其输出称为 logits，经 softmax 归一化后得到类别概率。式 (11) 给出了多分类交叉熵损失，PyTorch 的 `CrossEntropyLoss` 直接接收 logits，即式中的 $`f_j`$。

交叉熵是模型预测概率的负对数在目标分布下的期望。硬标签监督下，逐样本交叉熵等于观测标签的负对数似然；自回归语言模型根据前缀预测下一 token，序列负对数似然因而分解为逐 token 损失之和：

```math
\mathcal L_{\mathrm{NLL}}(x_{1:T};\theta)
=-\ln P_\theta(x_{1:T})
=-\sum_{t=1}^{T}\ln P_\theta(x_t\mid x_{<t}).\tag{12}
```

式 (12) 的每一项是 token 在模型分布下的条件自信息，单位为 nat；除以 $`\ln 2`$ 后，可解释为以 bit 计量的[理想码长](https://d2l.smola.org/chapter_mdl-information-theory/mdl-coding-compression.html)。对有效 token 取平均时，损失单位为 nat/token。该码长刻画概率编码代价。

# 🐻 Entropy and cross-entropy view, equivalence of KL-divergence minimization and maximum likelihood

设离散随机变量 $`X`$ 服从分布 $`P_X`$，其 Shannon 熵定义为

```math
H(P_X)=-\sum_{x\in\mathcal X}P_X(x)\log P_X(x).
\tag{13}
```

事件的自信息 $`-\log P_X(x)`$ 随概率降低而增大，独立事件的联合自信息等于各事件自信息之和。自信息在 $`P_X`$ 下的期望即为熵，用于度量分布的不确定性；若采用模型分布 $`Q_X`$ 计算信息量，并在真实分布 $`P_X`$ 下取期望，则得到交叉熵：

```math
H(P_X,Q_X)=-\sum_{x\in\mathcal X}P_X(x)\log Q_X(x).
\tag{14}
```

交叉熵一般不满足对称性，其与目标分布自身的熵之差为 KL 散度：

```math
D_{\mathrm{KL}}(P\Vert Q)
=\left[-\sum_{x\in\mathcal X}P(x)\log Q(x)\right]
-\left[-\sum_{x\in\mathcal X}P(x)\log P(x)\right] =H(P,Q)-H(P)
=\sum_{x\in\mathcal X}P(x)\log\frac{P(x)}{Q(x)}.
\tag{15}
```

KL 散度衡量按 $`Q`$ 编码来自 $`P`$ 的样本时增加的期望编码代价，其值非负，且仅在两分布相同时为零。对于 softmax 分类模型，记真实条件分布为 $`P_{Y\mid X}`$、模型分布为 $`\hat P_{Y\mid X}`$，固定输入 $`x`$ 后有

```math
\begin{aligned}
D_{\mathrm{KL}}\!\left(P_{Y\mid X=x}\Vert\hat P_{Y\mid X=x};\theta\right)
&=-\sum_{j=1}^{C}P(y=j\mid x)\log\hat P(y=j\mid x;\theta) +\sum_{j=1}^{C}P(y=j\mid x)\log P(y=j\mid x)\\[4pt]
&=\sum_{j=1}^{C}P(y=j\mid x)
\log\frac{P(y=j\mid x)}{\hat P(y=j\mid x;\theta)}.
\end{aligned}\tag{16}
```

若保持输入边缘分布 $`P_X`$ 不变，仅对标签条件分布建模，则 $`\hat P_{X,Y}(x,y;\theta)=P_X(x)\hat P_{Y\mid X}(y\mid x;\theta)`$。由联合分布计算 KL 散度时，概率比值中的 $`P_X(x)`$ 因子相消：

```math
\begin{aligned}
D_{\mathrm{KL}}(P_{X,Y}\Vert\hat P_{X,Y};\theta)
&=\sum_{x,y}P_{X,Y}(x,y)
\log\frac{P_{X,Y}(x,y)}{\hat P_{X,Y}(x,y;\theta)} =\sum_{x,y}P_{X,Y}(x,y)
\log\frac{P(y\mid x)\cancel{P_X(x)}}
{\hat P(y\mid x;\theta)\cancel{P_X(x)}}\\[4pt]
&=\mathbb E_{X,Y}\!\left[
\log\frac{P(Y\mid X)}{\hat P(Y\mid X;\theta)}\right].
\end{aligned}\tag{17}
```

给定 $`n`$ 个独立同分布样本 $`(x^{(i)},y^{(i)})`$，若对数概率比可积，则对固定参数 $`\theta`$，大数定律给出

```math
\frac{1}{n}\sum_{i=1}^{n}
\log\frac{P(y^{(i)}\mid x^{(i)})}
{\hat P(y^{(i)}\mid x^{(i)};\theta)}
\xrightarrow[n\to\infty]{}
\mathbb E_{X,Y}\!\left[
\log\frac{P(Y\mid X)}{\hat P(Y\mid X;\theta)}\right] =D_{\mathrm{KL}}(P_{X,Y}\Vert\hat P_{X,Y};\theta).
\tag{18}
```

式 (18) 以样本均值近似总体 KL 目标。若进一步满足最优解唯一、经验目标一致收敛及极小值稳定性等正则条件，经验最优参数可收敛至总体最优参数：

```math
\underset{\theta}{\operatorname{argmin}}
\left\{\frac{1}{n}\sum_{i=1}^{n}
\log\frac{P(y^{(i)}\mid x^{(i)})}
{\hat P(y^{(i)}\mid x^{(i)};\theta)}\right\}\;\xrightarrow[n\to\infty]{}
\underset{\theta}{\operatorname{argmin}}
D_{\mathrm{KL}}(P_{X,Y}\Vert\hat P_{X,Y};\theta).
\tag{19}
```

对于固定数据集，真实分布项与 $`\theta`$ 无关，经验目标的最小化等价于最大似然估计：

```math
\begin{aligned}
&\underset{\theta}{\operatorname{argmin}}
\left\{\frac{1}{n}\sum_{i=1}^{n}
\log\frac{P(y^{(i)}\mid x^{(i)})}
{\hat P(y^{(i)}\mid x^{(i)};\theta)}\right\} =\underset{\theta}{\operatorname{argmax}}
\left\{\frac{1}{n}\sum_{i=1}^{n}
\log\hat P(y^{(i)}\mid x^{(i)};\theta)\right\}\\[4pt]
&\quad=\underset{\theta}{\operatorname{argmax}}
\left\{\frac{1}{n}\log\prod_{i=1}^{n}
\hat P(y^{(i)}\mid x^{(i)};\theta)\right\} =\underset{\theta}{\operatorname{argmax}}
\left\{\frac{1}{n}\ell(\theta)\right\}.
\end{aligned}\tag{20}
```

经验 KL 目标与最大似然估计由此对应同一优化问题。在总体层面，真实分布的熵不依赖 $`\theta`$，因此最小化 KL 散度等价于最小化交叉熵：

```math
\underset{\theta}{\operatorname{argmin}}
D_{\mathrm{KL}}(P_{X,Y}\Vert\hat P_{X,Y};\theta)
=\underset{\theta}{\operatorname{argmin}}
\left\{\underbrace{H(P_{X,Y},\hat P_{X,Y})}_{\text{cross-entropy}}
-\underbrace{H(P_{X,Y})}_{\text{entropy}}\right\} =\underset{\theta}{\operatorname{argmin}}
H(P_{X,Y},\hat P_{X,Y}).
\tag{21}
```

# 🐰 The special case of model distillation

模型蒸馏以教师分布 $`P^{\mathrm{teacher}}(y\mid x)`$ 为监督目标，使学生模型拟合教师的预测分布。下文以 $`P_{Y\mid X}`$ 表示教师分布，以 $`\hat P_{Y\mid X}`$ 表示学生分布；保持输入边缘分布不变，将联合分布分解为边缘分布与条件分布，可得

```math
\begin{aligned}
&\underset{\theta}{\operatorname{argmin}}
D_{\mathrm{KL}}(P_{X,Y}\Vert\hat P_{X,Y};\theta)
=\underset{\theta}{\operatorname{argmin}}H(P_{X,Y},\hat P_{X,Y})\\[4pt]
&\quad=\underset{\theta}{\operatorname{argmin}}
\left\{-\sum_{x,y}P_{X,Y}(x,y)\log\hat P_{X,Y}(x,y;\theta)\right\}\\[4pt]
&\quad=\underset{\theta}{\operatorname{argmin}}
\left\{-\sum_x P_X(x)\sum_y P_{Y\mid X}(y\mid x)
\log\!\left[\hat P_{Y\mid X}(y\mid x;\theta)P_X(x)\right]\right\}\\[4pt]
&\quad=\underset{\theta}{\operatorname{argmin}}
\left\{-\sum_x P_X(x)\sum_y P_{Y\mid X}(y\mid x)
\left[\log\hat P_{Y\mid X}(y\mid x;\theta)+\log P_X(x)\right]\right\}\\[4pt]
&\quad=\underset{\theta}{\operatorname{argmin}}\left\{
-\sum_x P_X(x)\sum_y P_{Y\mid X}(y\mid x)
\log\hat P_{Y\mid X}(y\mid x;\theta)-\underbrace{\sum_x P_X(x)\sum_y P_{Y\mid X}(y\mid x)\log P_X(x)}
_{-H(P_X),\;\text{independent of }\theta}
\right\}\\[4pt]
&\quad=\underset{\theta}{\operatorname{argmin}}
\left\{-\sum_x P_X(x)\sum_y P_{Y\mid X}(y\mid x)
\log\hat P_{Y\mid X}(y\mid x;\theta)\right\}\\[4pt]
&\quad=\underset{\theta}{\operatorname{argmin}}\left\{-\mathbb E_X\!\left[
\sum_y P_{Y\mid X}(y\mid X)
\log\hat P_{Y\mid X}(y\mid X;\theta)\right]\right\}.
\end{aligned}\tag{22}
```

输入边缘分布固定，因此 $`H(P_X)`$ 不依赖 $`\theta`$，可从优化目标中略去。教师为每个输入提供完整的条件分布，类别维度可直接求和，只需以样本均值估计对 $`X`$ 的期望。当输入样本独立同分布且相应期望存在时，对固定参数 $`\theta`$ 有

```math
R_n(\theta)
=-\frac{1}{n}\sum_{i=1}^{n}\sum_{y\in\mathcal Y}
P_{Y\mid X}(y\mid x^{(i)})
\log\hat P_{Y\mid X}(y\mid x^{(i)};\theta) \xrightarrow[n\to\infty]{}
-\mathbb E_X\!\left[\sum_{y\in\mathcal Y}
P_{Y\mid X}(y\mid X)
\log\hat P_{Y\mid X}(y\mid X;\theta)\right].
\tag{23}
```

在式 (19) 所述的一致性条件下，经验最优参数可收敛至总体最优参数。采用 softmax 参数化时，预测条件概率为

```math
\hat P_{Y\mid X}(y\mid x;\theta)
=\frac{\exp(f_y(x;\theta_y))}
{\sum_{j=1}^{C}\exp(f_j(x;\theta_j))}.
\tag{24}
```

将式 (24) 代入经验目标，得到概率标签下的平均交叉熵损失：

```math
J(\theta)
=-\frac{1}{n}\sum_{i=1}^{n}\sum_{y\in\mathcal Y}
P_{Y\mid X}(y\mid x^{(i)})
\log\hat P_{Y\mid X}(y\mid x^{(i)};\theta) =-\frac{1}{n}\sum_{i=1}^{n}\sum_{y\in\mathcal Y}
P_{Y\mid X}(y\mid x^{(i)})
\log\frac{\exp(f_y(x^{(i)};\theta_y))}
{\sum_{j=1}^{C}\exp(f_j(x^{(i)};\theta_j))}.
\tag{25}
```

式 (25) 对应 PyTorch 在概率标签、类别权重均为 1 时的平均交叉熵。[`CrossEntropyLoss`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html) 支持类别索引与概率标签，后者适用于混合标签与分布蒸馏。沿用其记号，$`x_{n,c}`$ 为 logit，$`y_{n,c}`$ 为目标概率，$`w_c`$ 为类别权重，$`C`$ 为类别数，$`N`$ 为批量及额外位置维度上的位置总数。逐位置损失为

```math
\ell(x,y)=L=(\ell_1,\ldots,\ell_N)^{\mathsf T},
\qquad
\ell_n=-\sum_{c=1}^{C}w_c y_{n,c}
\log\frac{\exp(x_{n,c})}{\sum_{i=1}^{C}\exp(x_{n,i})}.
\tag{26}
```

对于概率标签，`mean` 与 `sum` 两种归约方式分别为

```math
\ell(x,y)=\begin{cases}
\dfrac{1}{N}\displaystyle\sum_{n=1}^{N}\ell_n,
&\text{reduction}=\texttt{mean},\\[8pt]
\displaystyle\sum_{n=1}^{N}\ell_n,
&\text{reduction}=\texttt{sum}.
\end{cases}\tag{27}
```

# 🐼 Log-Probability Computation

设训练序列为 $`x=(x_0,\ldots,x_T)`$，其中 $`x_0`$ 为起始 token。自回归预训练对后续所有 token 计算负对数似然，仅监督回答的 SFT 则仅在回答位置计算损失。若 $`x_{1:M}`$ 为提示词，$`x_{M+1:T}`$ 为参考回答，则

```math
\begin{aligned}
\mathcal L_{\mathrm{PT}}(\theta)
&=-\frac{1}{T}\sum_{t=0}^{T-1}\ln P_\theta(x_{t+1}\mid x_{\le t}),\\[4pt]
\mathcal L_{\mathrm{SFT}}(\theta)
&=-\frac{1}{T-M}\sum_{t=M}^{T-1}\ln P_\theta(x_{t+1}\mid x_{\le t}).
\end{aligned}\tag{28}
```

语言模型在每个位置执行词表分类，以前缀 $`x_{\le t}`$ 为输入，以下一 token $`x_{t+1}`$ 为观测类别。记词表大小为 $`V`$、当前位置的 logits 为 $`z_t\in\mathbb R^V`$，观测标签可表示为 one-hot 向量 $`q_{t,j}=\mathbf 1\{j=x_{t+1}\}`$。该向量编码当前样本的监督目标，并不意味着真实条件分布必须为 one-hot。代入交叉熵后，词表求和仅保留目标 token 对应的项：

```math
\begin{aligned}
\ell_t
&=\operatorname{CE}(q_t,\operatorname{softmax}(z_t))\\[4pt]
&=-\sum_{j=0}^{V-1}q_{t,j}\ln\operatorname{softmax}(z_t)_j\\[4pt]
&=-\ln P_\theta(x_{t+1}\mid x_{\le t})
=\ell_t^{\mathrm{NLL}}.
\end{aligned}\tag{29}
```

隐藏状态 $`h_t`$ 经 LM head 投影为 $`z_t=Wh_t+b`$。目标 token 的 log-prob 等于目标 logit 与全词表 log-sum-exp 之差：

```math
\begin{aligned}
\ln P_\theta(x_{t+1}\mid x_{\le t})
&=\operatorname{logsoftmax}(z_t)_{x_{t+1}}\\[4pt]
&=\ln\frac{\exp(z_{t,x_{t+1}})}
{\sum_{j=0}^{V-1}\exp(z_{t,j})}\\[4pt]
&=z_{t,x_{t+1}}-\ln\sum_{j=0}^{V-1}\exp(z_{t,j})\\[4pt]
&=z_{t,x_{t+1}}-\ln\left[
\exp(m_t)\sum_{j=0}^{V-1}\exp(z_{t,j}-m_t)\right]\\[4pt]
&=z_{t,x_{t+1}}-
\left[m_t+\ln\sum_{j=0}^{V-1}\exp(z_{t,j}-m_t)\right],
\qquad m_t=\max_j z_{t,j}.
\end{aligned}\tag{30}
```

式 (30) 将概率比值的对数展开为两项之差，并从指数和中提取 $`\exp(m_t)`$。平移后的指数输入均不大于零，从而避免指数上溢；结果取负即得到式 (29) 的逐 token 损失。

<strong>教师强制</strong>（teacher forcing）将真实 token 序列作为 `input_ids`，经嵌入层和带因果掩码的 Transformer 得到隐藏状态。位置 $`t`$ 仅依赖真实前缀 $`x_{\le t}`$，以 $`x_{t+1}`$ 为监督目标，因此各位置的条件概率可在一次前向传播中并行计算。预训练与 SFT 均采用真实前缀进行预测。

Transformers [`LlamaForCausalLM.forward`（v4.44.2）](https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/llama/modeling_llama.py#L1208) 接收与 `input_ids` 位置对齐的 `labels`，并在损失分支中将 logits 与标签错开一位：

```python Transformers · LlamaForCausalLM.forward
logits = self.lm_head(hidden_states)
logits = logits.float()

if labels is not None:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, self.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
```

`labels[..., 1:]` 提供式 (29) 中 one-hot 向量非零项的类别索引，`logits[..., :-1, :]` 提供对应前缀的预测。这一错位使前缀预测与下一 token 对齐；计算时仅需类别索引，无需显式构造 one-hot 张量。`CrossEntropyLoss` 忽略标记为 `-100` 的位置，并对其余标签计算负对数似然。

**Megatron Core**：[词表并行交叉熵](https://github.com/NVIDIA/Megatron-LM/blob/37cb4f261f103a7c882b5e310082fd0b7dbb226f/megatron/core/tensor_parallel/cross_entropy.py#L159)将式 (30) 的计算分配至各张量并行 rank。MAX 归约得到全词表最大值，SUM 归约汇总目标 logit 与指数和。以下前向计算取 `label_smoothing=0`：

```python Megatron Core · vocab_parallel_cross_entropy
import torch.distributed as dist

z = vocab_parallel_logits.float()                 # [..., V_local]
m = z.amax(dim=-1)
dist.all_reduce(m, op=dist.ReduceOp.MAX, group=tp_group)
z = z - m.unsqueeze(-1)

outside = (target < vocab_start_index) | (target >= vocab_end_index)
local_id = (target - vocab_start_index).masked_fill(outside, 0)
target_z = z.gather(-1, local_id.unsqueeze(-1)).squeeze(-1)
target_z = target_z.masked_fill(outside, 0)
sum_exp = z.exp().sum(dim=-1)

dist.all_reduce(target_z, op=dist.ReduceOp.SUM, group=tp_group)
dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=tp_group)
log_prob = target_z - sum_exp.log()               # 等于 -CE
```

`target` 为全局词表 ID，仅包含该 token 的词表分片提供目标 logit，其余 rank 的贡献为零。`target_z` 已减去全局最大值 $`m_t`$，再减去 `sum_exp.log()` 即得到 log-prob，也就是逐 token 交叉熵的相反数。

**vLLM**：[指定 token 的 kernel](https://github.com/vllm-project/vllm/blob/dfebb42d096d5c6a04aa5a1722195e37cb8e0b2d/vllm/v1/worker/gpu/sample/logprob.py#L17)在同一计算内核中完成全词表归约与目标 logit 提取。`row_ptr` 指向当前行，`topk_ids` 为待评分 token 的词表 ID：

```python vLLM · _topk_log_softmax_kernel
max_val = float("-inf")
for i in range(0, vocab_size, BLOCK_SIZE):
    block = i + tl.arange(0, BLOCK_SIZE)
    z = tl.load(row_ptr + block, mask=block < vocab_size, other=-float("inf"))
    max_val = tl.max(tl.maximum(z, max_val))
max_val = max_val.to(tl.float32)

sum_exp = 0.0
for i in range(0, vocab_size, BLOCK_SIZE):
    block = i + tl.arange(0, BLOCK_SIZE)
    z = tl.load(row_ptr + block, mask=block < vocab_size, other=0.0)
    e = tl.exp(z.to(tl.float32) - max_val)
    sum_exp += tl.sum(tl.where(block < vocab_size, e, 0.0))

target_z = tl.load(row_ptr + topk_ids).to(tl.float32)
log_prob = target_z - max_val - tl.log(sum_exp)
```

两次遍历分别计算整行最大值与 FP32 指数和。由于 `target_z` 保留原始 logit，计算时需同时减去 `max_val` 和 `log(sum_exp)`，归一化分母覆盖完整词表。

**Liger-Kernel**：[`_selective_logprob_forward`](https://github.com/linkedin/Liger-Kernel/blob/4dce24bbb70d8bda18e373b5f0f469a51ab77a2c/src/liger_kernel/chunked_loss/fused_linear_ppo.py#L18)分块计算词表投影，并在线更新最大值与指数和。以下代码省略外层的位置分块，取 `bias=None`：

```python Liger-Kernel · _selective_logprob_forward
# h: 当前块隐藏状态 [C, D]；targets: [C]；weight: [V, D]
m = torch.full((h.shape[0],), -float("inf"), device=h.device, dtype=torch.float32)
sum_exp = torch.zeros_like(m)
target_z = torch.zeros_like(m)

for start in range(0, weight.shape[0], vocab_chunk_size):
    end = min(start + vocab_chunk_size, weight.shape[0])
    z = (h @ weight[start:end].to(h.dtype).T).float()

    m_new = torch.maximum(m, z.amax(dim=-1))
    sum_exp = sum_exp * torch.exp(m - m_new) + (
        z - m_new.unsqueeze(-1)
    ).exp().sum(dim=-1)
    m = m_new

    in_chunk = (targets >= start) & (targets < end)
    local_id = (targets - start).clamp(0, end - start - 1)
    target_z += z.gather(-1, local_id.unsqueeze(-1)).squeeze(-1) * in_chunk

log_z = m + sum_exp.log()
log_prob = target_z - log_z
```

最大值由 `m` 更新为 `m_new` 时，先以 `exp(m - m_new)` 缩放已有指数和，再累加当前块的指数和。目标 token 所在块提供 `target_z`；遍历结束后，`log_z` 为全词表的 log-sum-exp，并保存用于反向传播时的重计算。

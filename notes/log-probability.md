# Cross-Entropy

# Cross-Entropy Loss

设二分类数据集为

```math
\mathcal D=\{(x^{(i)},y^{(i)})\}_{i=1}^{m},
\qquad y^{(i)}\in\{0,1\}.\tag{1}
```

模型通过 logistic 函数将得分映射为类别概率：

```math
\hat y^{(i)}=h\!\left(f(x^{(i)};\theta)\right)
=\frac{1}{1+e^{-f(x^{(i)};\theta)}}.\tag{2}
```

其中，$`f(x;\theta)`$ 为参数化得分函数，$`\theta`$ 为模型参数，$`h`$ 为 logistic 函数，式 (2) 对应 PyTorch 的 `torch.sigmoid(model(x))`。将输出 $`\hat y^{(i)}`$ 解释为标签取 1 的条件概率，则

```math
\begin{aligned}
P(y^{(i)}=1\mid x^{(i)};\theta)&=\hat y^{(i)},\\
P(y^{(i)}=0\mid x^{(i)};\theta)&=1-\hat y^{(i)}.
\end{aligned}\tag{3}
```

两种情形可统一写为伯努利分布的概率质量函数：

```math
\begin{aligned}
P(y^{(i)}\mid x^{(i)};\theta)
&=\bigl(\hat y^{(i)}\bigr)^{y^{(i)}}
\bigl(1-\hat y^{(i)}\bigr)^{1-y^{(i)}}\\[4pt]
&=\begin{cases}
\hat y^{(i)},&y^{(i)}=1,\\
1-\hat y^{(i)},&y^{(i)}=0.
\end{cases}
\end{aligned}\tag{4}
```

最大似然估计选择使观测标签的联合条件概率最大的参数 $`\theta`$。对相互独立的样本，数据集 $`X,y`$ 的似然函数为

```math
\begin{aligned}
\mathcal L(\theta;X,y)
&=\prod_{i=1}^{m}P(y^{(i)}\mid x^{(i)};\theta)\\[4pt]
&=\prod_{i=1}^{m}
\bigl(\hat y^{(i)}\bigr)^{y^{(i)}}
\bigl(1-\hat y^{(i)}\bigr)^{1-y^{(i)}}.
\end{aligned}\tag{5}
```

相应的参数估计为

```math
\theta_{\mathrm{MLE}}
=\underset{\theta}{\operatorname{argmax}}\,
\mathcal L(\theta;X,y).\tag{6}
```

由于对数函数在正数域上严格递增，最大化似然与最大化对数似然具有相同的最优解：

```math
\theta_{\mathrm{MLE}}
=\underset{\theta}{\operatorname{argmax}}\,\mathcal L
=\underset{\theta}{\operatorname{argmax}}\,\log\mathcal L.
\tag{7}
```

记 $`\ell(\theta)=\log\mathcal L(\theta)`$，将乘积转为对数之和，得到

```math
\begin{aligned}
\ell(\theta)
&=\log\prod_{i=1}^{m}
\bigl(\hat y^{(i)}\bigr)^{y^{(i)}}
\bigl(1-\hat y^{(i)}\bigr)^{1-y^{(i)}}\\[4pt]
&=\sum_{i=1}^{m}\log\left[
\bigl(\hat y^{(i)}\bigr)^{y^{(i)}}
\bigl(1-\hat y^{(i)}\bigr)^{1-y^{(i)}}\right]\\[4pt]
&=\sum_{i=1}^{m}\left[
y^{(i)}\log\hat y^{(i)}
+(1-y^{(i)})\log(1-\hat y^{(i)})\right].
\end{aligned}\tag{8}
```

对数似然取负即为二元交叉熵损失：

```math
J(\theta)=-\ell(\theta)
=-\sum_{i=1}^{m}\left[
y^{(i)}\log\hat y^{(i)}
+(1-y^{(i)})\log(1-\hat y^{(i)})\right].\tag{9}
```

多分类将标签空间扩展为 $`C`$ 个类别；自回归语言建模即在词表 $`V`$ 上估计下一 token 的条件概率 $`P(w_i\mid x)`$。记各类别得分为 $`f_j(x;\theta_j)`$。类别概率之和须为 1，独立应用 sigmoid 不满足此约束；softmax 则将得分向量 $`t=(t_1,\ldots,t_C)`$ 映射为概率向量：

```math
\operatorname{softmax}(t)_j
=\frac{\exp(t_j)}{\sum_{k=1}^{C}\exp(t_k)},
\qquad j=1,\ldots,C.\tag{10}
```

对有限得分，各分量均为正且总和为 1，相应的类别概率为

```math
\begin{aligned}
\hat y^{(i)}_j
&=\hat P(y^{(i)}=j\mid x^{(i)};\theta)\\[4pt]
&=\operatorname{softmax}\!\left(
f_1(x^{(i)};\theta_1),\ldots,f_C(x^{(i)};\theta_C)\right)_j\\[4pt]
&=\frac{\exp(f_j(x^{(i)};\theta_j))}
{\sum_{k=1}^{C}\exp(f_k(x^{(i)};\theta_k))}.
\end{aligned}\tag{11}
```

沿用最大似然原则，负对数似然可写为

```math
\begin{aligned}
J(\theta)=-\ell(\theta)
&=-\log\prod_{i=1}^{m}\hat P(y^{(i)}\mid x^{(i)};\theta)
=-\sum_{i=1}^{m}\log\hat y^{(i)}_{y^{(i)}}\\[4pt]
&=-\sum_{i=1}^{m}\log
\frac{\exp(f_{y^{(i)}}(x^{(i)};\theta_{y^{(i)}}))}
{\sum_{j=1}^{C}\exp(f_j(x^{(i)};\theta_j))}.
\end{aligned}\tag{12}
```

在神经网络中，分类头通常为输出维度等于类别数 $`C`$ 的线性层，其输出称为 logits；对 logits 应用 softmax，即得到预测类别概率。式 (12) 对应多分类交叉熵损失。PyTorch 的 `CrossEntropyLoss` 直接接收 logits，也就是式中的 $`f_j`$。

# Entropy and KL Divergence

离散随机变量 $`X`$ 的分布记为 $`P_X`$，其 Shannon 熵定义为

```math
H(P_X)=-\sum_{x\in\mathcal X}P_X(x)\log P_X(x).
\tag{13}
```

自信息 $`-\log P_X(x)`$ 随概率降低而增大，对独立事件具有可加性；其在 $`P_X`$ 下的期望即为熵，度量分布的不确定性。以 $`Q_X`$ 为模型分布，在真实分布 $`P_X`$ 下对 $`-\log Q_X(x)`$ 取期望，则得到交叉熵：

```math
H(P_X,Q_X)=-\sum_{x\in\mathcal X}P_X(x)\log Q_X(x).
\tag{14}
```

交叉熵通常不对称，其与目标分布自身的熵之差为 KL 散度：

```math
\begin{aligned}
D_{\mathrm{KL}}(P\Vert Q)
&=H(P,Q)-H(P)\\[4pt]
&=\sum_{x\in\mathcal X}P(x)\log\frac{P(x)}{Q(x)}.
\end{aligned}\tag{15}
```

KL 散度衡量以 $`Q`$ 代替 $`P`$ 所增加的期望信息量，非负且仅在两分布相同时为零。对于 softmax 分类模型，令真实条件分布为 $`P_{Y\mid X}`$、模型分布为 $`\hat P_{Y\mid X}`$，固定输入 $`x`$ 后有

```math
\begin{aligned}
D_{\mathrm{KL}}\!\left(P_{Y\mid X=x}\Vert\hat P_{Y\mid X=x};\theta\right)
&=\sum_{j=1}^{C}P(y=j\mid x)
\log\frac{P(y=j\mid x)}{\hat P(y=j\mid x;\theta)}.
\end{aligned}\tag{16}
```

若只对条件分布建模，并保持输入边缘分布 $`P_X`$ 不变，则 $`\hat P_{X,Y}(x,y;\theta)=P_X(x)\hat P_{Y\mid X}(y\mid x;\theta)`$。由联合分布计算 KL 散度时，概率比值中的 $`P_X(x)`$ 因子相消：

```math
\begin{aligned}
D_{\mathrm{KL}}(P_{X,Y}\Vert\hat P_{X,Y};\theta)
&=\sum_{x,y}P_{X,Y}(x,y)
\log\frac{P_{X,Y}(x,y)}{\hat P_{X,Y}(x,y;\theta)}\\[4pt]
&=\sum_{x,y}P_{X,Y}(x,y)
\log\frac{P(y\mid x)\cancel{P_X(x)}}
{\hat P(y\mid x;\theta)\cancel{P_X(x)}}\\[4pt]
&=\mathbb E_{X,Y}\!\left[
\log\frac{P(Y\mid X)}{\hat P(Y\mid X;\theta)}\right].
\end{aligned}\tag{17}
```

给定 $`n`$ 个独立同分布样本 $`(x^{(i)},y^{(i)})`$，在相应期望存在时，大数定律给出固定 $`\theta`$ 下的收敛关系：

```math
\begin{aligned}
\frac{1}{n}\sum_{i=1}^{n}
\log\frac{P(y^{(i)}\mid x^{(i)})}
{\hat P(y^{(i)}\mid x^{(i)};\theta)}
&\xrightarrow[n\to\infty]{}
\mathbb E_{X,Y}\!\left[
\log\frac{P(Y\mid X)}{\hat P(Y\mid X;\theta)}\right]\\[4pt]
&=D_{\mathrm{KL}}(P_{X,Y}\Vert\hat P_{X,Y};\theta).
\end{aligned}\tag{18}
```

式 (18) 给出了总体 KL 目标的样本近似。若进一步满足最优解唯一、经验目标一致收敛及极小值稳定性等正则条件，经验最优参数可收敛到总体最优参数：

```math
\begin{aligned}
&\underset{\theta}{\operatorname{argmin}}
\left\{\frac{1}{n}\sum_{i=1}^{n}
\log\frac{P(y^{(i)}\mid x^{(i)})}
{\hat P(y^{(i)}\mid x^{(i)};\theta)}\right\}\\[4pt]
&\qquad\xrightarrow[n\to\infty]{}
\underset{\theta}{\operatorname{argmin}}
D_{\mathrm{KL}}(P_{X,Y}\Vert\hat P_{X,Y};\theta).
\end{aligned}\tag{19}
```

最优解收敛需要上述一致性条件；对任意固定数据集，真实分布项与 $`\theta`$ 无关，经验目标则可直接化为最大似然估计：

```math
\begin{aligned}
&\underset{\theta}{\operatorname{argmin}}
\left\{\frac{1}{n}\sum_{i=1}^{n}
\log\frac{P(y^{(i)}\mid x^{(i)})}
{\hat P(y^{(i)}\mid x^{(i)};\theta)}\right\}\\[4pt]
&\quad=\underset{\theta}{\operatorname{argmax}}
\left\{\frac{1}{n}\sum_{i=1}^{n}
\log\hat P(y^{(i)}\mid x^{(i)};\theta)\right\}\\[4pt]
&\quad=\underset{\theta}{\operatorname{argmax}}
\left\{\frac{1}{n}\ell(\theta)\right\}.
\end{aligned}\tag{20}
```

因此，信息论中的分布拟合与最大似然估计对应相同的经验优化问题。由于真实分布的熵不依赖 $`\theta`$，最小化 KL 等价于最小化交叉熵：

```math
\begin{aligned}
\underset{\theta}{\operatorname{argmin}}
D_{\mathrm{KL}}(P_{X,Y}\Vert\hat P_{X,Y};\theta)
&=\underset{\theta}{\operatorname{argmin}}
\left\{H(P_{X,Y},\hat P_{X,Y})-H(P_{X,Y})\right\}\\[4pt]
&=\underset{\theta}{\operatorname{argmin}}
H(P_{X,Y},\hat P_{X,Y}).
\end{aligned}\tag{21}
```

# Knowledge Distillation

模型蒸馏以教师模型 $`M^{\mathrm{teacher}}`$ 的条件分布 $`P^{\mathrm{teacher}}(y\mid x)`$ 为监督目标，使通常规模更小的学生模型 $`M^{\mathrm{student}}`$ 学习其预测分布。常规监督学习仅观测标签 $`y^{(i)}`$，蒸馏则为每个输入提供完整的目标概率向量。

标签平滑同样采用概率标签，例如将真实类别的概率设为 $`\varepsilon\%`$，向其余 $`k-1`$ 类各分配 $`\frac{100-\varepsilon}{k-1}\%`$。对这类目标，沿用 KL 最小化与交叉熵的等价关系，将联合分布分解为输入边缘分布和标签条件分布：

```math
\begin{aligned}
H(P_{X,Y},\hat P_{X,Y})
&=-\sum_{x,y}P_{X,Y}(x,y)\log\hat P_{X,Y}(x,y;\theta)\\[4pt]
&=-\sum_x P_X(x)\sum_y P_{Y\mid X}(y\mid x)
\log\!\left[\hat P_{Y\mid X}(y\mid x;\theta)P_X(x)\right]\\[4pt]
&=H(P_X)-\sum_x P_X(x)\sum_y P_{Y\mid X}(y\mid x)
\log\hat P_{Y\mid X}(y\mid x;\theta)\\[4pt]
&=H(P_X)-\mathbb E_X\!\left[
\sum_y P_{Y\mid X}(y\mid X)
\log\hat P_{Y\mid X}(y\mid X;\theta)\right].
\end{aligned}\tag{22}
```

模型仅拟合条件分布、保持输入边缘分布不变，因此 $`H(P_X)`$ 与 $`\theta`$ 无关，可从优化目标中略去。每个输入的监督目标为完整条件分布，只需估计对 $`X`$ 的期望；当输入样本独立同分布且期望存在时，对固定 $`\theta`$ 有

```math
\begin{aligned}
R_n(\theta)
&=-\frac{1}{n}\sum_{i=1}^{n}\sum_{y\in\mathcal Y}
P_{Y\mid X}(y\mid x^{(i)})
\log\hat P_{Y\mid X}(y\mid x^{(i)};\theta)\\[4pt]
&\xrightarrow[n\to\infty]{}
-\mathbb E_X\!\left[\sum_{y\in\mathcal Y}
P_{Y\mid X}(y\mid X)
\log\hat P_{Y\mid X}(y\mid X;\theta)\right].
\end{aligned}\tag{23}
```

在式 (19) 所述的一致性条件下，最小化经验目标可用于估计总体目标的最优参数。对于 softmax 模型，预测条件概率为

```math
\hat P_{Y\mid X}(y\mid x;\theta)
=\frac{\exp(f_y(x;\theta_y))}
{\sum_{j=1}^{C}\exp(f_j(x;\theta_j))}.
\tag{24}
```

代入经验目标，得到以概率分布为标签的交叉熵损失：

```math
\begin{aligned}
J(\theta)
&=-\frac{1}{n}\sum_{i=1}^{n}\sum_{y\in\mathcal Y}
P_{Y\mid X}(y\mid x^{(i)})
\log\hat P_{Y\mid X}(y\mid x^{(i)};\theta)\\[4pt]
&=-\frac{1}{n}\sum_{i=1}^{n}\sum_{y\in\mathcal Y}
P_{Y\mid X}(y\mid x^{(i)})
\log\frac{\exp(f_y(x^{(i)};\theta_y))}
{\sum_{j=1}^{C}\exp(f_j(x^{(i)};\theta_j))}.
\end{aligned}\tag{25}
```

式 (25) 对应 PyTorch 在类别权重均为 1 时的平均交叉熵。[`CrossEntropyLoss`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html) 支持类别索引与概率标签，后者适用于混合标签、标签平滑等场景。沿用其记号，令 $`x_{n,c}`$ 为 logit，$`y_{n,c}`$ 为目标概率，$`w_c`$ 为类别权重，$`C`$ 为类别数；$`N`$ 为批量及额外位置维度上的位置总数。未归约损失为

```math
\ell(x,y)=L=(\ell_1,\ldots,\ell_N)^{\mathsf T},
\qquad
\ell_n=-\sum_{c=1}^{C}w_c y_{n,c}
\log\frac{\exp(x_{n,c})}{\sum_{i=1}^{C}\exp(x_{n,i})}.
\tag{26}
```

对于概率标签，`mean` 与 `sum` 分别对应

```math
\ell(x,y)=\begin{cases}
\dfrac{1}{N}\displaystyle\sum_{n=1}^{N}\ell_n,
&\text{reduction}=\texttt{mean},\\[8pt]
\displaystyle\sum_{n=1}^{N}\ell_n,
&\text{reduction}=\texttt{sum}.
\end{cases}\tag{27}
```

# Log-Probability Computation

设训练序列为 $`x=(x_0,\ldots,x_T)`$，$`x_0`$ 为起始 token。自回归预训练对后续所有 token 计算负对数似然；仅监督回答的 SFT 将损失限制在回答位置。若 $`x_{1:M}`$ 为提示词，$`x_{M+1:T}`$ 为参考回答，则

```math
\begin{aligned}
\mathcal L_{\mathrm{PT}}(\theta)
&=-\frac{1}{T}\sum_{t=0}^{T-1}\ln P_\theta(x_{t+1}\mid x_{\le t}),\\[4pt]
\mathcal L_{\mathrm{SFT}}(\theta)
&=-\frac{1}{T-M}\sum_{t=M}^{T-1}\ln P_\theta(x_{t+1}\mid x_{\le t}).
\end{aligned}\tag{28}
```

语言模型在每个位置执行一次词表分类：输入为前缀 $`x_{\le t}`$，观测类别为下一 token $`x_{t+1}`$。记词表大小为 $`V`$、当前位置的 logits 为 $`z_t\in\mathbb R^V`$，该观测标签可表示为 one-hot 向量 $`q_{t,j}=\mathbf 1\{j=x_{t+1}\}`$。它表示当前样本的监督目标，完整的真实条件分布并不要求为 one-hot。将这一标签代入交叉熵，词表求和只保留目标 token 对应的一项：

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

式 (30) 先将概率比值的对数展开为两项之差，再从指数和中提出 $`\exp(m_t)`$。移位后的指数输入均不大于零，可避免指数上溢；计算结果取负即为式 (29) 的逐 token 损失。

<strong>教师强制</strong>（teacher forcing）以真实 token 序列作为 `input_ids`，经 embedding 和带因果掩码的 Transformer 得到隐藏状态。位置 $`t`$ 仅依赖真实前缀 $`x_{\le t}`$，以 $`x_{t+1}`$ 为监督目标，因此整段序列的条件概率可在一次前向中并行计算。预训练与 SFT 均使用真实前缀构造这些预测。

Transformers [`LlamaForCausalLM.forward`（v4.44.2）](https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/llama/modeling_llama.py#L1208) 接收与 `input_ids` 原始位置对应的 `labels`，在损失分支中将 logits 与标签错开一位：

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

`labels[..., 1:]` 提供式 (29) 中 one-hot 向量非零项的类别索引，`logits[..., :-1, :]` 提供对应前缀的预测。标签错位来自输入前缀与下一 token 的位置关系；实际计算只需整数标签，无需显式构造 one-hot 张量。`CrossEntropyLoss` 忽略标记为 `-100` 的位置，并对其余标签计算负对数似然。

**Megatron Core**：[词表并行交叉熵](https://github.com/NVIDIA/Megatron-LM/blob/37cb4f261f103a7c882b5e310082fd0b7dbb226f/megatron/core/tensor_parallel/cross_entropy.py#L159)将式 (30) 的计算分布到各个 tensor parallel rank。MAX 归约得到全词表最大值，SUM 归约合并目标 logit 与指数和。核心前向取 `label_smoothing=0`：

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

`target` 为全局词表 ID，只有持有该 token 的 rank 提供目标值。`target_z` 已减去全局最大值 $`m_t`$，最终减去 `sum_exp.log()` 得到 log-prob，即逐 token 交叉熵的相反数。

**vLLM**：[指定 token 的 kernel](https://github.com/vllm-project/vllm/blob/dfebb42d096d5c6a04aa5a1722195e37cb8e0b2d/vllm/v1/worker/gpu/sample/logprob.py#L17)在同一个 kernel 中完成全词表归约和目标值提取。`row_ptr` 指向当前行，`topk_ids` 为请求评分的词表 ID：

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

两次遍历分别计算整行最大值与 FP32 指数和。`target_z` 为原始 logit，因此同时减去 `max_val` 和 `log(sum_exp)`；分母仍覆盖完整词表。

**Liger-Kernel**：[`_selective_logprob_forward`](https://github.com/linkedin/Liger-Kernel/blob/4dce24bbb70d8bda18e373b5f0f469a51ab77a2c/src/liger_kernel/chunked_loss/fused_linear_ppo.py#L18)逐块计算词表投影，并在线更新最大值和指数和。核心计算省略外层位置分块，取 `bias=None`：

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

最大值由 `m` 更新为 `m_new` 时，旧累积和乘以 `exp(m - m_new)`，再与新块的指数和相加。目标 token 所在块提供 `target_z`；遍历结束后的 `log_z` 为全词表 log-sum-exp，并保存供反向重计算使用。

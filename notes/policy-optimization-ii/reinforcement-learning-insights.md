## ☀️ 𝐈𝐧𝐬𝐢𝐠𝐡𝐭𝐬 𝐨𝐟 𝐑𝐞𝐢𝐧𝐟𝐨𝐫𝐜𝐞𝐦𝐞𝐧𝐭 𝐋𝐞𝐚𝐫𝐧𝐢𝐧𝐠

**统一梯度分析框架将 SFT、RFT、DPO、PPO 和 GRPO 等方法分解为数据来源、奖励函数与更新算法，并通过实验考察各组成部分的作用**

- 训练方法 $`\mathcal A`$ 关于参数 $`\theta`$ 的梯度可统一写为

```math
\nabla_\theta\mathcal J_{\mathcal A}(\theta)
=\mathbb E_{(q,o)\sim\mathcal D}\left[
\frac{1}{|o|}\sum_{t=1}^{|o|}
GC_{\mathcal A}(q,o,t,\pi_{\mathrm{rf}})
\nabla_\theta\log\pi_\theta(o_t\mid q,o_{<t})
\right]\tag{9}
```

- **数据来源** $`\mathcal D`$ 决定训练样本的分布；**奖励函数** $`\pi_{\mathrm{rf}}`$ 提供评价信号；**算法** $`\mathcal A`$ 将样本及评价信号转化为梯度系数 $`GC_{\mathcal A}`$，决定各样本在更新中的方向与权重
- **监督微调（SFT）**：使用人工筛选的示范数据对预训练模型进行微调
- **拒绝采样微调（RFT）**：针对 SFT 数据中的问题，从固定的 SFT 模型采样回答，依据答案正确性筛选样本，再使用保留的回答进行微调
- **直接偏好优化（DPO）**：在本节的离线设置中，从 SFT 模型采样回答并构造偏好对，通过成对的 DPO 目标进一步训练模型
- **在线拒绝采样微调（Online RFT）**：使用 SFT 模型初始化策略，在训练过程中持续从当前策略采样回答，并根据正确性筛选后进行更新
- **PPO / GRPO**：使用 SFT 模型初始化策略，周期性采样当前策略的回答，结合奖励信号与优势估计进行强化学习更新

**关于数据来源的观察**

- 本节将采样方式区分为离线与在线两类：离线方法使用初始 SFT 模型生成的固定数据，在线方法持续使用训练过程中策略模型的探索结果。因此，RFT 和本节的 DPO 属于离线方法，Online RFT、PPO 和 GRPO 属于在线方法
- 在 GSM8K 与 MATH 实验中，Online RFT 在训练初期与 RFT 表现接近，随后逐渐取得优势。训练初期的策略与初始 SFT 模型较为接近，两者生成的数据差异有限；随着策略更新，在线采样能够持续提供与当前策略分布相匹配的训练样本

**关于梯度系数的观察**

- 实验区分两类奖励信号：规则奖励根据答案正确性评价回答，模型奖励则由训练得到的奖励模型提供连续分数；奖励模型的训练标签同样依据规则判定结果构造
- Online RFT 对正确回答使用相同的筛选系数，对错误回答赋予零权重。GRPO 则根据组内相对奖励构造优势，使不同回答获得不同符号与幅度的梯度系数，从而同时实现强化与抑制

**强化学习为何有效**

- 在 DeepSeekMath 的 GSM8K 与 MATH 实验中，强化学习提高了多数投票正确率 Maj@K，但未提高衡量 $`K`$ 次采样中是否至少包含一个正确回答的 Pass@K。这一结果支持以下解释：改进主要表现为**提高已有正确推理路径的生成概率，使输出分布更稳定**
- 类似地，Wang et al. 将 SFT 模型在推理任务中的部分不足归因于偏好对齐，并观察到后续对齐训练能够改善推理表现

**各方法的目标与梯度**

- 记 $`h_t=(q,o_{<t})`$ 为生成第 $`t`$ 个 token 时的上下文，$`\theta_0`$ 为当前批次的采样策略参数；求导时固定采样数据、参考策略、奖励模型及优势估计

<details class="lake-derivation" id="insights-sft" open>
<summary>SFT · 监督微调</summary>
<div class="lake-derivation-body">

- SFT 直接最大化示范回答的平均对数似然，其中 $`P_{\mathrm{sft}}`$ 为示范数据的联合分布

```math
\mathcal J_{\mathrm{SFT}}(\theta)
=\mathbb E_{(q,o)\sim P_{\mathrm{sft}}}\left[
\frac{1}{|o|}\sum_{t=1}^{|o|}\log\pi_\theta(o_t\mid h_t)
\right]\tag{10}
```

- 对参数求导可得

```math
\nabla_\theta\mathcal J_{\mathrm{SFT}}(\theta)
=\mathbb E_{(q,o)\sim P_{\mathrm{sft}}}\left[
\frac{1}{|o|}\sum_{t=1}^{|o|}\nabla_\theta\log\pi_\theta(o_t\mid h_t)
\right]\tag{11}
```

- **数据来源**为 SFT 示范数据，人工筛选体现了对样本质量的判断；无需额外的显式奖励模型，梯度系数 $`GC_{\mathrm{SFT}}=1`$

</div>
</details>

<details class="lake-derivation" id="insights-rft" open>
<summary>RFT · 拒绝采样微调</summary>
<div class="lake-derivation-body">

- 对 SFT 数据中的问题，从固定的 SFT 模型采样回答。记这一联合分布为 $`\mathcal D_{\mathrm{off}}(q,o)=P_{\mathrm{sft}}(q)\pi_{\mathrm{sft}}(o\mid q)`$，并用 $`\mathbb I(q,o)`$ 表示答案是否正确，目标为

```math
\mathcal J_{\mathrm{RFT}}(\theta)
=\mathbb E_{(q,o)\sim\mathcal D_{\mathrm{off}}}\left[
\frac{\mathbb I(q,o)}{|o|}\sum_{t=1}^{|o|}\log\pi_\theta(o_t\mid h_t)
\right]\tag{12}
```

- 由于采样分布与正确性标签在更新时固定，梯度为

```math
\nabla_\theta\mathcal J_{\mathrm{RFT}}(\theta)
=\mathbb E_{(q,o)\sim\mathcal D_{\mathrm{off}}}\left[
\frac{\mathbb I(q,o)}{|o|}\sum_{t=1}^{|o|}\nabla_\theta\log\pi_\theta(o_t\mid h_t)
\right]\tag{13}
```

- **奖励信号**来自正确性规则，**梯度系数**为二值筛选权重

```math
GC_{\mathrm{RFT}}(q,o,t)=\mathbb I(q,o)=
\begin{cases}
1,&o\text{ 对问题 }q\text{ 的回答正确}\\
0,&o\text{ 对问题 }q\text{ 的回答错误}
\end{cases}\tag{14}
```

</div>
</details>

<details class="lake-derivation" id="insights-online-rft" open>
<summary>Online RFT · 在线拒绝采样微调</summary>
<div class="lake-derivation-body">

- Online RFT 将固定 SFT 模型替换为当前批次的采样策略。记 $`\mathcal D_0(q,o)=P_{\mathrm{sft}}(q)\pi_{\theta_0}(o\mid q)`$，在固定这批样本的条件下，更新方向为

```math
\left.\nabla_\theta\mathcal J_{\mathrm{OnRFT}}(\theta;\theta_0)\right|_{\theta=\theta_0}
=\mathbb E_{(q,o)\sim\mathcal D_0}\left[
\frac{\mathbb I(q,o)}{|o|}\sum_{t=1}^{|o|}
\left.\nabla_\theta\log\pi_\theta(o_t\mid h_t)\right|_{\theta=\theta_0}
\right]\tag{15}
```

- 正确性判定与梯度系数均与 RFT 相同；区别在于后续批次会使用更新后的策略重新采样，求导时不对本批次的离散采样过程反向传播

</div>
</details>

<details class="lake-derivation" id="insights-dpo" open>
<summary>DPO · 直接偏好优化</summary>
<div class="lake-derivation-body">

- 将采样回答按偏好标注整理为 $`(q,o^+,o^-)\sim\mathcal D_{\mathrm{pref}}`$，其中 $`o^+`$ 为偏好回答。按回答长度归一化，记平均对数概率比为 $`s_\theta(q,o)`$，并定义 $`\Delta_\theta=s_\theta(q,o^+)-s_\theta(q,o^-)`$

```math
\begin{aligned}
s_\theta(q,o)&=\frac{1}{|o|}\sum_{t=1}^{|o|}
\log\frac{\pi_\theta(o_t\mid h_t)}{\pi_{\mathrm{ref}}(o_t\mid h_t)}\\
\mathcal J_{\mathrm{DPO}}(\theta)
&=\mathbb E_{\mathcal D_{\mathrm{pref}}}\left[\log\sigma(\beta\Delta_\theta)\right]
\end{aligned}\tag{16}
```

- 对同一偏好对，两个回答共享系数 $`c_\theta`$，但在梯度中符号相反，其中 $`h_t^\pm=(q,o_{<t}^\pm)`$

```math
\nabla_\theta\mathcal J_{\mathrm{DPO}}(\theta)
=\mathbb E_{\mathcal D_{\mathrm{pref}}}\left[c_\theta\left(
\frac{1}{|o^+|}\sum_{t=1}^{|o^+|}\nabla_\theta\log\pi_\theta(o_t^+\mid h_t^+)
-\frac{1}{|o^-|}\sum_{t=1}^{|o^-|}\nabla_\theta\log\pi_\theta(o_t^-\mid h_t^-)
\right)\right]\tag{17}
```

- **奖励信号**体现在偏好标签中，可以来自人工判断或数学答案正确性规则。**梯度系数**取决于完整偏好对的分数差，而非单个 token 的局部对数比

```math
c_\theta=\beta\sigma(-\beta\Delta_\theta),\qquad
GC_{\mathrm{DPO}}^+=c_\theta,\qquad GC_{\mathrm{DPO}}^-=-c_\theta\tag{18}
```

- 若采用前文标准 DPO 的序列对数概率比约定，则去掉 $`s_\theta`$ 及对应梯度中的长度归一化因子，链式求导形式不变

</div>
</details>

<details class="lake-derivation" id="insights-ppo" open>
<summary>PPO · 近端策略优化</summary>
<div class="lake-derivation-body">

- 对当前批次，固定旧策略 $`\pi_{\theta_0}`$ 与优势 $`A_t`$，记概率比 $`\rho_t(\theta)=\pi_\theta(o_t\mid h_t)/\pi_{\theta_0}(o_t\mid h_t)`$，PPO 的裁剪目标为

```math
\mathcal J_{\mathrm{PPO}}(\theta;\theta_0)
=\mathbb E_{(q,o)\sim\mathcal D_0}\left[
\frac{1}{|o|}\sum_{t=1}^{|o|}
\min\left(\rho_t(\theta)A_t,
\operatorname{clip}(\rho_t(\theta),1-\varepsilon,1+\varepsilon)A_t\right)
\right]\tag{19}
```

- 在更新起点 $`\theta=\theta_0`$，概率比为 1，位于裁剪区间内部。因此，该点的梯度可以由以下未裁剪的代理目标计算

```math
\widetilde{\mathcal J}_{\mathrm{PPO}}(\theta;\theta_0)
=\mathbb E_{(q,o)\sim\mathcal D_0}\left[
\frac{1}{|o|}\sum_{t=1}^{|o|}\rho_t(\theta)A_t
\right]\tag{20}
```

- 在更新起点求导得到

```math
\left.\nabla_\theta\mathcal J_{\mathrm{PPO}}(\theta;\theta_0)\right|_{\theta=\theta_0}
=\mathbb E_{(q,o)\sim\mathcal D_0}\left[
\frac{1}{|o|}\sum_{t=1}^{|o|}A_t
\left.\nabla_\theta\log\pi_\theta(o_t\mid h_t)\right|_{\theta=\theta_0}
\right]\tag{21}
```

- 此时的**梯度系数**就是优势估计

```math
GC_{\mathrm{PPO}}(q,o,t)=A_t\tag{22}
```

- $`A_t`$ 由奖励与学习得到的价值函数通过 GAE 估计。上述简化描述的是更新起点的梯度；复用同一批样本进行多轮更新时，仍需考虑概率比与裁剪

</div>
</details>

<details class="lake-derivation" id="insights-grpo" open>
<summary>GRPO · 组相对策略优化</summary>
<div class="lake-derivation-body">

- 对每个问题采样 $`G`$ 条回答，记联合采样分布为 $`\mathcal D_{G,0}`$，上下文为 $`h_{i,t}=(q,o_{i,<t})`$。沿用 PPO 的概率比定义 $`\rho_{i,t}(\theta)`$，并记 $`u_{i,t}(\theta)=\pi_{\mathrm{ref}}(o_{i,t}\mid h_{i,t})/\pi_\theta(o_{i,t}\mid h_{i,t})`$
- 固定组内优势 $`\hat A_{i,t}`$，在更新起点附近，GRPO 的未裁剪代理目标可写为

```math
\widetilde{\mathcal J}_{\mathrm{GRPO}}(\theta;\theta_0)
=\mathbb E_{\mathcal D_{G,0}}\left[
\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}
\left\{\rho_{i,t}(\theta)\hat A_{i,t}
-\beta\left[u_{i,t}(\theta)-\log u_{i,t}(\theta)-1\right]\right\}
\right]\tag{23}
```

- 对 KL 估计项求导，得到 $`-\beta\nabla_\theta(u-\log u-1)=\beta(u-1)\nabla_\theta\log\pi_\theta`$。因此，在更新起点的总梯度为

```math
\left.\nabla_\theta\widetilde{\mathcal J}_{\mathrm{GRPO}}(\theta;\theta_0)\right|_{\theta=\theta_0}
=\mathbb E_{\mathcal D_{G,0}}\Bigg[
\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}
\left\{\hat A_{i,t}+\beta\left[u_{i,t}(\theta_0)-1\right]\right\}
\left.\nabla_\theta\log\pi_\theta(o_{i,t}\mid h_{i,t})\right|_{\theta=\theta_0}
\Bigg]
\tag{24}
```

- **奖励信号**用于构造组内相对优势，**梯度系数**还包含参考策略约束带来的修正

```math
GC_{\mathrm{GRPO}}(q,\{o_j\}_{j=1}^{G},i,t)
=\hat A_{i,t}+\beta\left[u_{i,t}(\theta_0)-1\right]\tag{25}
```

</div>
</details>

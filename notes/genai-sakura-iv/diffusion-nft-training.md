**在线强化学习：** 考虑预训练扩散策略 $`\pi^{\mathrm{old}}`$ 与提示数据集 $`\{c\}`$。每次迭代针对提示 $`c`$ 生成 $`K`$ 张图像 $`x_0^{1:K}`$，并以标量奖励 $`r\in[0,1]`$ 评估各样本。将奖励解释为最优性概率 $`r(x_0,c):=p(o=1\mid x_0,c)`$，便可以把生成数据随机划分为两个假想子集：图像以概率 $`r`$ 进入正样本集 $`\mathcal D^+`$，否则进入负样本集 $`\mathcal D^-`$。在样本数量无限的情况下，两类数据对应的分布分别为

```math
\begin{aligned}
\pi^+(x_0\mid c)
&:=\pi^{\mathrm{old}}(x_0\mid o=1,c)
=\frac{p(o=1\mid x_0,c)\pi^{\mathrm{old}}(x_0\mid c)}{p_{\pi^{\mathrm{old}}}(o=1\mid c)}
=\frac{r(x_0,c)}{p_{\pi^{\mathrm{old}}}(o=1\mid c)}\pi^{\mathrm{old}}(x_0\mid c),\\[5pt]
\pi^-(x_0\mid c)
&:=\pi^{\mathrm{old}}(x_0\mid o=0,c)
=\frac{p(o=0\mid x_0,c)\pi^{\mathrm{old}}(x_0\mid c)}{p_{\pi^{\mathrm{old}}}(o=0\mid c)}
=\frac{1-r(x_0,c)}{1-p_{\pi^{\mathrm{old}}}(o=1\mid c)}\pi^{\mathrm{old}}(x_0\mid c).
\end{aligned}\tag{3}
```

- 策略改进要求优化后的 $`\pi^*`$ 具有更高的期望奖励，即

```math
\mathbb E_{\pi^*(\cdot\mid c)}r(x_0,c)
>\mathbb E_{\pi^{\mathrm{old}}(\cdot\mid c)}r(x_0,c).\tag{4}
```

- **基于正样本数据的策略改进：** 当正、负分布均有定义，且奖励在旧策略下具有非零方差时，按期望奖励比较，有 $`\pi^+\succ\pi^{\mathrm{old}}\succ\pi^-`$。因此，一种直接方式是令 $`\pi^*=\pi^+`$，仅在 $`\mathcal D^+`$ 上微调扩散模型，即拒绝采样微调（RFT）。这种方式简单，但没有利用 $`\mathcal D^-`$ 中的负样本信息

**强化引导：** 同时利用正、负样本，可以构造相对于旧策略的改进方向 $`\Delta\in\mathbb R^n`$，并将目标速度场定义为

```math
v^*(x_t,c,t):=v^{\mathrm{old}}(x_t,c,t)+\frac1\beta\Delta(x_t,c,t).\tag{5}
```

- 其中，$`v^{\mathrm{old}}`$ 为旧策略的速度预测器，$`\Delta(x_t,c,t)`$ 称为强化引导，$`1/\beta`$ 控制引导强度。下面通过正、负策略与旧策略之间的关系确定这一方向

<div class="nft-theorem">

**定理 3.1（改进方向）。** 考虑分别对应 $`\pi^+`$、$`\pi^-`$ 和 $`\pi^{\mathrm{old}}`$ 的扩散模型 $`v^+`$、$`v^-`$ 和 $`v^{\mathrm{old}}`$，它们之间的方向差异成比例

```math
\begin{aligned}
\Delta
&:=[1-\alpha(x_t)]\,[v^{\mathrm{old}}(x_t,c,t)-v^-(x_t,c,t)]
=\alpha(x_t)\,[v^+(x_t,c,t)-v^{\mathrm{old}}(x_t,c,t)],\\[5pt]
\text{where}\quad\alpha(x_t)
&:=\frac{\pi_t^+(x_t\mid c)}{\pi_t^{\mathrm{old}}(x_t\mid c)}
\mathbb E_{\pi^{\mathrm{old}}(x_0\mid c)}r(x_0,c),\qquad 0\leq\alpha(x_t)\leq1.
\end{aligned}\tag{6}
```

</div>

- 式（6）给出相对于 $`v^{\mathrm{old}}`$ 的理想改进方向。例如，在式（5）中令 $`\beta=\alpha(x_t)>0`$，便有

```math
v^*(x_t,c,t)=v^{\mathrm{old}}(x_t,c,t)+\frac1{\alpha(x_t)}\Delta(x_t,c,t)=v^+(x_t,c,t).\tag{7}
```

- 此时恢复正样本策略 $`\pi^+=\pi^*`$。接下来通过隐式参数化，将正、负两项监督信号用于同一个可训练模型 $`v_\theta`$，直接学习沿强化引导方向改进的策略

<div class="nft-theorem">

**定理 3.2（策略优化）。** 考虑以下训练目标

```math
\mathcal L(\theta)=\mathbb E_{c,\,\pi^{\mathrm{old}}(x_0\mid c),\,t}
\left[r\left\|v_\theta^+(x_t,c,t)-v\right\|_2^2
+(1-r)\left\|v_\theta^-(x_t,c,t)-v\right\|_2^2\right].\tag{8}
```

其中，省略共同输入 $`(x_t,c,t)`$，隐式正、负策略分别为

```math
v_\theta^+:=(1-\beta)v^{\mathrm{old}}+\beta v_\theta,\qquad
v_\theta^-:=(1+\beta)v^{\mathrm{old}}-\beta v_\theta.\tag{9}
```

在数据量和模型容量不受限制的条件下，式（8）的最优解满足

```math
v_{\theta^*}(x_t,c,t)=v^{\mathrm{old}}(x_t,c,t)+\frac2\beta\Delta(x_t,c,t).\tag{10}
```

</div>

**DiffusionNFT 通过监督学习目标利用在线正、负样本进行策略优化。** 正、负预测器均由旧模型与同一个在线模型组合得到，训练时固定旧模型，仅更新 $`\theta`$。这一形式称为扩散负样本感知微调（Diffusion Negative-aware FineTuning），能够直接结合标准扩散训练

![DiffusionNFT 的隐式正负策略与奖励加权训练目标](assets/images/1791185148314-345cef10-fd19-4a9e-859d-7db2bfb572f6.png)

- **前向一致性：** 在前向过程中定义标准扩散损失，使 $`x_t`$ 与 $`x_0`$ 通过联合分布 $`\pi_\theta(x_t,x_0)=\pi_\theta(x_0)\pi_{t\mid0}(x_t\mid x_0)`$ 耦合，保留模型与前向扩散过程之间的联系
- **求解器灵活性：** 策略训练与数据采样解耦，收集数据时可以使用黑盒求解器；训练仅需干净图像及其奖励，无需保存完整的逆向采样轨迹
- **隐式引导整合：** 通过式（9）的参数化，将强化引导直接融入目标策略，无需单独学习引导模型 $`\Delta_\theta`$，也无需在采样时额外组合引导项
- **无需似然的优化：** 训练目标由前向回归损失构成，无需利用变分界近似数据似然，也无需从离散逆向轨迹构造序列似然

<h2 id="nft-proof-of-theorems" style="text-align: center">📐 Proof Of Theorems</h2>

**先由正、负分布的混合关系得到后验分解，再证明速度场的改进方向，最后将训练目标配方为平方误差。** 以下对应[原论文附录 A](https://proceedings.iclr.cc/paper_files/paper/2026/file/d8e68ddfe22520b45b8fb8d5cbde5a21-Paper-Conference.pdf#page=14)。固定提示 $`c`$，记 $`\rho_c:=p_{\pi^{\mathrm{old}}}(o=1\mid c)=\mathbb E_{\pi^{\mathrm{old}}(\cdot\mid c)}r(x_0,c)`$，并假设 $`0<\rho_c<1`$，使正、负分布均有定义

**引理 A.1（分布分解）。** 旧策略是正、负分布的混合

```math
\pi^{\mathrm{old}}(x_0\mid c)=\rho_c\pi^+(x_0\mid c)+(1-\rho_c)\pi^-(x_0\mid c).\tag{A.1}
```

- **证明：** 将式（3）代入右侧，两项分别为 $`r(x_0,c)\pi^{\mathrm{old}}(x_0\mid c)`$ 与 $`[1-r(x_0,c)]\pi^{\mathrm{old}}(x_0\mid c)`$，相加即得结论

**引理 A.2（后验分解）。** 三个分布经过相同的前向扰动核后，给定 $`x_t`$ 的干净样本后验满足

```math
\pi^{\mathrm{old}}(x_0\mid x_t,c)=\alpha(x_t)\pi^+(x_0\mid x_t,c)+[1-\alpha(x_t)]\pi^-(x_0\mid x_t,c).\tag{A.2}
```

- **证明：** 对式（A.1）两侧应用共同的前向核 $`\pi_{t\mid0}(x_t\mid x_0)`$，并对 $`x_0`$ 积分，得到受噪边缘分布的混合关系

```math
\pi_t^{\mathrm{old}}(x_t\mid c)=\rho_c\pi_t^+(x_t\mid c)+(1-\rho_c)\pi_t^-(x_t\mid c).\tag{A.3}
```

- 根据 Bayes 公式，奖励加权的旧策略后验可以改写为正、负后验。以下简记 $`r=r(x_0,c)`$、$`\alpha=\alpha(x_t)=\rho_c\pi_t^+(x_t\mid c)/\pi_t^{\mathrm{old}}(x_t\mid c)`$；由式（A.3），另一项的系数为 $`1-\alpha`$

```math
\begin{aligned}
r\,\pi^{\mathrm{old}}(x_0\mid x_t,c)
&=\frac{\rho_c\pi^+(x_0\mid c)\pi_{t\mid0}(x_t\mid x_0)}{\pi_t^{\mathrm{old}}(x_t\mid c)}
=\alpha\pi^+(x_0\mid x_t,c),\\[4pt]
(1-r)\,\pi^{\mathrm{old}}(x_0\mid x_t,c)
&=\frac{(1-\rho_c)\pi^-(x_0\mid c)\pi_{t\mid0}(x_t\mid x_0)}{\pi_t^{\mathrm{old}}(x_t\mid c)}
=(1-\alpha)\pi^-(x_0\mid x_t,c).
\end{aligned}\tag{A.4}
```

- 两式相加即得式（A.2）。后验混合系数 $`\alpha(x_t)`$ 与前向噪声调度 $`\alpha_t`$ 是不同的量

**定理 3.1 的证明（附录定理 A.3）。** 由 $`x_t=\alpha_t x_0+\sigma_t\epsilon`$ 与 $`v=\dot\alpha_t x_0+\dot\sigma_t\epsilon`$，消去噪声 $`\epsilon`$ 后，理想速度预测器可写成干净样本后验均值的仿射函数

```math
v^s(x_t,c,t)=a_t x_t+b_t\mathbb E_{\pi^s(x_0\mid x_t,c)}[x_0],
\qquad s\in\{\mathrm{old},+,-\}.\tag{A.5}
```

- 其中，$`a_t=\dot\sigma_t/\sigma_t`$、$`b_t=\dot\alpha_t-(\dot\sigma_t/\sigma_t)\alpha_t`$，上述表达式用于 $`\sigma_t>0`$ 的时间点。将式（A.2）代入后验期望，并省略速度场共同的输入 $`(x_t,c,t)`$，有

```math
v^{\mathrm{old}}=\alpha v^++(1-\alpha)v^-
\quad\Longrightarrow\quad
(1-\alpha)(v^{\mathrm{old}}-v^-)=\alpha(v^+-v^{\mathrm{old}})=\Delta.\tag{A.6}
```

- 这正是式（6）的强化引导关系，定理 3.1 得证

**定理 3.2 的证明（附录定理 A.4）。** 先对式（8）按照受噪样本 $`x_t`$ 取条件期望，再利用式（A.4），将奖励权重吸收到正、负后验中。随后使用平方误差的偏差—方差分解，以条件均值 $`v^+`$ 和 $`v^-`$ 替换随机目标速度

```math
\begin{aligned}
\mathcal L(\theta)
&=\mathbb E_{c,t,\,x_t\sim\pi_t^{\mathrm{old}}}
\left[\alpha\mathbb E_{\pi^+(x_0\mid x_t,c)}\|v_\theta^+-v\|_2^2
+(1-\alpha)\mathbb E_{\pi^-(x_0\mid x_t,c)}\|v_\theta^--v\|_2^2\right]\\
&=\mathbb E_{c,t,\,x_t\sim\pi_t^{\mathrm{old}}}
\left[\alpha\|v_\theta^+-v^+\|_2^2+(1-\alpha)\|v_\theta^--v^-\|_2^2\right]+C_1.
\end{aligned}\tag{A.7}
```

- $`C_1`$ 来自条件方差，与待优化参数 $`\theta`$ 无关。令 $`d=v_\theta-v^{\mathrm{old}}`$，由式（9），两项残差分别为 $`\beta d-(v^+-v^{\mathrm{old}})`$ 和 $`-\beta d+(v^{\mathrm{old}}-v^-)`$。展开平方，并利用式（A.6）中的两项均等于 $`\Delta`$，得到

```math
\begin{aligned}
\mathcal L(\theta)
&=\beta^2\mathbb E\|d\|_2^2
-2\beta\mathbb E\!\left\langle d,\alpha(v^+-v^{\mathrm{old}})+(1-\alpha)(v^{\mathrm{old}}-v^-)\right\rangle+C_2\\
&=\beta^2\mathbb E\|d\|_2^2-4\beta\mathbb E\langle d,\Delta\rangle+C_2
=\beta^2\mathbb E\left\|d-\frac2\beta\Delta\right\|_2^2+C_3.
\end{aligned}\tag{A.8}
```

- 这里的期望均针对 $`c,t,x_t\sim\pi_t^{\mathrm{old}}(\cdot\mid c)`$，$`C_2,C_3`$ 也与 $`\theta`$ 无关。因此，在模型容量足够且达到最优时，$`d=2\Delta/\beta`$，即 $`v_{\theta^*}=v^{\mathrm{old}}+2\Delta/\beta`$，得到式（10），定理 3.2 得证

<h2 id="nft-practical-implementation" style="text-align: center">🛠️ Practical Implementation</h2>

**每轮先由旧策略生成图像并计算奖励，再对这些图像重新加噪、更新在线模型，最后软更新旧策略，进入下一轮。** 其中，奖励归一化决定正、负损失的权重，EMA 控制采样策略的更新速度，自适应加权调整回归损失的尺度

**最优性奖励：** 将取值不受限制的原始奖励 $`r^{\mathrm{raw}}`$ 减去同一提示下的均值，再归一化并截断，得到最优性概率 $`r\in[0,1]`$

```math
r(x_0,c):=\frac12+\frac12\operatorname{clip}\!\left[
\frac{r^{\mathrm{raw}}(x_0,c)-\mathbb E_{\pi^{\mathrm{old}}(\cdot\mid c)}r^{\mathrm{raw}}(x_0,c)}{Z_c},-1,1\right].\tag{11}
```

- $`Z_c>0`$ 为归一化因子，例如全局奖励标准差；式中的期望用同一提示下 $`K`$ 张图像的平均奖励估计。高于组均值的样本满足 $`r>1/2`$，正项权重更大；低于组均值的样本满足 $`r<1/2`$，负项权重更大。训练直接使用 $`r`$ 和 $`1-r`$ 加权，无需将样本硬划分为两组

**采样策略的软更新：** 当前轮次内固定旧模型 $`v^{\mathrm{old}}`$，用其收集样本，并按式（9）构造隐式正、负预测器；梯度仅更新在线模型 $`v_\theta`$。完成这一轮优化后，再通过指数移动平均（EMA）更新旧模型

```math
\theta^{\mathrm{old}}\leftarrow\eta_i\theta^{\mathrm{old}}+(1-\eta_i)\theta.\tag{12}
```

- 其中，$`i`$ 为迭代次数。$`\eta_i=0`$ 表示直接将在线模型复制为采样模型；$`\eta_i`$ 越接近 1，旧模型保留的比例越高，更新越缓慢。较快更新有利于及时利用策略改进，但可能降低稳定性。更新后清空当前数据缓冲区，下一轮使用新的采样策略重新生成图像

**自适应损失加权：** 将速度预测器转换为 $`x_0`$ 预测器 $`x_\theta`$。对于 Rectified Flow，$`x_\theta=x_t-tv_\theta`$；借鉴 DMD 的归一化方式，用自归一化的 $`x_0`$ 回归替代式（1）中手动选择的时间权重

```math
w(t)\left\|v_\theta(x_t,c,t)-v\right\|_2^2
\quad\longleftarrow\quad
\frac{\left\|x_\theta(x_t,c,t)-x_0\right\|_2^2}
{\operatorname{sg}\!\left(\operatorname{mean}\!\left(\operatorname{abs}\!\left(x_\theta(x_t,c,t)-x_0\right)\right)\right)}.\tag{13}
```

- 分子是干净样本预测的平方误差，分母是该误差向量各维度的平均绝对值。$`\operatorname{sg}`$ 表示停止梯度，使分母只参与数值缩放，不通过分母反向传播梯度。该归一化用于调整回归项的尺度，正、负两项之间的比例仍由 $`r`$ 与 $`1-r`$ 决定

下述算法汇总数据收集、策略优化与软更新的循环。梯度步骤采用式（8）的基础回归目标；使用自适应加权时，将正、负预测器各自的回归项替换为式（13）对应的形式

<figure class="nft-algorithm" aria-labelledby="nft-algorithm-1">
<figcaption id="nft-algorithm-1"><strong>Algorithm 1</strong> Diffusion Negative-Aware FineTuning</figcaption>

```math
\begin{array}{l}
\textbf{Input: }v^{\mathrm{ref}},\ r^{\mathrm{raw}},\ \{c\},\ K,\ \beta,\ \lambda,\ \{\eta_i\}.\\[-0.2em]
\theta^{\mathrm{old}}\leftarrow\theta^{\mathrm{ref}},\quad\theta\leftarrow\theta^{\mathrm{ref}},\quad\mathcal D\leftarrow\varnothing.\\[-0.2em]
\textbf{for }\text{each iteration }i\textbf{ do}\\[-0.2em]
\quad\textbf{for }\text{each sampled prompt }c\textbf{ do}\\[-0.2em]
\qquad x_0^{1:K}\sim\pi^{\mathrm{old}}(\cdot\mid c),\quad r_k^{\mathrm{raw}}\leftarrow r^{\mathrm{raw}}(x_0^{(k)},c).\\[-0.2em]
\qquad\bar r_c^{\mathrm{raw}}\leftarrow\frac1K\sum_{k=1}^K r_k^{\mathrm{raw}},\quad
r_k\leftarrow\frac12+\frac12\operatorname{clip}((r_k^{\mathrm{raw}}-\bar r_c^{\mathrm{raw}})/Z_c,-1,1).\\[-0.2em]
\qquad\mathcal D\leftarrow\mathcal D\cup\{(c,x_0^{(k)},r_k)\}_{k=1}^K.\\[-0.2em]
\quad\textbf{end for}\\[-0.2em]
\quad\textbf{for }\text{each mini-batch }(c,x_0,r)\in\mathcal D\textbf{ do}\\[-0.2em]
\qquad\text{Sample time }t\text{ and }\epsilon\sim\mathcal N(0,I);\quad x_t\leftarrow\alpha_t x_0+\sigma_t\epsilon,\quad v\leftarrow\dot\alpha_t x_0+\dot\sigma_t\epsilon.\\[-0.2em]
\qquad v_\theta^+\leftarrow(1-\beta)v^{\mathrm{old}}+\beta v_\theta,\quad v_\theta^-\leftarrow(1+\beta)v^{\mathrm{old}}-\beta v_\theta.\\[-0.2em]
\qquad\theta\leftarrow\theta-\lambda\nabla_\theta\operatorname{mean}_{\mathrm{batch}}
\left[r\|v_\theta^+-v\|_2^2+(1-r)\|v_\theta^--v\|_2^2\right].\\[-0.2em]
\quad\textbf{end for}\\[-0.2em]
\quad\theta^{\mathrm{old}}\leftarrow\eta_i\theta^{\mathrm{old}}+(1-\eta_i)\theta,\quad\mathcal D\leftarrow\varnothing.\\[-0.2em]
\textbf{end for}\\[-0.2em]
\textbf{return }v_\theta.
\end{array}
```

</figure>

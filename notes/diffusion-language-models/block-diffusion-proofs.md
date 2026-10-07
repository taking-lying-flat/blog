## 🧮 Proofs

**以下推导依次展开 Block Diffusion 的变分目标、掩码转移与连续时间极限，再说明自回归特例、注意力条件依赖和采样过程之间的关系。**

### Block Diffusion NELBO

**将长度为 $`L`$ 的序列划分为 $`B`$ 个块，记干净的历史上下文为 $`c_b=\mathbf{x}^{<b}`$。令 $`t_j=j/T`$、$`s_j=(j-1)/T`$，并用 $`\mathbf{z}^{b}=\mathbf{x}_{t_1:t_T}^{b}`$ 表示第 $`b`$ 个块的整条含噪路径**

```math
\begin{aligned}
q(\mathbf{z}^{b}\mid\mathbf{x}^{b})
&=\prod_{j=1}^{T}q(\mathbf{x}_{t_j}^{b}\mid\mathbf{x}_{s_j}^{b})\\
p_\theta(\mathbf{x}^{b},\mathbf{z}^{b}\mid c_b)
&=p_{\mathrm{prior}}(\mathbf{x}_{t_T}^{b})\prod_{j=1}^{T}p_\theta(\mathbf{x}_{s_j}^{b}\mid\mathbf{x}_{t_j}^{b},c_b)
\end{aligned}\tag{A.1}
```

<div class="lake-derivation" style="--derivation-border:#94adc8;--derivation-background:#fffcf5">
<p class="lake-derivation-title">Variational Bound From Path Distributions</p>
<div class="lake-derivation-body">

- 固定数据块和历史上下文，对前向路径分布取期望；在重要性比值有定义的支持集条件下，利用 $`-\log`$ 的凸性得到

```math
\begin{aligned}
-\log p_\theta(\mathbf{x})
&=-\sum_{b=1}^{B}\log\mathbb{E}_{q}\!\left[\frac{p_\theta(\mathbf{x}^{b},\mathbf{z}^{b}\mid c_b)}{q(\mathbf{z}^{b}\mid\mathbf{x}^{b})}\right]\\
&\leq\sum_{b=1}^{B}\mathbb{E}_{q}\!\left[\log\frac{q(\mathbf{z}^{b}\mid\mathbf{x}^{b})}{p_\theta(\mathbf{x}^{b},\mathbf{z}^{b}\mid c_b)}\right]
=\mathcal{L}_{\mathrm{BD}}(\mathbf{x};\theta)
\end{aligned}\tag{A.2}
```

- 将前向链按逆向后验重新分解，便可把每个块的损失拆成重建项、扩散项和先验项

```math
\begin{aligned}
\mathcal{L}_{\mathrm{BD}}&=\sum_{b=1}^{B}\bigl(\mathcal{L}_{\mathrm{recons}}^{b}+\mathcal{L}_{\mathrm{diffusion}}^{b}+\mathcal{L}_{\mathrm{prior}}^{b}\bigr)\\
\mathcal{L}_{\mathrm{recons}}^{b}&=-\mathbb{E}_{q}\log p_\theta(\mathbf{x}^{b}\mid\mathbf{x}_{t_1}^{b},c_b)\\
\mathcal{L}_{\mathrm{diffusion}}^{b}&=\sum_{j=2}^{T}\mathbb{E}_{q}D_{\mathrm{KL}}\!\left[q(\mathbf{x}_{s_j}^{b}\mid\mathbf{x}_{t_j}^{b},\mathbf{x}^{b})\,\middle\|\,p_\theta(\mathbf{x}_{s_j}^{b}\mid\mathbf{x}_{t_j}^{b},c_b)\right]\\
\mathcal{L}_{\mathrm{prior}}^{b}&=D_{\mathrm{KL}}\!\left[q(\mathbf{x}_{t_T}^{b}\mid\mathbf{x}^{b})\,\middle\|\,p_{\mathrm{prior}}(\mathbf{x}_{t_T}^{b})\right]
\end{aligned}\tag{A.3}
```

- 重建项已经包含第一步，因此扩散项从 $`j=2`$ 开始求和。有限步数时使用显式求和，可以避免把 $`T-1`$ 项的均匀期望误乘为 $`T`$

</div>
</div>

**上界的紧性**取决于真实前向路径后验与模型路径后验之间的差距

```math
\mathcal{L}_{\mathrm{BD}}(\mathbf{x};\theta)+\log p_\theta(\mathbf{x})
=\sum_{b=1}^{B}D_{\mathrm{KL}}\!\left[q(\mathbf{z}^{b}\mid\mathbf{x}^{b})\,\middle\|\,p_\theta(\mathbf{z}^{b}\mid\mathbf{x}^{b},c_b)\right]\geq0\tag{A.4}
```

- 对固定的模型和分块，上界取等当且仅当各块的两个路径后验几乎处处相同；仅凭这个关系，不能推出不同块大小、分别训练的模型之间存在统一的损失单调关系

### Masked Forward And Reverse Processes

**采用独热列向量表示 token ，$`\mathbf{m}`$ 为掩码 token ，$`\mathbf{1}`$ 为全一列向量。$`\alpha_t`$ 表示 token 保持未被掩码的概率，满足 $`\alpha_0=1`$、$`\alpha_1=0`$。区分累计转移矩阵 $`\overline{Q}_t`$ 与从 $`s`$ 到 $`t`$ 的转移矩阵 $`Q_{t\mid s}`$**

```math
\overline{Q}_t=\alpha_t I+(1-\alpha_t)\mathbf{m}\mathbf{1}^{\top},\qquad
Q_{t\mid s}=\frac{\alpha_t}{\alpha_s}I+\left(1-\frac{\alpha_t}{\alpha_s}\right)\mathbf{m}\mathbf{1}^{\top}
\quad(0\leq s<t\leq1,\ \alpha_s>0)\tag{A.5}
```

- 矩阵的列对应出发状态、行对应到达状态，因此非掩码 token 到掩码 token 的概率位于第 $`m`$ 行。掩码是吸收态，已经被掩码的 token 保持不变
- 每个 token 独立加噪。由 $`(\mathbf{m}\mathbf{1}^{\top})^2=\mathbf{m}\mathbf{1}^{\top}`$，逐步转移可以合并为累计转移

```math
\begin{aligned}
\overline{Q}_{t_j}&=Q_{t_j\mid t_{j-1}}\cdots Q_{t_1\mid0},\qquad Q_{t\mid s}\overline{Q}_s=\overline{Q}_t\\
q(\mathbf{x}_t^{\ell}\mid\mathbf{x}^{\ell})
&=\operatorname{Cat}\!\left(\mathbf{x}_t^{\ell};\overline{Q}_t\mathbf{x}^{\ell}\right)
=\operatorname{Cat}\!\left(\mathbf{x}_t^{\ell};\alpha_t\mathbf{x}^{\ell}+(1-\alpha_t)\mathbf{m}\right)
\end{aligned}\tag{A.6}
```

<div class="lake-derivation" style="--derivation-border:#98b9ac;--derivation-background:#fffcf5">
<p class="lake-derivation-title">Reverse Posterior Via Bayes Rule</p>
<div class="lake-derivation-body">

固定一个干净 token $`\mathbf{x}`$，对前向概率应用 Bayes 公式。$`\odot`$ 表示逐元素乘法

```math
q(\mathbf{x}_s\mid\mathbf{x}_t,\mathbf{x})
=\operatorname{Cat}\!\left(\mathbf{x}_s;
\frac{(Q_{t\mid s}^{\top}\mathbf{x}_t)\odot(\overline{Q}_s\mathbf{x})}
{\mathbf{x}_t^{\top}\overline{Q}_t\mathbf{x}}\right)\tag{A.7}
```

- $`Q_{t\mid s}^{\top}\mathbf{x}_t`$ 给出各候选前态到当前观测的似然，$`\overline{Q}_s\mathbf{x}`$ 给出前态的边缘概率；两者相乘后归一化
- 若当前 token 未被掩码，则前一时刻必然是同一个 token ；若当前 token 已被掩码，则前一时刻可能是干净 token ，也可能仍为掩码

```math
q(\mathbf{x}_s\mid\mathbf{x}_t,\mathbf{x})=
\begin{cases}
\delta_{\mathbf{x}_t}(\mathbf{x}_s),&\mathbf{x}_t\neq\mathbf{m},\\[3pt]
\operatorname{Cat}\!\left(\mathbf{x}_s;
\dfrac{\alpha_s-\alpha_t}{1-\alpha_t}\mathbf{x}
+\dfrac{1-\alpha_s}{1-\alpha_t}\mathbf{m}\right),&\mathbf{x}_t=\mathbf{m}
\end{cases}\tag{A.8}
```

</div>
</div>

生成时无法访问干净 token ，用去噪网络对候选干净 token 的预测进行混合，得到块内每个位置的逆向转移

```math
p_\theta(\mathbf{x}_s^{b,\ell}\mid\mathbf{x}_t^{b},c_b)
=\sum_{\mathbf{y}}q(\mathbf{x}_s^{b,\ell}\mid\mathbf{x}_t^{b,\ell},\mathbf{y})
\,p_\theta(\mathbf{y}\mid\mathbf{x}_t^{b},c_b)\tag{A.9}
```

- 去噪网络以整个含噪块 $`\mathbf{x}_t^{b}`$ 和历史上下文 $`c_b`$ 为条件；前向过程逐 token 独立，并不意味着网络只能看见当前位置

### From Discrete KL To Continuous NELBO

**SUBS 参数化对干净 token 预测施加两个约束：掩码类别的预测概率为零，未被掩码的 token 直接复制**

```math
p_\theta(\mathbf{y}=\mathbf{m}\mid\mathbf{x}_t^{b},c_b)=0,\qquad
p_\theta(\mathbf{y}=\mathbf{x}_t^{b,\ell}\mid\mathbf{x}_t^{b},c_b)=1
\quad\text{if }\mathbf{x}_t^{b,\ell}\neq\mathbf{m}\tag{A.10}
```

<div class="lake-derivation" style="--derivation-border:#c2ae89;--derivation-background:#fffcf5">
<p class="lake-derivation-title">Diffusion KL As Weighted Cross Entropy</p>
<div class="lake-derivation-body">

令 $`w_{s,t}=(\alpha_s-\alpha_t)/(1-\alpha_t)`$。对于已被掩码的位置，真实逆向分布与模型逆向分布具有相同的保留掩码概率 $`1-w_{s,t}`$；只有恢复干净 token 的分支产生 KL

```math
\begin{aligned}
D_{\mathrm{KL}}(q\|p_\theta)
&=w_{s,t}\log\frac{w_{s,t}}{w_{s,t}\,p_\theta(\mathbf{x}^{b,\ell}\mid\mathbf{x}_t^{b},c_b)}
+(1-w_{s,t})\log\frac{1-w_{s,t}}{1-w_{s,t}}\\
&=-w_{s,t}\log p_\theta(\mathbf{x}^{b,\ell}\mid\mathbf{x}_t^{b},c_b)
\end{aligned}\tag{A.11}
```

- 未被掩码的位置在两侧都是相同的点质量分布，KL 为零；由复制约束，对应的对数预测概率也为零。因此可以对整个块统一求和
- 块内干净 token 预测按位置因子化，$`\log p_\theta(\mathbf{x}^{b}\mid\mathbf{x}_t^{b},c_b)=\sum_{\ell=1}^{L'}\log p_\theta(\mathbf{x}^{b,\ell}\mid\mathbf{x}_t^{b},c_b)`$

```math
\mathcal{L}_{\mathrm{diffusion}}
=\sum_{b=1}^{B}\sum_{j=2}^{T}\mathbb{E}_{q}\!\left[
\frac{\alpha_{t_j}-\alpha_{s_j}}{1-\alpha_{t_j}}
\log p_\theta(\mathbf{x}^{b}\mid\mathbf{x}_{t_j}^{b},c_b)\right]\tag{A.12}
```

- 令 $`\Delta t=1/T`$，利用 $`\alpha_{t_j}-\alpha_{s_j}=\alpha'_{t_j}\Delta t+o(\Delta t)`$，在可积且极限可交换的条件下，离散求和收敛为连续积分

```math
\lim_{T\to\infty}\mathcal{L}_{\mathrm{diffusion}}
=\sum_{b=1}^{B}\int_0^1\mathbb{E}_{q}\!\left[
\frac{\alpha'_t}{1-\alpha_t}\log p_\theta(\mathbf{x}^{b}\mid\mathbf{x}_t^{b},c_b)\right]\,\mathrm{d}t\tag{A.13}
```

</div>
</div>

当重建交叉熵在 $`t\downarrow0`$ 附近满足相应可积条件时，重建项随掩码概率趋于零而消失；终点全部为掩码，先验项也为零

```math
\lim_{T\to\infty}\mathcal{L}_{\mathrm{recons}}^{b}=0,\qquad
\mathcal{L}_{\mathrm{prior}}^{b}
=D_{\mathrm{KL}}\!\left[\delta_{\mathbf{m}^{L'}}\,\middle\|\,\delta_{\mathbf{m}^{L'}}\right]=0\tag{A.14}
```

因此，完整的连续时间目标只剩下加权交叉熵

```math
\mathcal{L}_{\mathrm{BD}}(\mathbf{x};\theta)
=\sum_{b=1}^{B}\mathbb{E}_{t\sim\mathcal{U}(0,1)}\mathbb{E}_{q}\!\left[
\frac{\alpha'_t}{1-\alpha_t}\log p_\theta(\mathbf{x}^{b}\mid\mathbf{x}_t^{b},c_b)\right]\tag{A.15}
```

- $`\alpha'_t<0`$ 且对数概率不大于零，损失非负。采用 $`\alpha_t=1-t`$ 时，权重为 $`-1/t`$

**噪声调度不变性**可以由变量替换直接得到。令 $`u=1-\alpha_t`$，用 $`q_u`$ 表示掩码概率为 $`u`$ 的前向分布

```math
\mathcal{L}_{\mathrm{BD}}(\mathbf{x};\theta)
=-\sum_{b=1}^{B}\int_0^1\frac{1}{u}\,
\mathbb{E}_{q_u}\log p_\theta(\mathbf{x}^{b}\mid\mathbf{x}_u^{b},c_b)\,\mathrm{d}u\tag{A.16}
```

- 这里采用论文中不显式输入时间步的去噪网络；若网络接收时间条件，还需随噪声水平同步重参数化该条件。积分目标不变，不代表均匀采样时间步得到的 Monte Carlo 估计量方差不变

### Recovering Autoregressive Likelihood

**当 $`L'=1`$、$`B=L`$ 时，每个块只有一个 token 。取 $`\alpha_t=1-t`$，采用 SUBS 复制约束，并沿用论文中不显式输入时间步的网络**

<div class="lake-derivation" style="--derivation-border:#94adc8;--derivation-background:#fffcf5">
<p class="lake-derivation-title">Single Token NELBO And Autoregressive NLL</p>
<div class="lake-derivation-body">

- 展开掩码与未掩码两种状态；未掩码 token 直接复制，其对数概率为 $`\log1=0`$

```math
\begin{aligned}
\mathbb{E}_{q}\log p_\theta(\mathbf{x}^{b}\mid\mathbf{x}_t^{b},c_b)
&=t\log p_\theta(\mathbf{x}^{b}\mid\mathbf{m},c_b)
+(1-t)\log1\\
&=t\log p_\theta(\mathbf{x}^{b}\mid\mathbf{m},c_b)
\end{aligned}\tag{A.17}
```

- 将其代入连续时间目标，掩码概率 $`t`$ 与权重中的 $`1/t`$ 抵消。掩码输入及历史上下文固定，剩下的预测不再依赖时间步

```math
\begin{aligned}
\mathcal{L}_{\mathrm{BD}}(\mathbf{x};\theta)
&=-\sum_{b=1}^{L}\int_0^1\frac{1}{t}\,t\log p_\theta(\mathbf{x}^{b}\mid\mathbf{m},c_b)\,\mathrm{d}t\\
&=-\sum_{b=1}^{L}\log p_\theta(\mathbf{x}^{b}\mid\mathbf{m},\mathbf{x}^{<b})
=-\log p_\theta(\mathbf{x})
\end{aligned}\tag{A.18}
```

</div>
</div>

**期望相同仍可能有不同方差**。固定上下文和 $`t>0`$，记单 token 交叉熵为 $`\ell=-\log p_\theta(\mathbf{x}^{b}\mid\mathbf{m},c_b)`$，是否被掩码为 $`M\sim\operatorname{Bernoulli}(t)`$。下面的标量例子说明随机掩码如何引入额外方差

```math
\widehat{\ell}=\frac{M}{t}\ell,\qquad
\mathbb{E}_{M}[\widehat{\ell}]=\ell,\qquad
\operatorname{Var}_{M}(\widehat{\ell})=\frac{1-t}{t}\ell^2\tag{A.19}
```

- 对 $`\theta`$ 求导可得到同样的掩码重加权形式。因此，单 token 块恢复自回归目标，并不保证采用随机掩码训练时也具有相同的梯度方差

### Specialized Attention Masks

**将所有含噪块与干净序列拼接为 $`\mathbf{x}_{\mathrm{full}}=\mathbf{x}_{\mathrm{noisy}}\oplus\mathbf{x}`$。总长度为 $`2L`$，注意力掩码由四个 $`L\times L`$ 子矩阵构成；行表示 query，列表示 key，$`1`$ 表示允许注意**

```math
M_{\mathrm{full}}=
\begin{bmatrix}
M_{\mathrm{BD}}&M_{\mathrm{OBC}}\\
0&M_{\mathrm{BC}}
\end{bmatrix}\tag{A.20}
```

以 $`g(i)=\lfloor(i-1)/L'\rfloor`$ 表示原序列中第 $`i`$ 个位置所属的块，三个非零子矩阵可以统一写成指示函数

```math
\begin{aligned}
[M_{\mathrm{BD}}]_{ij}&=\mathbf{1}\{g(j)=g(i)\}\\
[M_{\mathrm{OBC}}]_{ij}&=\mathbf{1}\{g(j)<g(i)\}\\
[M_{\mathrm{BC}}]_{ij}&=\mathbf{1}\{g(j)\leq g(i)\}
\end{aligned}\tag{A.21}
```

<div class="lake-derivation" style="--derivation-border:#98b9ac;--derivation-background:#fffcf5">
<p class="lake-derivation-title">Conditional Dependencies In One Forward Pass</p>
<div class="lake-derivation-body">

- **含噪部分**：通过 $`M_{\mathrm{BD}}`$ 读取当前含噪块，通过 $`M_{\mathrm{OBC}}`$ 读取严格更早的干净块，恰好对应去噪条件 $`(\mathbf{x}_t^{b},\mathbf{x}^{<b})`$
- **干净部分**：通过 $`M_{\mathrm{BC}}`$ 读取当前及更早的干净块，计算历史块的键和值；左下角为零，保证干净上下文不受含噪分支影响
- 右上角必须严格使用 $`g(j)<g(i)`$，否则含噪 token 可以看到当前块的干净答案，造成信息泄漏

</div>
</div>

块结构可以直接用于稀疏注意力实现，跳过整块被屏蔽的区域。合并前向传播改变计算组织方式，但不改变各去噪预测可访问的上下文

### Stratified Time And First Hitting Sampling

**分层采样时间步**。对批量大小 $`K`$、每条序列 $`B`$ 个块的情况，将 $`[0,1]`$ 划分为 $`KB`$ 个等长区间，分别为每个序列和块采样时间步

```math
t(k,b)\sim\mathcal{U}\!\left(
\frac{(k-1)B+b-1}{KB},\frac{(k-1)B+b}{KB}
\right),\qquad 1\leq k\leq K,\quad1\leq b\leq B\tag{A.22}
```

- 这使一个批次中的时间步更均匀地覆盖噪声区间。若需要对任意不同的逐块被积函数保持均匀时间目标的无偏性，还应随机打乱区间与块的对应关系；固定区间分配本身不提供这一保证

训练时，论文将采样变量线性映射到指定的掩码比例区间

```math
u(k,b)=1-\alpha_{t(k,b)}=\beta+(\omega-\beta)t(k,b),\qquad
0\leq\beta<\omega\leq1\tag{A.23}
```

- 评估完整 NELBO 时仍需覆盖完整噪声范围。仅在 $`[\beta,\omega]`$ 内采样、且不补偿区间外贡献，不能直接称为式（A.16）的无偏估计

<div class="lake-derivation" style="--derivation-border:#c2ae89;--derivation-background:#fffcf5">
<p class="lake-derivation-title">Sampling The Next Unmasking Time</p>
<div class="lake-derivation-body">

采用线性调度 $`\alpha_t=1-t`$，逆过程从 $`t=1`$ 向 $`t=0`$ 运行。设当前块仍有 $`n`$ 个掩码，当前时间为 $`t_n`$

- 对每个仍被掩码的 token ，其解掩码时刻条件分布为 $`\mathcal{U}(0,t_n)`$。逆时间中遇到的下一次事件是这些时刻的最大值
- 由于各 token 的掩码转移独立且具有相同概率，下一事件时间的分布函数为

```math
\Pr(t_{n-1}\leq s\mid t_n)
=\left(\frac{s}{t_n}\right)^n,\qquad 0\leq s\leq t_n\tag{A.24}
```

对分布函数求逆即可直接采样，无需逐个经过没有 token 变化的离散时间步

```math
t_{n-1}=t_n U_n^{1/n},\qquad U_n\sim\mathcal{U}(0,1),\quad n=L',\ldots,1\tag{A.25}
```

- 从当前 $`n`$ 个掩码位置中均匀选取一个位置，按去噪网络预测采样其 token ，随后令 $`n\leftarrow n-1`$。完成整个块后，再将该干净块的键和值加入历史缓存

</div>
</div>

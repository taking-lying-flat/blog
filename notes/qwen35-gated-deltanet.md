# Gated DeltaNet：模型调用、Chunk 与 Recurrent 实现

## GDN 与 Token Mixer

- 标准因果注意力先构造 token 两两之间的相似度，再经过 softmax 归一化。以下采用不含 softmax 的点积形式表示线性注意力，其历史键值对可累积到固定大小的状态矩阵。设 Q/K/V 按 token 排成行，$`M_{ij}=\mathbf 1_{j\le i}`$ 为二值因果掩码，$`\mathcal M_{ij}`$ 在可见位置为 0、不可见位置为 $`-\infty`$，两种输出分别为：

```math
O_{\mathrm{attn}}=\operatorname{softmax}(sQK^\top+\mathcal M)V,
\qquad O_{\mathrm{linear}}=s(QK^\top\odot M)V,
\qquad s=d_k^{-1/2}.
```

- 上述并行表达式包含序列长度平方数量的成对权重。FlashAttention 可通过分块与融合避免完整权重矩阵的显存往返，但成对计算量仍随序列长度平方增长。线性注意力的优势来自其等价递推：每个 head 维护 $`d_v\times d_k`$ 的状态，单步更新与读出均为 $`O(d_kd_v)`$。因果掩码不能直接移到矩阵乘法外，训练时可通过 chunkwise 分解，将块内计算组织为矩阵乘法，将块间依赖压缩到状态传递。

- 状态矩阵构成从 key 到 value 的线性关联记忆。外积叠加时，非正交 key 之间的交叉内积会耦合不同关联，形成检索干扰。GDN 以 $`\alpha_tS_{t-1}`$ 为衰减后的参考状态，以 $`v_t-\alpha_tS_{t-1}k_t`$ 为当前键值对的预测残差，沿 key 方向执行秩一校正。其中，$`\alpha_t`$ 调节历史状态的整体保留比例，$`\beta_t`$ 调节残差校正的步长，二者共同决定状态的遗忘与写入。输出门对状态读出结果进行逐通道调制，不参与状态递推。

<figure style="width: 100%; max-width: 600px; margin: 20px auto;">
  <img src="../../assets/gdn-architecture.png" alt="Gated DeltaNet 模型结构与 token mixer" width="2058" height="1122">
</figure>

- Token mixer 的 Q/K/V 分支依次经过线性投影、逐通道短因果卷积和 SiLU，Q/K 再沿每个 head 的特征维做 L2 归一化。短卷积提供局部时序特征，Gated Delta Rule 维护跨 token 的矩阵状态；两者在解码时分别保留卷积窗口与 recurrent state。

- 衰减门与写入系数由独立投影生成，不经过 Q/K/V 的短卷积。状态输出先按 head 归一化，再与输出门逐元素相乘，最后投影回隐藏维度。

## 前置知识：线性注意力与 DeltaNet

### 线性注意力与 Mamba2

- 对单个注意力头，$`\boldsymbol q_t,\boldsymbol k_t\in\mathbb R^{d_k}`$、$`\boldsymbol v_t\in\mathbb R^{d_v}`$ 均为列向量，状态 $`\mathbf S_t\in\mathbb R^{d_v\times d_k}`$。下述论文推导取输出缩放 $`s=1`$，实现部分保留 $`s=d_k^{-1/2}`$。论文 §2.1 的线性注意力通过外积累积键值关联，通过矩阵向量乘法读取状态：

```math
\mathbf S_t=\mathbf S_{t-1}+\boldsymbol v_t\boldsymbol k_t^\top,
\qquad \boldsymbol o_t=\mathbf S_t\boldsymbol q_t.
```

- 在零初态下，将状态展开并代入输出，得到矩阵并行形式。$`\mathbf Q,\mathbf K,\mathbf V`$ 按 token 排成行，M 为包含对角线的二值因果掩码：

```math
\boldsymbol o_t=\sum_{j=1}^{t}
(\boldsymbol q_t^\top\boldsymbol k_j)\boldsymbol v_j,
\qquad
\mathbf O=(\mathbf Q\mathbf K^\top\odot\mathbf M)\mathbf V.
```

- Mamba2 的标量衰减形式在状态更新中引入数据相关的 $`\alpha_t`$。令 $`\gamma_t=\prod_{i=1}^{t}\alpha_i`$，位置 j 的写入传播到位置 t 时保留 $`\gamma_t/\gamma_j`$，并行形式中的因果掩码相应变为带衰减的 $`\Gamma`$：

```math
\begin{aligned}
\mathbf S_t&=\alpha_t\mathbf S_{t-1}+\boldsymbol v_t\boldsymbol k_t^\top,
&\boldsymbol o_t&=\mathbf S_t\boldsymbol q_t,\\
\mathbf O&=(\mathbf Q\mathbf K^\top\odot\Gamma)\mathbf V,
&\Gamma_{ij}&=\begin{cases}\gamma_i/\gamma_j,&i\ge j,\\0,&i<j.\end{cases}
\end{aligned}
```

- 分块训练将序列划分为长度 C 的 chunk，块内累计衰减为 $`\gamma_{[t]}^r=\prod_{i=1}^{r}\alpha_{[t]}^i`$，$`\gamma_{[t]}^0=1`$。入口状态向块内位置 r 传播时乘 $`\gamma_{[t]}^r`$，当前位置的写入传播到块末时乘 $`\gamma_{[t]}^C/\gamma_{[t]}^r`$。将这些系数分别吸收到 Q、K 和入口状态中，得到论文 Eq. (2) 的分块形式：

```math
\begin{aligned}
\mathbf S_{[t+1]}&=\overrightarrow{\mathbf S}_{[t]}
+\mathbf V_{[t]}^\top\overrightarrow{\mathbf K}_{[t]},\\
\mathbf O_{[t]}&=\overleftarrow{\mathbf Q}_{[t]}\mathbf S_{[t]}^\top
+(\mathbf Q_{[t]}\mathbf K_{[t]}^\top\odot\Gamma_{[t]})\mathbf V_{[t]}.
\end{aligned}
```

- 这里 $`\overleftarrow{\boldsymbol q}_{[t]}^r=\gamma_{[t]}^r\boldsymbol q_{[t]}^r`$，$`\overrightarrow{\boldsymbol k}_{[t]}^r=(\gamma_{[t]}^C/\gamma_{[t]}^r)\boldsymbol k_{[t]}^r`$，$`\overrightarrow{\mathbf S}_{[t]}=\gamma_{[t]}^C\mathbf S_{[t]}`$。状态在 chunk 之间递推，块内输出通过矩阵乘法并行计算。

### DeltaNet 与 Gated Delta Rule

- 标量门对整个状态统一衰减。DeltaNet 则先由 $`\mathbf S_{t-1}\boldsymbol k_t`$ 读取当前 key 对应的旧 value，再以 beta 控制旧值擦除和新值写入。对应论文 §2.2 的更新为：

```math
\begin{aligned}
\mathbf S_t&=\mathbf S_{t-1}
-\beta_t(\mathbf S_{t-1}\boldsymbol k_t)\boldsymbol k_t^\top
+\beta_t\boldsymbol v_t\boldsymbol k_t^\top\\
&=\mathbf S_{t-1}(\mathbf I-\beta_t\boldsymbol k_t\boldsymbol k_t^\top)
+\beta_t\boldsymbol v_t\boldsymbol k_t^\top.
\end{aligned}
```

- $`\mathbf I-\beta_t\boldsymbol k_t\boldsymbol k_t^\top`$ 是广义 Householder 转移矩阵；单位 key 且 $`\beta_t=2`$ 时为标准 Householder 反射。当前 sigmoid 写入门取 $`\beta_t\in(0,1)`$，控制沿 key 方向的部分擦除；正交方向不受这项更新影响。连续转移矩阵的乘积可用 WY 表示组织为块矩阵计算。

- Gated DeltaNet 将全局衰减与 Delta 更新结合，对应论文 Eq. (10)：先衰减入口状态，再按当前键值关联完成残差写入。

```math
\begin{aligned}
\mathbf S_t&=\mathbf S_{t-1}
\left[\alpha_t(\mathbf I-\beta_t\boldsymbol k_t\boldsymbol k_t^\top)\right]
+\beta_t\boldsymbol v_t\boldsymbol k_t^\top,\\
\boldsymbol o_t&=\mathbf S_t\boldsymbol q_t.
\end{aligned}
```

- $`\alpha_t`$ 控制历史状态的整体衰减，$`\beta_t`$ 控制当前 key 方向的更新强度。令 $`\alpha_t=1`$ 即退化为 DeltaNet；单位 key 且 $`\beta_t=1`$ 时，该方向上的旧 value 被新 value 替换。

### 在线学习与 TTT 视角

- 论文 Table 1 将 GDN 更新表示为在线目标的闭式解。该目标的参考状态为衰减后的 $`\alpha_t\mathbf S_{t-1}`$，关联项使用当前键值对相对于该参考状态的残差：

```math
\mathbf S_t=\underset{\mathbf S}{\arg\min}\left[\|\mathbf S-\alpha_t\mathbf S_{t-1}\|_F^2-2\left\langle\mathbf S\boldsymbol k_t,\beta_t(\boldsymbol v_t-\alpha_t\mathbf S_{t-1}\boldsymbol k_t)\right\rangle\right].
```

- 从 TTT（Test-Time Training）视角，状态矩阵作为在线回归模型的参数。DeltaNet 对平方回归损失执行一步梯度下降，beta 对应更新步长：

```math
\mathcal L(\mathbf S)=\tfrac12\|\mathbf S\boldsymbol k_t-\boldsymbol v_t\|_2^2,\qquad \mathbf S_t=\mathbf S_{t-1}-\beta_t\nabla\mathcal L(\mathbf S_{t-1})=\mathbf S_{t-1}-\beta_t(\mathbf S_{t-1}\boldsymbol k_t-\boldsymbol v_t)\boldsymbol k_t^\top.
```

- GDN 在这一步更新前加入自适应权重衰减 alpha。在线目标给出状态更新的闭式形式，TTT 则将同一状态递推解释为对键值回归问题的逐 token 优化。

## Chunk 的矩阵表示

<figure class="gdn-chunk-diagram">
  <img src="../../assets/gdn-chunk-parallel.png" alt="顺序计算与分离状态递推后的 Chunk 并行计算示意图" width="1260" height="370" loading="lazy">
</figure>

### DeltaNet 的 WY 表示与 UT 变换

- 将序列划分为长度 C 的块，块内 Q/K/V 按 token 排成行。$`\mathbf S_{[t]}`$ 为第 t 块的入口状态，$`\mathbf S_{[t]}^r`$ 为处理块内前 r 个 token 后的状态。不含衰减的 DeltaNet 在块内累积广义 Householder 转移矩阵；WY 表示将这一连乘写成单位矩阵减去低秩项，对应论文 Eq. (4)：

```math
\mathbf P_{[t]}^r
=\prod_{i=1}^{r}\left(\mathbf I-\beta_{[t]}^i\boldsymbol k_{[t]}^i\boldsymbol k_{[t]}^{i\top}\right)
=\mathbf I-\sum_{i=1}^{r}\boldsymbol w_{[t]}^i\boldsymbol k_{[t]}^{i\top}.
```

- 乘积按 token 顺序从左向右排列。WY 系数按以下递推生成，对应论文 Eq. (4)。每个新 key 的写入系数需要扣除此前 key 内积带来的贡献：

```math
\boldsymbol w_{[t]}^r=\beta_{[t]}^r\left(
\boldsymbol k_{[t]}^r-\sum_{i<r}\boldsymbol w_{[t]}^i
(\boldsymbol k_{[t]}^{i\top}\boldsymbol k_{[t]}^r)\right).
```

- 新增状态同样可以写成 $`\sum_{i=1}^r\boldsymbol u_{[t]}^i\boldsymbol k_{[t]}^{i\top}`$，U 的递推将右端的当前 key 换成 value；两者共享此前 key 内积形成的下三角系数。

- UT 变换将上述逐位置递推写为同一个下三角系统，对应论文 Eq. (6)–(7)：

```math
\begin{aligned}
\mathbf T_{[t]}&=
\left[\mathbf I+\operatorname{strictLower}\!\left(
\operatorname{diag}(\beta_{[t]})\mathbf K_{[t]}\mathbf K_{[t]}^\top
\right)\right]^{-1}\operatorname{diag}(\beta_{[t]}),\\
\mathbf W_{[t]}&=\mathbf T_{[t]}\mathbf K_{[t]},\qquad
\mathbf U_{[t]}=\mathbf T_{[t]}\mathbf V_{[t]}.
\end{aligned}
```

- WY 表示把转移矩阵的连乘改写为低秩更新的累加；UT 变换进一步将 W/U 的计算组织成三角求解与矩阵乘法。求解内部的因果依赖仍然存在，通过 16×16 局部前代与块间合并完成计算。

### Gated DeltaNet 的衰减与扩展 WY 表示

- 在每个 chunk 内定义累计衰减 $`\gamma_{[t]}^r=\prod_{i=1}^r\alpha_{[t]}^i`$。将 Gated Delta Rule 展开，块内状态由入口状态的传播与当前块的新增状态组成：

```math
\begin{aligned}
\mathbf S_{[t]}^r&=\mathbf S_{[t]}\mathbf F_{[t]}^r+\mathbf G_{[t]}^r,\\
\mathbf F_{[t]}^r&=\prod_{i=1}^{r}\alpha_{[t]}^i
\left(\mathbf I-\beta_{[t]}^i\boldsymbol k_{[t]}^i\boldsymbol k_{[t]}^{i\top}\right)
=\gamma_{[t]}^r\mathbf P_{[t]}^r.
\end{aligned}
```

- 扩展的 WY 表示将每次新增写入传播到位置 r 的衰减显式保留。$`\widetilde{\boldsymbol u}_{[t]}^r`$ 扣除此前写入在当前 key 上的预测，再乘当前位置的写入系数：

```math
\begin{aligned}
\mathbf G_{[t]}^r&=\sum_{i=1}^{r}
\frac{\gamma_{[t]}^r}{\gamma_{[t]}^i}
\widetilde{\boldsymbol u}_{[t]}^i\boldsymbol k_{[t]}^{i\top},\\
\widetilde{\boldsymbol u}_{[t]}^r
&=\beta_{[t]}^r\left(\boldsymbol v_{[t]}^r-
\sum_{i=1}^{r-1}\widetilde{\boldsymbol u}_{[t]}^i
\left(\frac{\gamma_{[t]}^r}{\gamma_{[t]}^i}
\boldsymbol k_{[t]}^{i\top}\boldsymbol k_{[t]}^r\right)\right).
\end{aligned}
```

- 令 $`(\Gamma_{[t]})_{ij}=\gamma_{[t]}^i/\gamma_{[t]}^j`$（$`i\ge j`$），上三角为零。按行堆叠后，得到论文 §3.3 的 UT 形式：

```math
\widetilde{\mathbf U}_{[t]}=
\left[\mathbf I+\operatorname{strictLower}\!\left(
\operatorname{diag}(\beta_{[t]})
\left(\Gamma_{[t]}\odot\mathbf K_{[t]}\mathbf K_{[t]}^\top\right)
\right)\right]^{-1}
\operatorname{diag}(\beta_{[t]})\mathbf V_{[t]}.
```

- 与 DeltaNet 相比，三角系统中的 Gram 矩阵由 $`\mathbf K_{[t]}\mathbf K_{[t]}^\top`$ 改为 $`\Gamma_{[t]}\odot\mathbf K_{[t]}\mathbf K_{[t]}^\top`$。

### 块间状态与块内输出

- 沿用论文的箭头记号，将入口到当前位置、当前位置到块末的衰减分别写为：

```math
\begin{aligned}
\overleftarrow{\boldsymbol q}_{[t]}^r&=\gamma_{[t]}^r\boldsymbol q_{[t]}^r,
&\overleftarrow{\boldsymbol w}_{[t]}^r&=\gamma_{[t]}^r\boldsymbol w_{[t]}^r,\\
\overrightarrow{\boldsymbol k}_{[t]}^r&=
\frac{\gamma_{[t]}^C}{\gamma_{[t]}^r}\boldsymbol k_{[t]}^r,
&\overrightarrow{\mathbf S}_{[t]}&=\gamma_{[t]}^C\mathbf S_{[t]}.
\end{aligned}
```

- 其中 W 为无衰减 DeltaNet 的 WY 系数。块间状态更新先将入口状态衰减至块末，再累积当前块的残差写入；论文原式为：

```math
\mathbf S_{[t+1]}=\overrightarrow{\mathbf S}_{[t]}+
\left(\widetilde{\mathbf U}_{[t]}-
\overleftarrow{\mathbf W}_{[t]}\mathbf S_{[t]}^\top\right)^\top
\overrightarrow{\mathbf K}_{[t]}.
```

- 块内输出由历史读出与当前块内的残差贡献相加。将论文 §3.3 输出式中的区间衰减显式写出，得到：

```math
\mathbf O_{[t]}=\overleftarrow{\mathbf Q}_{[t]}\mathbf S_{[t]}^\top+
\left(\mathbf Q_{[t]}\mathbf K_{[t]}^\top\odot\Gamma_{[t]}\right)
\left(\widetilde{\mathbf U}_{[t]}-
\overleftarrow{\mathbf W}_{[t]}\mathbf S_{[t]}^\top\right).
```

- 这里必须使用带衰减的因果掩码 $`\Gamma_{[t]}`$，其可见位置取 $`\gamma_{[t]}^i/\gamma_{[t]}^j`$。论文原式在此使用 M 记号；若只把它解释为二值因果掩码，就会漏掉写入从位置 j 传播到 query 位置 i 的衰减。FLA 的输出 kernel 显式计算这个比例。

- W/U 可以按 chunk 并行生成；状态更新按 chunk 顺序推进；各块入口状态得到后，输出再次按 chunk 并行。块内残差 $`\widetilde{\mathbf U}_{[t]}-\overleftarrow{\mathbf W}_{[t]}\mathbf S_{[t]}^\top`$ 在状态更新与输出计算之间复用。

## 模型调用方式：Qwen3.5 GatedDeltaNet

### 输入投影

- 设输入为 $`X\in\mathbb R^{T\times d_{\mathrm{model}}}`$。Q/K 的总投影维度为 $`d_Q=H_Kd_k`$，V 的总投影维度为 $`d_V=H_Vd_v`$。四个线性映射分别产生卷积输入、衰减参数、写入参数和输出门：

```math
P=XW_{qkv}^{\top},\qquad a=XW_a^{\top},\quad b=XW_b^{\top},\qquad Z=XW_z^{\top}.
```

- `Qwen3_5MoeGatedDeltaNet` 将 Q/K/V 合并为一次投影，随后按通道拆分。衰减参数和写入参数只需要每个 value head 一个标量，因此 `in_proj_a`、`in_proj_b` 的输出维度为 `num_v_heads`；输出门需要逐通道调节读出结果，其投影维度与 V 相同。

```python
def __init__(self, config: Qwen3_5MoeConfig, layer_idx: int):
    super().__init__()
    self.hidden_size = config.hidden_size
    self.num_v_heads = config.linear_num_value_heads
    self.num_k_heads = config.linear_num_key_heads
    self.head_k_dim = config.linear_key_head_dim
    self.head_v_dim = config.linear_value_head_dim
    self.key_dim = self.head_k_dim * self.num_k_heads
    self.value_dim = self.head_v_dim * self.num_v_heads
    self.conv_kernel_size = config.linear_conv_kernel_dim
    self.layer_idx = layer_idx
    self.activation = config.hidden_act
    self.layer_norm_epsilon = config.rms_norm_eps
    self.conv_dim = self.key_dim * 2 + self.value_dim
    self.conv1d = nn.Conv1d(
        in_channels=self.conv_dim, out_channels=self.conv_dim, bias=False,
        kernel_size=self.conv_kernel_size, groups=self.conv_dim, padding=self.conv_kernel_size - 1,
    )
    self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
    A = torch.empty(self.num_v_heads).uniform_(0.01, 16)
    self.A_log = nn.Parameter(torch.log(A))
    self.norm = Qwen3_5MoeRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
    self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
    self.layer_type = config.layer_types[layer_idx]
    self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
    self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
    self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
    self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
```

### Short Conv、门控与 Head 映射

- **Short Conv 的计算。** Q/K/V 的联合投影 P 先经过 depthwise causal convolution，再应用 SiLU。每个通道有独立的宽度 w 的卷积核，不进行通道间混合。按 PyTorch 的权重排列，第 c 个通道在位置 t 的输出为：

```math
\operatorname{ShortConv}(P)_{t,c}
=\operatorname{SiLU}\!\left(\sum_{j=0}^{w-1}
\theta_{c,j}P_{t-w+1+j,c}\right).
```

- 序列起点之前补零；w=4 时，输出依赖当前 token 与之前三个 token。代码使用 `groups=hidden_size` 实现逐通道卷积，`padding=w-1` 后截取前 `seq_len` 个位置，保证不使用未来输入。以下是 Transformers 中 `causal_conv1d_fn` 的实际 PyTorch 后备实现；对应加速算子可用时由 kernel dispatch 替换。

```python
def causal_conv1d_fn(
    hidden_states: torch.Tensor,
    weight: nn.Parameter,
    bias: nn.Parameter | None = None,
    activation: str | None = None,
    **kwargs,
):
    _, hidden_size, seq_len = hidden_states.shape
    padding = weight.shape[-1] - 1

    out = F.conv1d(
        hidden_states.to(weight.dtype),
        weight=weight.unsqueeze(1),
        bias=bias,
        padding=padding,
        groups=hidden_size,
    )[:, :, :seq_len]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)
```

- **Short Conv 的作用。** K 经局部卷积后包含邻近 token 的组合特征，可作为局部 n-gram 地址；Q/V 的卷积分别补充局部查询与写入内容。从在线回归视角看，这一步改变状态更新所使用的键值表示。随后 Q/K 做 L2 归一化，局部卷积产生的特征再进入 Gated Delta Rule 的长期状态递推。

- 令 $`\mathcal C`$ 表示按通道独立计算的因果卷积。卷积后的 Q/K 尚未归一化，记为 $`Q^0,K^0`$。该阶段计算：

```math
\begin{aligned}
[Q^0,K^0,V]&=\operatorname{SiLU}\!\left(\mathcal C(P)\right),\\
\beta&=\sigma(b),\\
g&=-\exp(A_{\log})\odot\operatorname{softplus}(a+d_{\mathrm{bias}}),
\qquad \alpha=\exp(g),\\
\bar Q_h&=Q^0_{\lfloor h/r\rfloor},\quad
\bar K_h=K^0_{\lfloor h/r\rfloor},\qquad r=H_V/H_K.
\end{aligned}
```

- 短卷积沿时间维混合局部上下文，各通道之间不发生卷积混合。合并投影的输出在卷积时采用 `[batch, channels, time]` 排列，卷积后恢复时间优先布局，再拆分为多头 Q/K/V。

- `A_log` 和 `dt_bias` 是每个 value head 的可学习参数。上述参数化保证 `g` 为负值，从而将历史衰减限制在 0 与 1 之间；`beta` 经 sigmoid 控制残差写入强度。Grouped Value Attention 通过 head 映射使多个 value head 共享 Q/K，模型代码使用 `repeat_interleave` 显式实现该映射。下面展示无缓存输入的投影与卷积分支。

```python
def forward(
    self, hidden_states: torch.Tensor, cache_params: Cache | None = None,
    attention_mask: torch.Tensor | None = None, **kwargs: Unpack[TransformersKwargs],
):
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape
    mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
    mixed_qkv = causal_conv1d_fn(
        mixed_qkv,
        self.conv1d.weight.squeeze(1),
        self.conv1d.bias,
        activation=self.activation,
    ).transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1,
    )
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = self.in_proj_b(hidden_states).sigmoid()
    a = self.in_proj_a(hidden_states)
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)

    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
```

- **解码时的卷积缓存。** 单 token 解码沿用相同的短卷积，只需接续缓存中的最近投影，按上述 Short Conv 公式计算当前位置。`conv_state` 保存短卷积输入，GDN 的 `recurrent_state` 保存矩阵状态；两份缓存分别对应局部窗口与长期递推。下面是 `causal_conv1d_update` 的实际后备实现：先拼接缓存与新输入，原位保留最近窗口，再取最后 `seq_len` 个卷积输出。

```python
def causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: nn.Parameter,
    bias: nn.Parameter | None = None,
    activation: str | None = None,
):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = out[:, :, -seq_len:]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)
```

### 状态算子与输出门控

- Q/K 在进入状态更新前沿每个 head 的特征维归一化。记 $`\mathcal D`$ 为 Gated Delta Rule 算子，则输入归一化、状态计算和输出映射可写为：

```math
\begin{aligned}
q_{t,h}&=\frac{\bar q_{t,h}}{\sqrt{\|\bar q_{t,h}\|_2^2+\varepsilon}},
&k_{t,h}&=\frac{\bar k_{t,h}}{\sqrt{\|\bar k_{t,h}\|_2^2+\varepsilon}},\\
(O,S_{\mathrm{final}})&=\mathcal D(Q,K,V,g,\beta;S_{\mathrm{initial}}),\\
Y&=\operatorname{Concat}_{h}\!\left[
\operatorname{RMSNorm}(O_h)\odot\operatorname{SiLU}(Z_h)
\right]W_o^\top.
\end{aligned}
```

- Chunk 与 recurrent 实现同一个状态算子。模型根据输入长度和已有状态选择计算方式：完整序列使用 chunk；存在缓存且输入长度为 1 时使用 recurrent。两者均返回当前输出；启用缓存时还返回序列末态并回写缓存，供后续输入接续计算。

- 输出归一化独立作用于每个 value head。`self.norm` 先执行 RMSNorm，再乘 `SiLU(z)`；head 维合并后，`out_proj` 将结果映射回模型隐藏维度。以下代码合并了两条分支中相同的调用参数。

```python
use_precomputed_states = (
    cache_params is not None
    and cache_params.has_previous_state(self.layer_idx, state_idx=0)
)
recurrent_state = (
    cache_params.layers[self.layer_idx].recurrent_states[0]
    if use_precomputed_states else None
)
if use_precomputed_states and seq_len == 1:
    delta_rule = torch_recurrent_gated_delta_rule
else:
    delta_rule = torch_chunk_gated_delta_rule

core_attn_out, last_recurrent_state = delta_rule(
    query, key, value,
    g=g, beta=beta,
    initial_state=recurrent_state,
    output_final_state=cache_params is not None,
    use_qk_l2norm_in_kernel=True,
    cu_seqlens=kwargs.pop("cu_seq_lens_q", None),
    **kwargs,
)
if cache_params is not None:
    cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

core_attn_out = self.norm(
    core_attn_out.reshape(-1, self.head_v_dim),
    z.reshape(-1, self.head_v_dim),
)
output = self.out_proj(core_attn_out.reshape(batch_size, seq_len, -1))
```

## Chunk-wise 算法代码解析

### 前向计算的矩阵分解

- 以下对应 FLA v0.5.2 的 Triton 实现，采用 scalar gate、`state_v_first=False`。记 `H` 为 Q/K head 数、`HV` 为 value head 数，代码中的 `K/V` 分别对应数学维度 $`d_k/d_v`$，`BT=64` 为 chunk 长度。固定长度输入的中间张量如下；$`N_T=\lceil T/BT\rceil`$ 为每条序列的 chunk 数。

| 代码变量 | 形状 | 含义 |
| --- | --- | --- |
| `q, k` | `[B, T, H, K]` | 进入状态算子前已完成可选的 L2 归一化 |
| `v, u, v_new, o` | `[B, T, HV, V]` | 原始 value、三角变换结果、实际写入残差、输出 |
| `g, beta` | `[B, T, HV]` | 逐 head 的衰减参数与写入步长 |
| `A` | `[B, T, HV, BT]` | 每个 token 保存所在 chunk 的一行三角逆矩阵 |
| `w` | `[B, T, HV, K]` | 用于扣除入口状态预测的变换后 key |
| `h` | `[B, N_T, HV, K, V]` | 每个 chunk 更新前的入口状态 |
| `initial_state, final_state` | `[N, HV, K, V]` | 每条逻辑序列的初态与末态；定长时 `N=B` |

- 论文状态 $`\mathbf S\in\mathbb R^{d_v\times d_k}`$ 在这一路径中按 $`\mathbf H=\mathbf S^\top\in\mathbb R^{d_k\times d_v}`$ 存放，所以代码中的读出是 `q @ h`，写入是 `k.T @ residual`。`state_v_first=True` 改变存储排列，不改变状态更新的数学定义。`HV/H` 个 value head 可共享同一个 Q/K head，但各自拥有 gate、beta 和状态。

- Chunk 前向围绕论文中的 $`\widetilde{\mathbf U}_{[t]}`$、$`\overleftarrow{\mathbf W}_{[t]}`$、$`\mathbf S_{[t]}`$ 和 $`\mathbf O_{[t]}`$ 展开。计算顺序为：累计门控、构造 KKT、求解下三角系统、生成 W/U、递推状态、计算输出。其中状态与输出复用同一个残差：

```math
\mathbf V_{\mathrm{new},[t]}=
\widetilde{\mathbf U}_{[t]}-
\overleftarrow{\mathbf W}_{[t]}\mathbf S_{[t]}^\top.
```

- `chunk_gated_delta_rule_fwd_intra` 构造并求解块内系统，生成逆矩阵 `A`、`u` 和 `w`；`chunk_gated_delta_rule_fwd_h` 引入块入口状态，计算 `v_new` 及下一块状态；`chunk_fwd_o` 完成输出计算。这里 `h` 保存所有块的入口状态，`final_state` 保存各序列的最终状态。

- 默认 64-token 路径将 KKT 构造与下三角求解融合执行，随后单独生成 W/U。块内变换可以跨 chunk 并行；状态更新仍按 chunk 顺序递推；入口状态生成后，各块输出又可以并行计算。下面给出前向入口，块内分析采用固定长度、无 context parallelism、外部已计算 gate 的默认分支。

```python
def chunk_gated_delta_rule_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
    scale: float, initial_state: torch.Tensor, output_final_state: bool,
    state_v_first: bool = False, cu_seqlens: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None, chunk_indices: torch.LongTensor | None = None,
    use_gate_in_kernel: bool = False, A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None, chunk_size: int = 64,
):
    g_input = g if use_gate_in_kernel else None
    if use_gate_in_kernel:
        g = gdn_gate_chunk_cumsum(
            g=g, A_log=A_log, chunk_size=chunk_size, scale=RCP_LN2, dt_bias=dt_bias,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        )
    else:
        g = chunk_local_cumsum(
            g, chunk_size=chunk_size, scale=RCP_LN2, cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
    w, u, A = chunk_gated_delta_rule_fwd_intra(
        k=k, v=v, g=g, beta=beta, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    if cp_context is not None:
        initial_state = chunk_gated_delta_rule_fwd_h_pre_process(
            k=k, w=w, u=u, g=g, cu_seqlens=cu_seqlens, initial_state=initial_state,
            context=cp_context, state_v_first=state_v_first, chunk_size=chunk_size,
        )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g, initial_state=initial_state, output_final_state=output_final_state,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, state_v_first=state_v_first,
        chunk_size=chunk_size,
    )
    if cp_context is not None:
        initial_state = compress_h0(initial_state, context=cp_context)
    o = chunk_fwd_o(
        q=q, k=k, v=v_new, h=h, g=g, scale=scale, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices, state_v_first=state_v_first, chunk_size=chunk_size,
    )
    return (g, o, A, final_state, initial_state, g_input)
```

- **分块为什么仍与逐 token 递推等价。** 在一个 chunk 内省略块编号，令 $`\mathbf B_\beta=\operatorname{diag}(\beta)`$、$`\mathbf D=\operatorname{diag}(\gamma)`$、$`\mathbf H_0=\mathbf S_{[t]}^\top`$，把实际写入向量按行组成 $`\mathbf E=\mathbf V_{\mathrm{new},[t]}`$。位置 i 的残差必须扣除入口状态和此前写入在当前 key 上的预测：

```math
\boldsymbol e_i=\beta_i\left(\boldsymbol v_i-\gamma_i\mathbf H_0^\top\boldsymbol k_i-\sum_{j<i}\frac{\gamma_i}{\gamma_j}(\boldsymbol k_i^\top\boldsymbol k_j)\boldsymbol e_j\right).
```

- 将依赖此前残差的项移到左侧，就得到单位下三角系统。令 $`\mathbf A=\operatorname{strictLower}(\mathbf B_\beta(\Gamma\odot\mathbf K\mathbf K^\top))`$，则：

```math
(\mathbf I+\mathbf A)\mathbf E=\mathbf B_\beta\mathbf V-\mathbf B_\beta\mathbf D\mathbf K\mathbf H_0,\qquad \mathbf E=\widetilde{\mathbf U}-\overleftarrow{\mathbf W}\mathbf H_0.
```

- 因此 U 和 W 可以先独立于入口状态生成，等入口状态到达后只需一次矩阵乘法和相减。块内的 token 依赖被三角求解吸收，块间依赖则由状态递推承担；并行化没有删除任何历史写入。

### 累积门控：将连乘转为前缀和

- 逐 token 的输入为 $`g_{[t]}^i=\log\alpha_{[t]}^i`$。对每个 chunk 独立求前缀和，即得到累计衰减的对数：

```math
\widehat g_{[t]}^r=\frac{1}{\ln 2}\sum_{i=1}^{r}g_{[t]}^i,
\qquad \gamma_{[t]}^r=2^{\widehat g_{[t]}^r},
\qquad \frac{\gamma_{[t]}^r}{\gamma_{[t]}^i}
=2^{\widehat g_{[t]}^r-\widehat g_{[t]}^i}.
```

- 算子位于 `fla/ops/utils/cumsum.py`，标量 kernel 的网格为 `(NT, B * H)`：每个 program 负责一个 chunk 和一个 gate head。`o_t` 构造块内 token 索引，`m_t` 屏蔽尾块越界位置；按 `[B, T, H]` 布局加载后，`tl.cumsum` 沿时间轴执行前缀和。累积在 FP32 中完成，并通过 `RCP_LN2` 将自然对数转换为以 2 为底的对数。后续 kernel 使用 `exp2` 还原整体衰减和区间衰减，避免在各阶段重复计算连乘。

```python
@triton.jit(do_not_specialize=['T'])
def chunk_local_cumsum_scalar_kernel(
    s, o, scale, cu_seqlens, chunk_indices, T, B: tl.constexpr, H: tl.constexpr,
    BT: tl.constexpr, REVERSE: tl.constexpr, HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr, HEAD_FIRST: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64),
        )
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32))
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    p_s = s + bos*H + i_h + o_t * H
    p_o = o + bos*H + i_h + o_t * H
    # [BT]
    b_s = tl.load(p_s, mask=m_t, other=0.0).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = tl.sum(b_s, axis=0)
        b_o = -b_o + b_z[None] + b_s
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_t)
```

- `REVERSE` 分支计算后缀和：$`\sum_{j\ge i}x_j=\sum_jx_j-\sum_{j\le i}x_j+x_i`$，对应前缀和的反向传播。`HAS_SCALE` 决定是否乘调用侧传入的缩放，GDN 前向传入 `RCP_LN2`。上述代码保留时间优先的访存分支。

- 每个 chunk 独立累计 gate。块前历史已经包含在入口状态中，再乘当前块的累计衰减即可；因此前缀和不需要跨 chunk 延伸。对于变长输入，块索引同时指定序列编号和序列内块编号，累计过程不能跨越序列边界。

- 对非正的逐 token gate，因果方向的累计差 $`\widehat g_i-\widehat g_j`$（$`i\ge j`$）也非正，`exp2` 得到 $`(0,1]`$ 内的衰减。实现直接对累计对数作差，不先生成很小的 gamma 再相除，避免长衰减下出现 `0/0`。`g` 在前缀和之后已表示 $`\widehat g`$，不再是原来的逐 token 自然对数 gate。

### KKT：构造块内递推系数

- 名称 KKT 指 $`\mathbf K\mathbf K^\top`$ 的 key Gram 矩阵。对角线必须排除：当前 value 的直接写入已放在三角系统右侧，若再把自身 key 内积放入左侧，就会改变 Delta Rule 的更新。

- 第 $`i`$ 个位置的 Delta 修正依赖更早位置的 key 内积。更新系数按行缩放，区间衰减逐元素作用于 Gram 矩阵，得到：

```math
\begin{aligned}
\mathbf A_{[t]}&=\operatorname{strictLower}\!\left(
\operatorname{diag}(\beta_{[t]})
\left(\Gamma_{[t]}\odot\mathbf K_{[t]}\mathbf K_{[t]}^\top\right)\right),\\
(\mathbf A_{[t]})_{ij}&=
\begin{cases}
\beta_{[t]}^i\,2^{\widehat g_{[t]}^i-\widehat g_{[t]}^j}
\boldsymbol k_{[t]}^{i\top}\boldsymbol k_{[t]}^j,&i>j,\\
0,&i\le j.
\end{cases}
\end{aligned}
```

- `fla/ops/common/chunk_scaled_dot_kkt.py` 的 `chunk_scaled_dot_kkt_fwd_kernel` 实现这一计算。网格为 `(NT, B * HV)`，每个 program 生成一个 head 的块内系数。K 维按 `BK` 分片，`tl.dot(b_k, tl.trans(b_k))` 将各片段的内积累加到 FP32 的 `[BT, BT]` 矩阵中。

- 这是 16/32-token 分步路径使用的 kernel；64-token 默认路径将相同计算与下三角求解融合。独立 KKT 对应内积、衰减、beta、严格下三角四个矩阵操作。

- **定位 chunk 并累计 Gram 矩阵。** `chunk_indices` 给出序列编号与序列内块编号，`cu_seqlens` 确定起点 `bos` 和有效长度 T；定长分支直接通过 batch 编号计算起点。K 维按 `BK` 分片，逐片累加 $`\mathbf K_{[t]}\mathbf K_{[t]}^\top`$。`i_h // (HV // H)` 将 value head 映射到共享的 key head；K 的时间步长为 `H * K`，beta 的时间步长为 `HV`。`m_t` 屏蔽尾块的无效 token。

```python
@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_kkt_fwd_kernel(
    k, g, beta, A, cu_seqlens, chunk_indices, T, H: tl.constexpr, HV: tl.constexpr,
    K: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64),
        )
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32))
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    p_b = beta + bos*HV + i_h + o_t * HV
    b_b = tl.load(p_b, mask=m_t, other=0.0)

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        p_k = k + (bos*H + i_h // (HV // H)) * K + o_t[:, None] * (H*K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_t[:, None] & (o_k < K)[None, :], other=0.0)
        b_A += tl.dot(b_k, tl.trans(b_k))
```

- **施加区间衰减、写入系数和因果约束。** `exp2(b_g_diff)` 计算区间衰减 $`(\Gamma_{[t]})_{ij}=\gamma_{[t]}^i/\gamma_{[t]}^j`$ 并逐元素乘入 Gram 矩阵；`b_b[:, None]` 按行乘 beta，`m_A` 仅保留 $`i>j`$ 的系数。结果通过 `p_A` 按 `[B, T, HV, BT]` 写回，每个有效 token 保存所在块的一行系数。以下代码接续上一段的 kernel 函数体。

```python
if USE_G:
    p_g = g + bos*HV + i_h + o_t * HV
    b_g = tl.load(p_g, mask=m_t, other=0.0)
    b_g_diff = b_g[:, None] - b_g[None, :]
    b_A *= exp2(b_g_diff)
b_A *= b_b[:, None]

m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
b_A = tl.where(m_A, b_A, 0)
p_A = A + (bos*HV + i_h) * BT + o_t[:, None] * (BT*HV) + tl.arange(0, BT)[None, :]
tl.store(p_A, b_A.to(p_A.dtype.element_ty), mask=m_t[:, None])
```

- 64-token 融合 kernel 位于 `fla/ops/gated_delta_rule/chunk_fwd.py`，名称为 `chunk_gated_delta_rule_fwd_kkt_solve_kernel`。它将时间维拆成四个 16-token 子块，只计算四个对角块和六个非对角下三角块；对角块额外应用严格下三角掩码。这十个子矩阵保留在寄存器中，直接进入前代与合并阶段。

- **融合路径的边界掩码。** 四个子块的起点为 `i_tc0` 至 `i_tc3`，`o_i=tl.arange(0, 16)`；`m_tc0` 至 `m_tc3` 分别判断位置是否有效。以对角块 (0,0) 和非对角块 (1,0) 为例，默认融合 kernel 的衰减与 beta 处理为：

```python
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_fwd_kkt_solve_kernel(
    k, g, beta, A, cu_seqlens, chunk_indices, T, H: tl.constexpr, HV: tl.constexpr,
    K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr, USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    ...
    m_d = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    b_A00 *= tl.where(m_d & m_tc0[:, None] & m_tc0[None, :], exp2(b_g0[:, None] - b_g0[None, :]), 0.)
    b_A10 *= tl.where(m_tc1[:, None] & m_tc0[None, :], exp2(b_g1[:, None] - b_g0[None, :]), 0.)
    b_A00 = b_A00 * b_b0[:, None]
    b_A10 = b_A10 * b_b1[:, None]
```

- 非对角块 (1,0) 的所有有效元素都满足行位置晚于列位置，因而不需要额外的局部三角掩码。beta 总取行所属子块；gate 则用行子块减列子块。尾块无效位置的 gate 会以 0 加载，若直接乘其指数差，可能出现 `0 * inf`；融合实现先用 `tl.where` 把无效衰减置零，再与 Gram 矩阵相乘。这也是分步示例不能完全替代融合 kernel 细节的原因。

### 下三角求解：局部前代与块间合并

- 上一步得到严格下三角矩阵 $`\mathbf A_{[t]}`$。包含 beta 的完整 UT 变换为：

```math
\mathbf T_{[t]}=
\left[\mathbf I+\operatorname{strictLower}\!\left(
\operatorname{diag}(\beta_{[t]})
\left(\Gamma_{[t]}\odot\mathbf K_{[t]}\mathbf K_{[t]}^\top\right)
\right)\right]^{-1}\operatorname{diag}(\beta_{[t]}).
```

- $`\mathbf I+\mathbf A_{[t]}`$ 是对角线为 1 的下三角矩阵，可用前代法逐行求解其逆。kernel 只生成逆矩阵；右端的 $`\operatorname{diag}(\beta_{[t]})`$ 留到 W/U 阶段相乘。因此，求解后的代码变量 `A` 对应 $`(\mathbf I+\mathbf A_{[t]})^{-1}`$，尚不是完整的 $`\mathbf T_{[t]}`$。

- `fla/ops/utils/solve_tril.py` 采用两级分块：先对 16×16 对角块逐行前代，再通过分块矩阵公式合并。`BT=16` 使用 `solve_tril_16x16_kernel`，`BT=32/64` 分别使用 `merge_16x16_to_32x32_inverse_kernel` 与 `merge_16x16_to_64x64_inverse_kernel`。当前 GDN 默认 64-token 路径进一步将 KKT 与这两级求解融合到 `chunk_gated_delta_rule_fwd_kkt_solve_kernel`；以下分段摘录该 kernel 的实际代码。

- **16×16 对角块前代。** 以第一个对角块为例，`b_A00` 对应严格下三角子块 $`\mathbf A_{00}`$，`b_Ai00` 最终对应 $`\mathbf A^{\mathrm{inv}}_{00}=(\mathbf I+\mathbf A_{00})^{-1}`$。对块内元素，前代关系为：

```math
(\mathbf A^{\mathrm{inv}}_{00})_{ii}=1,\qquad
(\mathbf A^{\mathrm{inv}}_{00})_{ij}
=-(\mathbf A_{00})_{ij}
-\sum_{k=j+1}^{i-1}(\mathbf A_{00})_{ik}
(\mathbf A^{\mathrm{inv}}_{00})_{kj}\quad(i>j).
```

```python
b_Ai00 = -b_A00
for i in range(2, min(BC, T - i_tc0)):
    b_a00 = tl.sum(tl.where((o_i == i)[:, None], -b_A00, 0.), 0)
    b_a00 = tl.where(o_i < i, b_a00, 0.)
    b_a00 = b_a00 + tl.sum(b_a00[:, None] * b_Ai00, 0)
    b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
b_Ai00 += m_I
```

- 逆矩阵的严格下三角部分初始化为 `-b_A00`。第 0 行没有下三角元素，第 1 行的结果已由初始化给出，因此循环从索引 2 开始。`tl.sum(tl.where(...), 0)` 从寄存器矩阵中提取当前行；第二个 `tl.sum` 累积已求出的行对当前行的贡献；`tl.where` 将结果写回对应行，最后加上单位对角线 `m_I`。其余三个对角块执行相同操作。

- **合并相邻的 16×16 块。** 以零基索引表示四行四列子块。$`\mathbf A^{\mathrm{inv}}_{rs}`$ 表示整体逆矩阵的第 (r,s) 个子块，不是非对角块单独求逆。三个紧邻对角线的子块分别为：

```math
\mathbf A^{\mathrm{inv}}_{10}=-\mathbf A^{\mathrm{inv}}_{11}\mathbf A_{10}\mathbf A^{\mathrm{inv}}_{00},\quad \mathbf A^{\mathrm{inv}}_{21}=-\mathbf A^{\mathrm{inv}}_{22}\mathbf A_{21}\mathbf A^{\mathrm{inv}}_{11},\quad \mathbf A^{\mathrm{inv}}_{32}=-\mathbf A^{\mathrm{inv}}_{33}\mathbf A_{32}\mathbf A^{\mathrm{inv}}_{22}.
```

```python
b_Ai10 = -tl.dot(
    tl.dot(b_Ai11, b_A10, input_precision=SOLVE_TRIL_DOT_PRECISION),
    b_Ai00,
    input_precision=SOLVE_TRIL_DOT_PRECISION
)
b_Ai21 = -tl.dot(
    tl.dot(b_Ai22, b_A21, input_precision=SOLVE_TRIL_DOT_PRECISION),
    b_Ai11,
    input_precision=SOLVE_TRIL_DOT_PRECISION
)
b_Ai32 = -tl.dot(
    tl.dot(b_Ai33, b_A32, input_precision=SOLVE_TRIL_DOT_PRECISION),
    b_Ai22,
    input_precision=SOLVE_TRIL_DOT_PRECISION
)
```

- 每个结果由两个对角块的逆和一个原始非对角块相乘得到。上述三个子块之间没有依赖；完成后才能计算跨越更多子块的结果。

- **合并更远的非对角块。** 第 (2,0)、(3,1) 个子块依赖相邻块结果，左下角第 (3,0) 个子块进一步依赖第 (2,0) 个结果：

```math
\begin{aligned}
\mathbf A^{\mathrm{inv}}_{20}&=-\mathbf A^{\mathrm{inv}}_{22}
\left(\mathbf A_{20}\mathbf A^{\mathrm{inv}}_{00}
+\mathbf A_{21}\mathbf A^{\mathrm{inv}}_{10}\right),\\
\mathbf A^{\mathrm{inv}}_{31}&=-\mathbf A^{\mathrm{inv}}_{33}
\left(\mathbf A_{31}\mathbf A^{\mathrm{inv}}_{11}
+\mathbf A_{32}\mathbf A^{\mathrm{inv}}_{21}\right),\\
\mathbf A^{\mathrm{inv}}_{30}&=-\mathbf A^{\mathrm{inv}}_{33}
\left(\mathbf A_{30}\mathbf A^{\mathrm{inv}}_{00}
+\mathbf A_{31}\mathbf A^{\mathrm{inv}}_{10}
+\mathbf A_{32}\mathbf A^{\mathrm{inv}}_{20}\right).
\end{aligned}
```

```python
b_Ai20 = -tl.dot(
    b_Ai22,
    tl.dot(b_A20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION) +
    tl.dot(b_A21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
    input_precision=SOLVE_TRIL_DOT_PRECISION,
)
b_Ai31 = -tl.dot(
    b_Ai33,
    tl.dot(b_A31, b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION) +
    tl.dot(b_A32, b_Ai21, input_precision=SOLVE_TRIL_DOT_PRECISION),
    input_precision=SOLVE_TRIL_DOT_PRECISION,
)
b_Ai30 = -tl.dot(
    b_Ai33,
    tl.dot(b_A30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION) +
    tl.dot(b_A31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION) +
    tl.dot(b_A32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
    input_precision=SOLVE_TRIL_DOT_PRECISION,
)
```

- 两级分块将逐行依赖限制在 16×16 对角块内部，再用 `tl.dot` 完成块间合并。四个对角块与六个非对角块共同构成完整的 64×64 下三角逆矩阵。默认融合路径直接复用 KKT 的寄存器结果，省去严格下三角系数的中间写回和重新加载。

- 通用 `solve_tril` 在环境支持时使用 TMA descriptor，`FLA_TRIL_PRECISION` 默认 `ieee`；支持 TMA 时，autotune 在 `ieee` 与用户指定精度之间选择。GDN 默认融合 kernel 使用普通指针加载，块合并在支持 TF32 时使用 `tf32`，否则使用 `ieee`。二者使用同一分块求解原理，但访存与精度配置不同。

- **逆矩阵如何写回。** 融合路径先以 `torch.zeros(B, T, HV, BT, dtype=k.dtype)` 分配 A，只覆盖十个下三角子块，上三角保持零。第 i 行的首地址为 `(bos * HV + head) * BT + i * HV * BT`，行内列号是 chunk 内位置；因此逻辑上的 `[BT, BT]` 矩阵在全局缓冲区中按 token、head 交错存放。16/32-token 分步路径先把 KKT 写为 FP32，再由 `solve_tril(..., output_dtype=k.dtype)` 输出逆矩阵，两条路径最终都以 K 的 dtype 保存 A。

- 对三角系统求逆的正确性不依赖 beta 或 gate 的具体值：严格下三角矩阵满足 $`\mathbf A^{BT}=0`$，所以 $`(\mathbf I+\mathbf A)^{-1}=\sum_{r=0}^{BT-1}(-\mathbf A)^r`$。实现使用前代和块矩阵乘法求同一个有限展开，避免显式计算各次幂。FP32 累加、矩阵乘法精度以及写回 dtype 仍会影响有限精度误差。

### W/U：共享同一个三角变换

- `fla/ops/gated_delta_rule/wy_fast.py` 的 `recompute_w_u_fwd_kernel` 使用网格 `(NT, B * HV)`，每个 program 处理一个 chunk、一个 value head。读取求解后的逆矩阵 `b_A`，分别对 V、K 执行矩阵乘法，生成 $`\widetilde{\mathbf U}_{[t]}`$ 与 $`\overleftarrow{\mathbf W}_{[t]}`$。

- **计算 value 的 UT 变换。** 由于 $`\mathbf T_{[t]}`$ 的定义已经包含 beta，此处不再额外写一次 $`\operatorname{diag}(\beta_{[t]})`$：

```math
\widetilde{\mathbf U}_{[t]}
=\mathbf T_{[t]}\mathbf V_{[t]}
=(\mathbf I+\mathbf A_{[t]})^{-1}
\left(\operatorname{diag}(\beta_{[t]})\mathbf V_{[t]}\right).
```

```python
@triton.jit(do_not_specialize=['T'])
def recompute_w_u_fwd_kernel(
    k, v, beta, w, u, A, g, cu_seqlens, chunk_indices, T, H: tl.constexpr, HV: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_G: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    ...
    o_t = i_t * BT + tl.arange(0, BT)
    o_A = tl.arange(0, BT)
    m_t = o_t < T
    m_A = m_t[:, None] & (o_A[None, :] < BT)
    p_b = beta + bos*HV + i_h + o_t * HV
    b_b = tl.load(p_b, mask=m_t, other=0.0)

    p_A = A + (bos*HV + i_h) * BT + o_t[:, None] * (HV*BT) + o_A[None, :]
    b_A = tl.load(p_A, mask=m_A, other=0.0)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_t[:, None] & (o_v[None, :] < V)
        p_v = v + (bos*HV + i_h) * V + o_t[:, None] * (HV*V) + o_v[None, :]
        p_u = u + (bos*HV + i_h) * V + o_t[:, None] * (HV*V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_v, other=0.0)
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), mask=m_v)
```

- `b_b` 为逐 token 的 beta，`b_A` 为整个 chunk 的逆矩阵。V 维按 `BV` 划分：先通过 `b_v * b_b[:, None]` 按行缩放 V，再用 `tl.dot(b_A, b_vb)` 应用三角逆变换。计算结果按 `[B, T, HV, V]` 写入 `u`；此时尚未扣除块入口状态的预测。

- **计算衰减后的 W。** 入口状态传播到位置 r 时需要乘 $`\gamma_{[t]}^r`$，因此 K 的右端项还需按累计衰减缩放：

```math
\overleftarrow{\mathbf W}_{[t]}
=\mathbf T_{[t]}\operatorname{diag}(\gamma_{[t]})\mathbf K_{[t]}
=(\mathbf I+\mathbf A_{[t]})^{-1}
\left(\operatorname{diag}(\beta_{[t]})
\operatorname{diag}(\gamma_{[t]})\mathbf K_{[t]}\right).
```

```python
if USE_G:
    p_g = g + (bos*HV + i_h) + o_t * HV
    b_g = exp2(tl.load(p_g, mask=m_t, other=0.0))

for i_k in range(tl.cdiv(K, BK)):
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = m_t[:, None] & (o_k[None, :] < K)
    p_k = k + (bos*H + i_h // (HV // H)) * K + o_t[:, None] * (H*K) + o_k[None, :]
    p_w = w + (bos*HV + i_h) * K + o_t[:, None] * (HV*K) + o_k[None, :]
    b_k = tl.load(p_k, mask=m_k, other=0.0)
    b_kb = b_k * b_b[:, None]
    if USE_G:
        b_kb *= b_g[:, None]
    b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
    tl.store(p_w, b_w.to(p_w.dtype.element_ty), mask=m_k)
```

- `exp2` 将累计 gate 还原为 gamma；`b_kb` 依次乘 beta 与 gamma，然后与同一份 `b_A` 相乘。衰减乘在右端 K 上，不能任意移到 `tl.dot` 之后。这里得到的 W 与论文 $`\overleftarrow{\boldsymbol w}_{[t]}^r=\gamma_{[t]}^r\boldsymbol w_{[t]}^r`$ 一致，其中无箭头 W 使用无衰减的三角系统。

- K 的 head 通过 `i_h // (HV // H)` 映射到共享的 key head，W 则为每个 value head 独立保存。U 不需要额外乘 gamma：它的块内衰减已经包含在三角系数 $`\Gamma_{[t]}\odot\mathbf K_{[t]}\mathbf K_{[t]}^\top`$ 中；W 的额外 gamma 用于入口状态向块内各位置的传播。

- **与无衰减 WY 系数的对应。** 令 $`\mathbf A_0=\operatorname{strictLower}(\mathbf B_\beta\mathbf K\mathbf K^\top)`$，在实数数学中有 $`\mathbf A=\mathbf D\mathbf A_0\mathbf D^{-1}`$。因此带衰减的三角逆与无衰减的三角逆满足相似变换：

```math
\overleftarrow{\mathbf W}=(\mathbf I+\mathbf A)^{-1}\mathbf B_\beta\mathbf D\mathbf K=\mathbf D(\mathbf I+\mathbf A_0)^{-1}\mathbf B_\beta\mathbf K=\mathbf D\mathbf W_0.
```

- 这说明代码中先给 K 乘 gamma 再求解，与论文给无衰减 W 逐行乘 gamma 完全一致；若仍使用带衰减的同一个逆矩阵，则不能直接把这次缩放移到输出端。实现也不显式构造 $`\mathbf D^{-1}`$，而是用累计对数之差计算区间衰减。

- `recompute_w_u_fwd` 的 K/V 分片尺寸均为 64。U 的 `tl.dot` 显式设置 `allow_tf32=False`，W 使用该调用的默认精度；中间乘积与累加按对应 Triton dtype 规则执行，最终 `w` 使用 K 的 dtype、`u` 使用 V 的 dtype。源码中的 FP32 累加不能理解为所有中间量都以 FP32 保存。

### 块间状态：残差校正与衰减写入

- `fla/ops/common/chunk_delta_h.py` 的 `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` 实现论文的块间递推：

```math
\mathbf S_{[t+1]}=\overrightarrow{\mathbf S}_{[t]}+
\left(\widetilde{\mathbf U}_{[t]}-
\overleftarrow{\mathbf W}_{[t]}\mathbf S_{[t]}^\top\right)^\top
\overrightarrow{\mathbf K}_{[t]}.
```

- 网格为 `(ceil(V / BV), N * HV)`，每个 program 处理一条序列、一个 value head 和一片 value 通道，并沿 chunk 顺序递推。下面展开 scalar gate、K=128 的源码分支：`b_h1`、`b_h2` 分别保存 $`\mathbf S_{[t]}^\top`$ 的两个 `[64, BV]` 片段，初值从 `h0` 加载，无初始状态时置零。各基址已定位到当前序列与 head；`o_k1`、`o_k2` 分别覆盖 K 维的 0–63、64–127 通道。两段代码属于同一个 `for i_t in range(NT)` 循环。

- **保存入口状态并计算写入残差。** W 乘入口状态得到此前历史在当前块的预测，从 U 中扣除后即为实际残差：

```math
\mathbf V_{\mathrm{new},[t]}=
\widetilde{\mathbf U}_{[t]}-
\overleftarrow{\mathbf W}_{[t]}\mathbf S_{[t]}^\top.
```

```python
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k, v, w, v_new, g, gk, h, h0, ht, cu_seqlens, chunk_offsets, T, H: tl.constexpr,
    HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr,
    USE_G: tl.constexpr, USE_GK: tl.constexpr, USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr, SAVE_NEW_VALUE: tl.constexpr, STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    ...
    for i_t in range(NT):
        i_t_int64 = i_t.to(tl.int64)
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        p_h1 = h + i_t_int64 * HV*K*V + o_k1[:, None] * V + o_v[None, :]
        m_h1 = m_k1[:, None] & m_v[None, :]
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), mask=m_h1)
        p_h2 = h + i_t_int64 * HV*K*V + o_k2[:, None] * V + o_v[None, :]
        m_h2 = m_k2[:, None] & m_v[None, :]
        tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), mask=m_h2)

        p_w = w + o_t[:, None] * (HV*K) + o_k1[None, :]
        b_w = tl.load(p_w, mask=m_t[:, None] & m_k1[None, :], other=0.0)
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        p_w = w + o_t[:, None] * (HV*K) + o_k2[None, :]
        b_w = tl.load(p_w, mask=m_t[:, None] & m_k2[None, :], other=0.0)
        b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        p_v = v + o_t[:, None] * (HV*V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_t[:, None] & m_v[None, :], other=0.0) - b_v

        p_v = v_new + o_t[:, None] * (HV*V) + o_v[None, :]
        tl.store(p_v, b_v.to(p_v.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])
```

- 开头两次 `tl.store` 保存当前块的入口状态，供输出 kernel 读取。两个 `tl.dot` 分别收缩 W 与状态的两片 K 通道，结果相加后才构成完整的 $`\overleftarrow{\mathbf W}_{[t]}\mathbf S_{[t]}^\top`$。参数 `v` 在调用时传入的是 U，因此 `tl.load(p_v) - b_v` 正好得到上述残差，写入 `v_new`。

- **衰减至块末并累积状态。** 对残差的第 r 行乘 $`\gamma_{[t]}^C/\gamma_{[t]}^r`$，等价于将相同衰减乘入论文中的 $`\overrightarrow{\mathbf K}_{[t]}`$。转置后的状态更新为：

```math
\mathbf S_{[t+1]}^\top
=\gamma_{[t]}^C\mathbf S_{[t]}^\top+
\mathbf K_{[t]}^\top
\operatorname{diag}\!\left(\frac{\gamma_{[t]}^C}{\gamma_{[t]}}\right)
\left(\widetilde{\mathbf U}_{[t]}-
\overleftarrow{\mathbf W}_{[t]}\mathbf S_{[t]}^\top\right).
```

```python
last_idx = min((i_t + 1) * BT, T) - 1
b_g_last = tl.load(g + (bos * HV + last_idx * HV + i_h).to(tl.int64)).to(tl.float32)
p_g = g + (bos * HV + i_h).to(tl.int64) + o_t * HV
b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
b_g_last = exp2(b_g_last)
b_h1 *= b_g_last
b_h2 *= b_g_last

b_v = b_v.to(k.dtype.element_ty)

p_k = k + o_k1[:, None] + o_t[None, :] * (H*K)
b_k = tl.load(p_k, mask=m_k1[:, None] & m_t[None, :], other=0.0)
b_h1 += tl.dot(b_k, b_v)
p_k = k + o_k2[:, None] + o_t[None, :] * (H*K)
b_k = tl.load(p_k, mask=m_k2[:, None] & m_t[None, :], other=0.0)
b_h2 += tl.dot(b_k, b_v)
```

- `last_idx` 取当前 chunk 的最后一个有效 token；尾块不足 64 个 token 时，衰减终点随有效长度缩短。`exp2(b_g_last - b_g)` 对残差施加位置到块末的衰减，`exp2(b_g_last)` 则整体缩放入口状态。随后两次 `tl.dot(b_k, b_v)` 分别将同一份残差写入两片状态，组成下一块的入口状态。

- `v_new` 必须在乘块末衰减之前保存，因为块内输出需要按每个 query 的位置计算区间衰减。状态寄存器在循环内持续更新；循环结束后，按 `STORE_FINAL_STATE` 写回最终状态 `ht`。packed 输入用 `cu_seqlens` 定位 token，用 `chunk_offsets` 定位块入口状态，各序列独立递推。

- **状态缓存的生命周期。** `h` 保存每个 chunk 更新前的状态，`final_state` 只保存整条序列处理完后的状态；不能用块末状态代替同一块的入口状态，否则输出会重复计入当前 chunk 的写入。状态在 kernel 内以 FP32 寄存器累积，保存给输出 kernel 的 `h` 使用 K 的 dtype，最终 `ht` 使用 FP32；矩阵乘法前还会按操作数 dtype 转换寄存器片段。

- 对 packed 输入，物理 batch 通常为 1，逻辑序列由 `cu_seqlens` 分开。`chunk_indices` 将全局 chunk 编号映射到 `(序列编号, 序列内 chunk 编号)`；`chunk_offsets` 是每条序列 chunk 数的前缀和，用于定位状态缓存。每条序列分别加载初态、重置累计门控并写回末态，不能让短序列的尾块与下一条序列合并成一个 chunk。

### 块内输出：历史项与局部项

- `fla/ops/common/chunk_o.py` 的 `chunk_fwd_kernel_o` 读取块入口状态与 `v_new`，实现论文输出式中的两项。将局部项的区间衰减显式写出，并保留代码中的缩放 $`s=d_k^{-1/2}`$：

```math
\mathbf O_{[t]}=s\left[
\overleftarrow{\mathbf Q}_{[t]}\mathbf S_{[t]}^\top+
\left(\mathbf Q_{[t]}\mathbf K_{[t]}^\top\odot\Gamma_{[t]}\right)
\left(\widetilde{\mathbf U}_{[t]}-
\overleftarrow{\mathbf W}_{[t]}\mathbf S_{[t]}^\top\right)
\right].
```

- 网格为 `(ceil(V / BV), NT, B * HV)`。每个 program 处理一个 chunk、一个 value head 和一片 value 通道；各块入口状态已在上一阶段生成，因此此处可以按 chunk 并行。以下两段连续摘录 scalar gate 分支，指针基址已按当前序列、head 和 chunk 定位。

- **沿 K 维累计两个矩阵乘积。** 两个 FP32 累加器分别保存尚未施加衰减的历史读出和块内 QK 权重：

```math
\mathtt{b\_o}\ \longleftrightarrow\;
\mathbf Q_{[t]}\mathbf S_{[t]}^\top,
\qquad
\mathtt{b\_A}\ \longleftrightarrow\;
\mathbf Q_{[t]}\mathbf K_{[t]}^\top.
```

```python
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_o(
    q, k, v, h, g, g_gamma, o, cu_seqlens, chunk_indices, scale, T, H: tl.constexpr,
    HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
    BV: tl.constexpr, USE_G: tl.constexpr, USE_G_GAMMA: tl.constexpr, STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    ...
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_v = i_v * BV + tl.arange(0, BV)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_q = q + o_t[:, None] * (H*K) + o_k[None, :]
        p_k = k + o_k[:, None] + o_t[None, :] * (H*K)
        p_h = h + o_k[:, None] * V + o_v[None, :]
        m_h = m_k[:, None] & (o_v[None, :] < V)
        # [BT, BK]
        b_q = tl.load(p_q, mask=m_t[:, None] & m_k[None, :], other=0.0)
        # [BK, BT]
        b_k = tl.load(p_k, mask=m_k[:, None] & m_t[None, :], other=0.0)
        b_h = tl.load(p_h, mask=m_h, other=0.0)

        # [BT, BK] @ [BK, BV] -> [BT, BV]
        b_o += tl.dot(b_q, b_h)
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_A += tl.dot(b_q, b_k)
```

- Q 的片段为 `[BT, BK]`，K 按 `[BK, BT]` 读取，入口状态片段为 `[BK, BV]`。同一份 `b_q` 同时参与两个 `tl.dot`；沿 K 维累加后，分别得到 `[BT, BV]` 的历史项与 `[BT, BT]` 的局部权重。这里的 `b_A` 是 QK 权重，与三角求解中的 A 无关。

- **施加区间衰减、因果掩码并合成输出。** 历史项的第 r 行乘 $`\gamma_{[t]}^r`$，局部权重的第 (r,i) 项乘 $`\gamma_{[t]}^r/\gamma_{[t]}^i`$ 并保留 $`i\le r`$，随后与残差相乘：

```math
\mathtt{b\_o}\leftarrow\overleftarrow{\mathbf Q}_{[t]}\mathbf S_{[t]}^\top,\qquad \mathtt{b\_A}\leftarrow\mathbf Q_{[t]}\mathbf K_{[t]}^\top\odot\Gamma_{[t]},\qquad \mathbf O_{[t]}=s(\mathtt{b\_o}+\mathtt{b\_A}\mathbf V_{\mathrm{new},[t]}).
```

```python
g += bos * HV + i_h
p_g = g + o_t * HV
b_g = tl.load(p_g, mask=m_t, other=0.0)
b_o = b_o * exp2(b_g)[:, None]
b_A = b_A * exp2(b_g[:, None] - b_g[None, :])
m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
b_A = tl.where(m_A, b_A, 0)

p_v = v + o_t[:, None] * (HV*V) + o_v[None, :]
p_o = o + o_t[:, None] * (HV*V) + o_v[None, :]

b_v = tl.load(p_v, mask=m_t[:, None] & (o_v < V)[None, :], other=0.0)
# to fix mma -> mma layout conversion
# already solved by triton v3.2 or higher
b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_t[:, None] & (o_v < V)[None, :])
```

- `exp2(b_g)` 将历史读出衰减到各 query 位置；`exp2(b_g[:, None] - b_g[None, :])` 对应 $`\Gamma_{[t]}`$ 的非零元素。这里的掩码使用 `>=`，包含对角线，使当前 token 读到本步写入；KKT 的掩码使用 `>`，只表示此前写入对当前残差的影响。

- 参数 `v` 传入上一阶段保存的 `v_new`。最后一次 `tl.dot` 汇总块内残差，并与历史项相加、乘 `scale` 后写回输出。变长输入通过全局 chunk 编号 `i_tg` 读取入口状态，通过序列内 chunk 编号 `i_t` 读取 Q/K/V，二者不能混用。

### 反向传播：状态伴随与三角变换梯度

- 输出计算与块末状态都依赖 $`\mathbf V_{\mathrm{new},[t]}`$，因此残差梯度由两条路径相加：

```math
\frac{\partial\mathcal L}{\partial\mathbf V_{\mathrm{new},[t]}}
=s\left(\mathbf Q_{[t]}\mathbf K_{[t]}^\top\odot\Gamma_{[t]}\right)^\top
\frac{\partial\mathcal L}{\partial\mathbf O_{[t]}}
+\overrightarrow{\mathbf K}_{[t]}
\left(\frac{\partial\mathcal L}{\partial\mathbf S_{[t+1]}}\right)^\top.
```

- `chunk_bwd_dv_local` 计算上式第一项；`chunk_gated_delta_rule_bwd_dhu` 加入后续状态贡献，并从后向前递推状态梯度。`chunk_bwd_dqkwg` 处理输出与状态更新对 Q/K/W 和 gate 的梯度；`prepare_wy_repr_bwd` 再通过三角变换将 W/U 梯度传回 K/V、beta 和 gate。

- **状态梯度为什么要倒序递推。** 省略块编号，令 $`\mathbf H=\mathbf S_{[t]}^\top`$、$`\mathbf H_+=\mathbf S_{[t+1]}^\top`$、$`\mathbf E=\mathbf V_{\mathrm{new},[t]}`$，并记 $`\mathbf C=\operatorname{diag}(\gamma_C/\gamma)`$、$`\mathbf P=s(\mathbf Q\mathbf K^\top\odot\Gamma)`$。对任意中间量 X，用 $`\overline{\mathbf X}=\partial\mathcal L/\partial\mathbf X`$ 表示同形状梯度。前向是 $`\mathbf E=\widetilde{\mathbf U}-\overleftarrow{\mathbf W}\mathbf H`$、$`\mathbf H_+=\gamma_C\mathbf H+\mathbf K^\top\mathbf C\mathbf E`$ 和 $`\mathbf O=s\mathbf D\mathbf Q\mathbf H+\mathbf P\mathbf E`$，逐项求导得到：

```math
\overline{\mathbf E}=\mathbf P^\top\overline{\mathbf O}+\mathbf C\mathbf K\overline{\mathbf H}_+,\qquad \overline{\widetilde{\mathbf U}}=\overline{\mathbf E},\qquad \overline{\overleftarrow{\mathbf W}}=-\overline{\mathbf E}\mathbf H^\top.
```

```math
\overline{\mathbf H}=\gamma_C\overline{\mathbf H}_++s\mathbf Q^\top\mathbf D\overline{\mathbf O}-\overleftarrow{\mathbf W}^\top\overline{\mathbf E}.
```

- 状态梯度的三项分别来自块末状态的整体衰减、当前块输出对历史的直接读取，以及残差中减去的历史预测。最后一项的负号不能遗漏。`chunk_bwd_dv_local` 先按 chunk 并行计算 $`\mathbf P^\top\overline{\mathbf O}`$，`chunk_gated_delta_rule_bwd_dhu` 再从最后一块向前加入状态路径的贡献。最后一块从 `dht` 开始；未提供 `dht` 时为零，全部块处理完后得到 `dh0`。

- 在 `STATE_V_FIRST=False` 的源码分支中，K 的各片段先累加出 `b_dv`，再乘位置到块末的衰减。下面是残差梯度合并和第一片状态梯度的更新；其他 K 片段以同样方式累加：

```python
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64(
    q, k, w, g, gk, dht, dh0, do, dh, dv, dv2, cu_seqlens, chunk_offsets, scale, T,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
    BV: tl.constexpr, USE_G: tl.constexpr, USE_GK: tl.constexpr, USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr, STATE_V_FIRST: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    ...
    b_dv *= tl.where(m_t, exp2(bg_last - b_g), 0)[:, None]
    b_dv += tl.load(p_dv, mask=m_t[:, None] & m_v[None, :], other=0.0)
    tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])
    b_dh1 *= bg_last_exp
    b_q = b_q * b_g_exp[None, :]
    b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
```

- **Q/K 梯度的直接路径与间接路径。** 暂时固定 U、W 和 gate，输出与状态更新直接产生的 Q/K 梯度为：

```math
\overline{\mathbf Q}_{\mathrm{direct}}=s\mathbf D\overline{\mathbf O}\mathbf H^\top+s\left(\overline{\mathbf O}\mathbf E^\top\odot\Gamma\right)\mathbf K.
```

```math
\overline{\mathbf K}_{\mathrm{direct}}=s\left(\overline{\mathbf O}\mathbf E^\top\odot\Gamma\right)^\top\mathbf Q+\mathbf C\mathbf E\overline{\mathbf H}_+^\top.
```

- `chunk_bwd_dqkwg` 计算这些直接路径，并生成 W 和 gate 的梯度；K 还通过 Gram 矩阵以及 W 的右端项影响 U/W，这部分必须由 `prepare_wy_repr_bwd` 加回。因而 `dk.add_(dk2)` 是同一个输入经两条计算路径的梯度求和。

- **三角逆如何求导。** 令 $`\mathbf R=(\mathbf I+\mathbf A)^{-1}`$、$`\mathbf X=\mathbf B_\beta\mathbf V`$、$`\mathbf Y=\mathbf B_\beta\mathbf D\mathbf K`$，于是 $`\widetilde{\mathbf U}=\mathbf R\mathbf X`$、$`\overleftarrow{\mathbf W}=\mathbf R\mathbf Y`$。先对两个矩阵乘法求导，再利用 $`d\mathbf R=-\mathbf R(d\mathbf A)\mathbf R`$：

```math
\overline{\mathbf R}=\overline{\widetilde{\mathbf U}}\mathbf X^\top+\overline{\overleftarrow{\mathbf W}}\mathbf Y^\top,\qquad \overline{\mathbf X}=\mathbf R^\top\overline{\widetilde{\mathbf U}},\qquad \overline{\mathbf Y}=\mathbf R^\top\overline{\overleftarrow{\mathbf W}}.
```

```math
\overline{\mathbf A}=\operatorname{strictLower}\!\left(-\mathbf R^\top\overline{\mathbf R}\mathbf R^\top\right).
```

- `prepare_wy_repr_bwd_kernel` 通过交换行列地址将代码变量 `b_A` 加载为 $`\mathbf R^\top`$，先累加两个乘法对逆矩阵的梯度，再执行两次 `tl.dot`。单位对角线是常数，上三角恒为零；只有严格下三角系数需要回传。以下摘录中最后乘入的指数差把梯度继续传过 $`\Gamma\odot\mathbf K\mathbf K^\top`$ 的逐元素衰减：

```python
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_kernel(
    k, v, beta, g, A, dw, du, dk, dv, db, dg, cu_seqlens, chunk_indices, T, H: tl.constexpr,
    HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
    BV: tl.constexpr, USE_G: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    ...
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA, 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
    b_dA *= exp2(b_g[:, None] - b_g[None, :])
    b_dA = tl.where(m_A, -b_dA, 0).to(k.dtype.element_ty)
```

- **beta、原始 V 和 K 的梯度。** 记 $`\mathbf J=\mathbf K\mathbf K^\top`$、$`\mathbf F=\mathbf B_\beta(\overline{\mathbf A}\odot\Gamma)`$，$`\langle\cdot,\cdot\rangle`$ 表示同一行向量的内积。右端项与三角系数共同贡献：

```math
\overline{\mathbf V}=\mathbf B_\beta\overline{\mathbf X},\qquad \overline{\mathbf K}_{\mathrm{WY}}=\mathbf D\mathbf B_\beta\overline{\mathbf Y}+(\mathbf F+\mathbf F^\top)\mathbf K.
```

```math
\overline\beta_i=\langle\overline{\mathbf X}_i,\mathbf V_i\rangle+\gamma_i\langle\overline{\mathbf Y}_i,\mathbf K_i\rangle+\sum_{j<i}\overline A_{ij}\Gamma_{ij}J_{ij}.
```

- K 同时出现在 Gram 矩阵的左右两侧，所以间接梯度含 $`\mathbf F+\mathbf F^\top`$。`beta` 也同时出现在右端缩放和三角系数中，不能只对 `v * beta` 求导。共享 Q/K head 时，各 value head 先计算自己的贡献，再沿组内 head 维求和；源码最后的 `dk.view(B, T, H, HV // H, K).sum(3)` 恢复原始 K 的形状。

- **gate 梯度与最后一次反向前缀和。** 令 $`\ell_i=\sum_{j\le i}g_j=\ln\gamma_i`$ 为自然对数累计 gate，$`Z_{ij}=\overline A_{ij}A_{ij}`$。仅来自 WY 变换的累计 gate 梯度为：

```math
\overline\ell_i^{\mathrm{WY}}=\langle\overline{\mathbf Y}_i,\mathbf Y_i\rangle+\sum_j Z_{ij}-\sum_j Z_{ji},\qquad \frac{\partial\mathcal L}{\partial g_i}=\sum_{r\ge i}\frac{\partial\mathcal L}{\partial\ell_r}.
```

- 行和减列和来自 $`\Gamma_{ij}=e^{\ell_i-\ell_j}`$：行位置贡献正号、列位置贡献负号。`chunk_bwd_dqkwg` 另行计算输出及状态传播中的 gate 梯度，两路相加后再做 chunk 内的后缀和。源码虽然前向存储 $`\widehat g=\ell/\ln2`$ 并调用 `exp2`，手写 backward 返回的是相对于 $`\ell`$ 的梯度，因此末尾的 `chunk_local_cumsum(..., reverse=True)` 不再乘 `RCP_LN2`；这与直接对 `exp2` 自动求导时出现的 $`\ln2`$ 因子相互抵消。

- 前向保存归一化后的 Q/K、原始 V、累计 gate、beta、三角逆 A、初态和序列索引，反向重算 W/U、块入口状态和 `v_new`。若启用了 kernel 内 Q/K 归一化，最外层还要调用 `l2norm_bwd`；融合 beta sigmoid 时再经过 `fused_beta_sigmoid_bwd`，融合衰减门时则由 `gdn_gate_bwd` 回传到 gate 输入、`A_log` 和 `dt_bias`。状态算子的梯度到达这些外层变换后，才是模型投影输出所需的梯度。

```python
def chunk_gated_delta_rule_bwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
    A: torch.Tensor, scale: float, initial_state: torch.Tensor, do: torch.Tensor, dht: torch.Tensor,
    state_v_first: bool = False, cu_seqlens: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None, chunk_indices: torch.LongTensor | None = None,
    use_gate_in_kernel: bool = False, g_input: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None, dt_bias: torch.Tensor | None = None, chunk_size: int = 64,
):
    w, u = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=A, g=g, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )
    if cp_context is not None:
        initial_state = expand_h0(initial_state, context=cp_context)
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g, initial_state=initial_state, output_final_state=False,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, state_v_first=state_v_first,
        chunk_size=chunk_size,
    )
    dv = chunk_bwd_dv_local(
        q=q, k=k, g=g, do=do, scale=scale, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    if cp_context is not None:
        dht, initial_state = chunk_gated_delta_rule_bwd_dhu_pre_process(
            q=q, k=k, w=w, do=do, dv=dv, g=g, scale=scale, cu_seqlens=cu_seqlens, dht=dht,
            initial_state=initial_state, context=cp_context, state_v_first=state_v_first,
            chunk_size=chunk_size,
        )
    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=q, k=k, w=w, g=g, h0=initial_state, dht=dht, do=do, dv=dv, scale=scale,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, state_v_first=state_v_first,
        chunk_size=chunk_size,
    )
    dq, dk, dw, dg = chunk_bwd_dqkwg(
        q=q, k=k, v=v_new, w=w, g=g, h=h, dv=dv, do=do, dh=dh, scale=scale, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices, state_v_first=state_v_first, chunk_size=chunk_size,
    )
    dk2, dv, db, dg2 = prepare_wy_repr_bwd(
        k=k, v=v, beta=beta, g=g, A=A, dw=dw, du=dv, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    dk.add_(dk2)
    dg.add_(dg2)
    dg = chunk_local_cumsum(
        dg, chunk_size=chunk_size, reverse=True, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )
    dA_log, ddt_bias = (None, None)
    if use_gate_in_kernel:
        dg, dA_log, ddt_bias = gdn_gate_bwd(g=g_input, A_log=A_log, dt_bias=dt_bias, dyg=dg)
    return (dq, dk, dv, db, dg, dh0, dA_log, ddt_bias)
```

- **并行化的代价。** 对单个 value head，块内 KKT 与三角变换的主要矩阵乘法开销随 $`T\,BT\,(d_k+d_v)`$ 增长，状态递推与读出随 $`T\,d_kd_v`$ 增长；固定 `BT` 后均随序列长度线性增长。显式三角求逆的分块计算还约有 $`(T/BT)\,BT^3`$ 的算术开销，`BT=64` 时是固定尺寸的块内工作。不同 chunk 的三角系统可并行求解，同一序列的状态仍顺序传递。

- A 占用约 `B * T * HV * BT` 个元素，入口状态缓存 `h` 占用约 `B * ceil(T / BT) * HV * K * V` 个元素。重算策略减少需要跨前向、反向长期保留的 W/U、h 和残差，但反向计算时仍要重新分配这些临时量；chunk 大小同时影响并行粒度、三角求解成本和状态缓存大小。

## Recurrent 算法代码解析

### 状态分块与并行网格

- Recurrent kernel 将每个序列、每个 value head 的状态沿 value 维划分。对于 scalar gate 分支，线程块尺寸与输出状态缓冲区为：

```math
\begin{aligned}
B_K&=2^{\lceil\log_2 d_k\rceil},\qquad
B_V=\min\!\left(8,2^{\lceil\log_2 d_v\rceil}\right),\\
N_V&=\left\lceil d_v/B_V\right\rceil,\qquad
\mathrm{grid}=(N_V,NH_V),\\
\mathcal H_{\mathrm{final}}&\in\mathbb R^{N\times H_V\times d_k\times d_v}.
\end{aligned}
```

- 每个 program 维护一个 `[BK, BV]` 状态片段，并沿时间维串行更新。不同 value 片段可以独立推进，因为每个片段只需要完整 key 向量和对应的 value 子向量。较小的 BV 限制单个 program 的状态寄存器用量；最终状态使用 FP32 保存，供下一次调用接续。

```python
def fused_recurrent_gated_delta_rule_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None, gv: torch.Tensor | None = None,
    beta: torch.Tensor | None = None, A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None, scale: float = None, initial_state: torch.Tensor = None,
    output_final_state: bool = False, use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False, allow_neg_eigval: bool = False,
    state_v_first: bool = False, cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = (*k.shape, v.shape[-1])
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    BV = min(8, triton.next_power_of_2(V)) if gv is None else triton.next_power_of_2(V)
    NV = triton.cdiv(V, BV)
    o = torch.empty_like(v)
    if output_final_state:
        if state_v_first:
            final_state = q.new_empty(N, HV, V, K, dtype=torch.float32)
        else:
            final_state = q.new_empty(N, HV, K, V, dtype=torch.float32)
    else:
        final_state = None
    grid = (NV, N * HV)
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q, k=k, v=v, g=g, gk=gk, gv=gv, beta=beta, A_log=A_log, dt_bias=dt_bias, o=o,
        h0=initial_state, ht=final_state, cu_seqlens=cu_seqlens, scale=scale, T=T, H=H, HV=HV, K=K,
        V=V, BK=BK, BV=BV, IS_BETA_HEADWISE=beta.ndim != v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        APPLY_BETA_SIGMOID=use_beta_sigmoid_in_kernel, ALLOW_NEG_EIGVAL=allow_neg_eigval,
        STATE_V_FIRST=state_v_first, num_warps=1, num_stages=3,
    )
    return (o, final_state)
```

- 这里 N 为序列数，$`H_V`$ 为 value head 数。Q/K head 通过 `i_hv // (HV // H)` 映射，同一组 value head 共享 Q/K 输入，但各自维护独立的状态。prefill 写出的 chunk 末态与此处初始状态使用同一种缓冲区排列，二者之间不需要重新组织数学状态。

### 逐 Token 更新：衰减、残差与读出

- 以代码中的状态矩阵 $`H_t=S_t^\top`$ 表示递推。包括 Q/K 归一化在内，一步更新由以下算式组成：

```math
\begin{aligned}
q_t&\leftarrow q_t/\sqrt{\|q_t\|_2^2+\varepsilon},\qquad
k_t\leftarrow k_t/\sqrt{\|k_t\|_2^2+\varepsilon},\\
\bar H_t&=\exp(g_t)H_{t-1},\\
e_t&=\beta_t\left(v_t-\bar H_t^\top k_t\right),\\
H_t&=\bar H_t+k_te_t^\top,\\
o_t&=sH_t^\top q_t.
\end{aligned}
```

- 对 K 维的归约实现 $`\bar H_t^\top k_t`$，读出衰减后的旧 value；外积 $`k_te_t^\top`$ 将残差写入当前 key 方向；最后的归约实现状态对 query 的读出。完整 key 维包含在每个 program 内，因此这些归约不需要跨 program 通信。

```python
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q, k, v, g, gk, gv, beta, A_log, dt_bias, o, h0, ht, cu_seqlens, scale, T, H: tl.constexpr,
    HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_G: tl.constexpr, USE_GK: tl.constexpr, USE_GV: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr, IS_BETA_HEADWISE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr, STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr, USE_GATE_IN_KERNEL: tl.constexpr, HAS_DT_BIAS: tl.constexpr,
    APPLY_BETA_SIGMOID: tl.constexpr, ALLOW_NEG_EIGVAL: tl.constexpr,
):
    ...
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in tl.range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q /= tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k /= tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q *= scale
        b_beta = tl.load(p_beta).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)

        b_h *= exp(b_g)
        b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
        b_h += b_k[:, None] * b_v
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H * K
        p_k += H * K
        p_v += HV * V
        p_o += HV * V
        p_g += HV
        p_beta += HV

    if STORE_FINAL_STATE:
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
```

- 代码在循环前加载初始状态，在循环后写回末态。逐 token 解码时通常只有一次迭代，Q/K/V 指针按各自的 head 数前进，gate 和 beta 则按 value head 前进。状态更新发生在寄存器中，输出与末态分别写回对应缓冲区。

- 该循环由逐元素乘法、外积和归约组成，没有矩阵乘法指令。Chunk 将多个时间步的依赖组织为三角求解与矩阵乘法，以提高整段序列计算的并行度；recurrent 则直接执行单步状态转移。当前 fused recurrent 实现未提供 backward，训练由 chunk 路径完成。

- 在数学上令 chunk 长度为 1，严格下三角系数为零，代码中保存的逆矩阵 $`A=I`$。此时 $`u_t=\beta_t v_t`$、$`w_t=\beta_t\alpha_t k_t`$，还需扣除入口状态的预测，才得到 $`e_t=u_t-H_{t-1}^\top w_t`$；随后按同一条递推更新状态并读出。因此 chunk 的末态可以直接作为 recurrent 的初始状态，prefill 与解码之间保持同一个状态递推关系。

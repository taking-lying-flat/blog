# Gated DeltaNet：模型调用、Chunk 与 Recurrent 实现

## 1. GDN 与 Token Mixer

- 标准因果注意力先构造 token 两两之间的相似度，再经过 softmax 归一化。以下采用不含 softmax 的点积形式表示线性注意力，其历史键值对可累积到固定大小的状态矩阵。设 Q/K/V 按 token 排成行，$`M_{ij}=\mathbf 1_{j\le i}`$ 为二值因果掩码，$`\mathcal M_{ij}`$ 在可见位置为 0、不可见位置为 $`-\infty`$，两种输出分别为：

```math
O_{\mathrm{attn}}=\operatorname{softmax}(sQK^\top+\mathcal M)V,
\qquad O_{\mathrm{linear}}=s(QK^\top\odot M)V,
\qquad s=d_k^{-1/2}.
```

- 上述并行表达式包含序列长度平方数量的成对权重。FlashAttention 可通过分块与融合避免完整权重矩阵的显存往返，但成对计算量仍随序列长度平方增长。线性注意力的优势来自其等价递推：每个 head 维护 $`d_v\times d_k`$ 的状态，<span style="white-space: nowrap;">单步更新与读出均为 $`O(d_kd_v)`$。</span>因果掩码不能直接移到矩阵乘法外，因此训练时需要 chunkwise 分解，将块内计算组织为矩阵乘法，将块间依赖压缩到状态传递。

- 状态通过外积将多组 key–value 关联叠加到同一个矩阵中。用某个 key 读取时，其他 key 对应的 value 也会按两者的内积参与结果：内积为零时没有这项干扰，内积非零时读出就会混入其他 value。GDN 先用 $`\alpha_t`$ 缩放整个旧状态，再用当前 key 读取旧 value 的预测；将新 value 与该预测作差，乘以 $`\beta_t`$ 后沿当前 key 方向写回。因此，$`\alpha_t`$ 控制历史信息的整体衰减，$`\beta_t`$ 控制本次预测误差的修正幅度。输出门则作用于状态读出之后，只调节传给下一层的结果，不改写状态。

<figure style="width: 100%; max-width: 600px; margin: 20px auto;">
  <a href="../../assets/gdn-architecture.png"><img src="../../assets/gdn-architecture.png" alt="Gated DeltaNet 模型结构与 token mixer" width="2058" height="1122"></a>
</figure>

- Token mixer 的 Q/K/V 分支依次经过线性投影、逐通道短因果卷积和 SiLU，Q/K 再沿 head 维做 L2 归一化。短卷积提供局部时序特征，Gated Delta Rule 维护跨 token 的矩阵状态；两者在解码时分别保留卷积窗口与 recurrent state。

- 衰减门与写入系数由独立投影生成，不经过 Q/K/V 的短卷积。状态输出先按 head 归一化，再与输出门逐元素相乘，最后投影回隐藏维度。

## 2. Gated Delta Rule 的计算公式

### 2.1 线性注意力与标量衰减

- 对单个 head，令 $`q_t,k_t\in\mathbb R^{d_k}`$、$`v_t\in\mathbb R^{d_v}`$，状态 $`S_t\in\mathbb R^{d_v\times d_k}`$。外积 $`v_tk_t^\top`$ 写入键值关联，矩阵向量乘法 $`S_tq_t`$ 读取关联。论文 §2.1 的线性注意力与 Mamba2 标量衰减形式分别为：

```math
\begin{aligned}
\text{Linear Attention:}\quad&S_t=S_{t-1}+v_tk_t^\top,\\
\text{Scalar decay:}\quad&S_t=\alpha_tS_{t-1}+v_tk_t^\top,
\qquad o_t=sS_tq_t.
\end{aligned}
```

- 令 $`\gamma_t=\prod_{r=1}^{t}\alpha_r`$。在 $`S_0=0`$ 时，递推展开为带区间衰减的键值累加，输出仍是对历史 value 的加权求和：

```math
S_t=\sum_{j=1}^{t}\frac{\gamma_t}{\gamma_j}v_jk_j^\top,
\qquad o_t=s\sum_{j=1}^{t}\frac{\gamma_t}{\gamma_j}(k_j^\top q_t)v_j.
```

- 系数 $`\gamma_t/\gamma_j=\prod_{r=j+1}^{t}\alpha_r`$ 表示位置 j 写入的信息传播到位置 t 后的保留比例。若存在初始状态，还需加上 $`\gamma_tS_0`$。标量门会同时缩放所有 key 方向，无法单独修正某个 key 对应的旧 value。

### 2.2 Delta 更新与 Gated Delta Rule

- DeltaNet 先用当前 key 从旧状态读取预测值，再写入新旧 value 的差。GDN 将读取位置改为衰减后的状态，对应论文 Eq. (10)。在 sigmoid 写入门与标量衰减的参数化下，$`\alpha_t,\beta_t\in(0,1)`$：

```math
\begin{aligned}
\text{DeltaNet:}\quad
S_t&=S_{t-1}+\beta_t(v_t-S_{t-1}k_t)k_t^\top,\\
\text{GDN:}\quad
S_t&=\alpha_tS_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top\\
&=\alpha_tS_{t-1}+\beta_t(v_t-\alpha_tS_{t-1}k_t)k_t^\top.
\end{aligned}
```

- 当 $`\|k_t\|_2=1`$ 时，当前 key 上的读出是旧值与新值的插值；对任意正交方向 $`x\perp k_t`$，Delta 项为零，只保留整体衰减：

```math
S_tk_t=(1-\beta_t)\alpha_tS_{t-1}k_t+\beta_tv_t,
\qquad S_tx=\alpha_tS_{t-1}x\quad(k_t^\top x=0).
```

- 因而 $`\beta_t=1`$ 在单位 key 方向上完成替换，$`\beta_t=0`$ 只执行衰减；令 $`\alpha_t=1`$ 则退化为 DeltaNet。与当前 key 非正交的其他关联仍可能受到更新影响，不能将“定向更新”理解为所有其他记忆都不变。

### 2.3 在线回归视角

- 将状态视为一个从 key 映射到 value 的线性模型，记衰减后的参考状态为 $`\bar S_t=\alpha_tS_{t-1}`$。GDN 等价于从该参考状态出发，对当前键值回归损失做一步梯度更新：

```math
\mathcal L_t(S)=\tfrac12\|Sk_t-v_t\|_2^2,
\qquad S_t=\bar S_t-\beta_t\nabla\mathcal L_t(\bar S_t)
=\bar S_t+\beta_t(v_t-\bar S_tk_t)k_t^\top.
```

- 论文 Table 1 将同一更新写为一个在线目标的闭式解。令 $`r_t=\beta_t(v_t-\bar S_tk_t)`$，将本步残差视为固定量，则：

```math
S_t=\underset{S}{\arg\min}\;
\left\{\tfrac12\|S-\bar S_t\|_F^2-\langle Sk_t,r_t\rangle\right\}
=\bar S_t+r_tk_t^\top.
```

- 第一项约束新状态偏离衰减参考点的程度，第二项沿当前 key 方向写入残差。这里更新的是每条序列的运行状态；$`\beta_t`$ 决定本次状态修正的幅度，$`\alpha_t`$ 决定进入该更新前保留多少历史。

## 3. Chunk 的矩阵表示

### 3.1 DeltaNet 的 WY 表示与 UT 变换

- 将序列划分为长度 C 的块，块内 Q/K/V 按 token 排成行。$`\mathbf S_{[t]}`$ 为第 t 块的入口状态，$`\mathbf S_{[t]}^r`$ 为处理块内前 r 个 token 后的状态。先考虑不含衰减的 DeltaNet，连续转移矩阵的乘积具有论文 Eq. (4) 的 WY 表示：

```math
\mathbf P_{[t]}^r
=\prod_{i=1}^{r}\left(\mathbf I-\beta_{[t]}^i\boldsymbol k_{[t]}^i\boldsymbol k_{[t]}^{i\top}\right)
=\mathbf I-\sum_{i=1}^{r}\boldsymbol w_{[t]}^i\boldsymbol k_{[t]}^{i\top}.
```

- 其中 $`\boldsymbol w_{[t]}^r=\beta_{[t]}^r(\boldsymbol k_{[t]}^r-\sum_{i<r}\boldsymbol w_{[t]}^i(\boldsymbol k_{[t]}^{i\top}\boldsymbol k_{[t]}^r))`$。新增状态同样可以写成 $`\sum_{i=1}^r\boldsymbol u_{[t]}^i\boldsymbol k_{[t]}^{i\top}`$，U 的递推只需将 W 递推右端的当前 key 换成 value。两者共享此前 key 内积形成的下三角系数。

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

- WY 表示把转移矩阵的连乘改写为低秩更新的累加；UT 变换进一步将 W/U 的计算组织成三角求解与矩阵乘法。求解内部的因果依赖仍然存在，第 5.4 节通过 16×16 局部前代与块间合并实现这一过程。

### 3.2 Gated DeltaNet 的衰减与扩展 WY 表示

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

- 与 DeltaNet 相比，三角系统中的 Gram 矩阵由 $`\mathbf K_{[t]}\mathbf K_{[t]}^\top`$ 改为 $`\Gamma_{[t]}\odot\mathbf K_{[t]}\mathbf K_{[t]}^\top`$。第 5.3 节构造这一系数，第 5.4 节求逆，第 5.5 节将逆变换应用于 K/V。

### 3.3 块间状态与块内输出

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

- 其中 W 为第 3.1 节无衰减 DeltaNet 的 WY 系数。块间状态更新先将入口状态衰减至块末，再累积当前块的残差写入；论文原式为：

```math
\mathbf S_{[t+1]}=\overrightarrow{\mathbf S}_{[t]}+
\left(\widetilde{\mathbf U}_{[t]}-
\overleftarrow{\mathbf W}_{[t]}\mathbf S_{[t]}^\top\right)^\top
\overrightarrow{\mathbf K}_{[t]}.
```

- 块内输出由历史读出与当前块内的残差贡献相加。[论文 §3.3](https://arxiv.org/html/2412.06464v3#S3.SS3) 将其写为：

```math
\mathbf O_{[t]}=\overleftarrow{\mathbf Q}_{[t]}\mathbf S_{[t]}^\top+
\left(\mathbf Q_{[t]}\mathbf K_{[t]}^\top\odot\mathbf M\right)
\left(\widetilde{\mathbf U}_{[t]}-
\overleftarrow{\mathbf W}_{[t]}\mathbf S_{[t]}^\top\right).
```

- 上式保留论文的 M 记号；若 M 仅表示二值因果掩码，局部项还缺少 $`\gamma_{[t]}^i/\gamma_{[t]}^j`$。FLA 显式乘入这一衰减，因此第 5.7 节对照代码时将该处写为 $`\Gamma_{[t]}`$，其余矩阵与箭头记号保持一致。

- W/U 可以按 chunk 并行生成；状态更新按 chunk 顺序推进；各块入口状态得到后，输出再次按 chunk 并行。块内残差 $`\widetilde{\mathbf U}_{[t]}-\overleftarrow{\mathbf W}_{[t]}\mathbf S_{[t]}^\top`$ 在状态更新与输出计算之间复用，分别对应第 5.6、5.7 节。

## 4. 模型调用方式：Qwen3.5 GatedDeltaNet

### 4.1 输入投影

- 设输入为 $`X\in\mathbb R^{T\times d_{\mathrm{model}}}`$。Q/K 的总投影维度为 $`d_Q=H_Kd_k`$，V 的总投影维度为 $`d_V=H_Vd_v`$。四个线性映射分别产生卷积输入、衰减参数、写入参数和输出门：

```math
P=XW_{qkv}^{\top},\qquad a=XW_a^{\top},\quad b=XW_b^{\top},\qquad Z=XW_z^{\top}.
```

- `Qwen3_5MoeGatedDeltaNet` 将 Q/K/V 合并为一次投影，随后按通道拆分。衰减参数和写入参数只需要每个 value head 一个标量，因此 `in_proj_a`、`in_proj_b` 的输出维度为 `num_v_heads`；输出门需要逐通道调节读出结果，其投影维度与 V 相同。

- **代码作用：建立输入投影。** 输入隐藏维度与 head 配置，创建 Q/K/V、衰减参数、写入参数和输出门的四组线性层。

```python
self.in_proj_qkv = nn.Linear(self.hidden_size, 2 * self.key_dim + self.value_dim, bias=False)
self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
```

### 4.2 短卷积、门控与 Head 映射

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

- `A_log` 和 `dt_bias` 是每个 value head 的可学习参数。上述参数化保证 `g` 为负值，从而将历史衰减限制在 0 与 1 之间；`beta` 经 sigmoid 控制残差写入强度。Grouped Value Attention 通过 head 映射使多个 value head 共享 Q/K，模型代码使用 `repeat_interleave` 显式实现该映射。

- **代码作用：生成状态算子的输入。** 输入 `hidden_states`，经过投影、短卷积、门控与 head 映射，输出 Q/K/V、g、beta 和输出门 z。

```python
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

- 有缓存的单 token 分支使用增量因果卷积维护短卷积窗口；它与 GDN 的矩阵状态分别保存。前者提供当前 token 的局部 Q/K/V 特征，后者累积更长范围的序列信息。

### 4.3 状态算子与输出门控

- Q/K 在进入状态更新前沿 head 维归一化。记 $`\mathcal D`$ 为 Gated Delta Rule 算子，则输入归一化、状态计算和输出映射可写为：

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

- Chunk 与 recurrent 实现同一个状态算子。模型根据输入长度和已有状态选择计算方式：完整序列使用 chunk；存在缓存且输入长度为 1 时使用 recurrent。两者均返回当前输出和序列末态，末态回写缓存，供后续输入接续计算。

- 输出归一化独立作用于每个 value head。`self.norm` 先执行 RMSNorm，再乘 `SiLU(z)`；head 维合并后，`out_proj` 将结果映射回模型隐藏维度。以下代码合并了两条分支中相同的调用参数。

- **代码作用：选择执行路径并回写缓存。** 输入 Q/K/V、门控及已有状态，调用 chunk 或 recurrent 算子，保存末态，再生成本层输出。

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

## 5. Chunk-wise 算法代码解析

### 5.1 前向计算的矩阵分解

- Chunk 前向围绕论文中的 $`\widetilde{\mathbf U}_{[t]}`$、$`\overleftarrow{\mathbf W}_{[t]}`$、$`\mathbf S_{[t]}`$ 和 $`\mathbf O_{[t]}`$ 展开。计算顺序为：累计门控、构造 KKT、求解下三角系统、生成 W/U、递推状态、计算输出。其中状态与输出复用同一个残差：

```math
\mathbf V_{\mathrm{new},[t]}=
\widetilde{\mathbf U}_{[t]}-
\overleftarrow{\mathbf W}_{[t]}\mathbf S_{[t]}^\top.
```

- `chunk_gated_delta_rule_fwd_intra` 构造并求解块内系统，生成逆矩阵 `A`、`u` 和 `w`；`chunk_gated_delta_rule_fwd_h` 引入块入口状态，计算 `v_new` 及下一块状态；`chunk_fwd_o` 完成输出计算。这里 `h` 保存所有块的入口状态，`final_state` 保存各序列的最终状态。

- 默认 64-token 路径将 KKT 构造与下三角求解融合执行，随后单独生成 W/U。块内变换可以跨 chunk 并行；状态更新仍按 chunk 顺序递推；入口状态生成后，各块输出又可以并行计算。下面保留固定长度、无 context parallelism 的调用主干。

- **代码作用：串联 chunk 前向阶段。** 输入 Q/K/V、逐 token gate、beta 与初始状态，依次生成累计 gate、W/U、块入口状态和最终输出。

```python
g = chunk_local_cumsum(g, chunk_size=64, scale=RCP_LN2)
w, u, A = chunk_gated_delta_rule_fwd_intra(
    k=k, v=v, g=g, beta=beta, chunk_size=64,
)
h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
    k=k, w=w, u=u, g=g,
    initial_state=initial_state,
    output_final_state=output_final_state,
    chunk_size=64,
)
o = chunk_fwd_o(
    q=q, k=k, v=v_new, h=h, g=g, scale=scale, chunk_size=64,
)
```

### 5.2 累积门控：将连乘转为前缀和

- 逐 token 的输入为 $`g_{[t]}^i=\log\alpha_{[t]}^i`$。对每个 chunk 独立求前缀和，即得到累计衰减的对数：

```math
\widehat g_{[t]}^r=\frac{1}{\ln 2}\sum_{i=1}^{r}g_{[t]}^i,
\qquad \gamma_{[t]}^r=2^{\widehat g_{[t]}^r},
\qquad \frac{\gamma_{[t]}^r}{\gamma_{[t]}^i}
=2^{\widehat g_{[t]}^r-\widehat g_{[t]}^i}.
```

- 算子位于 `fla/ops/utils/cumsum.py`，标量 kernel 的网格为 `(NT, B * H)`：每个 program 负责一个 chunk 和一个 gate head。`o_t` 构造块内 token 索引，`m_t` 屏蔽尾块越界位置；按 `[B, T, H]` 布局加载后，`tl.cumsum` 沿时间轴执行前缀和。累积在 FP32 中完成，并通过 `RCP_LN2` 将自然对数转换为以 2 为底的对数。后续 kernel 使用 `exp2` 还原整体衰减和区间衰减，避免在各阶段重复计算连乘。

- **代码作用：计算每个 chunk 的累计 gate。** 输入逐 token 对数衰减，输出 FP32 前缀和；GDN 前向再乘 `RCP_LN2`，供后续 `exp2` 使用。保留时间优先布局的源码分支。

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
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
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

### 5.3 KKT：构造块内递推系数

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

- 这是 16/32-token 分步路径使用的 kernel；64-token 默认路径将相同计算与下一节的求解融合。先展开独立 KKT，能够直接对应“内积、衰减、beta、严格下三角”四个矩阵操作。

- **代码作用：构造严格下三角系数。** 输入 K、累计 gate 和 beta，分片累计 key 内积并施加衰减与掩码，输出 `[B, T, HV, BT]` 系数缓冲区。

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
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
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

- `i_h // (HV // H)` 将 value head 映射到共享的 Q/K head。K 的时间步长为 `H * K`，gate 和 beta 的时间步长为 `HV`；因此 head 共享改变输入索引，但每个 value head 仍生成独立的块内系数。

- 变长输入通过 `chunk_indices[i_t]` 取得“序列编号、序列内 chunk 编号”，再由 `cu_seqlens` 确定 `bos` 和当前序列长度。`o_t < T` 限制实际 token 范围，`o_t[:, None] > o_t[None, :]` 只保留此前位置对当前残差的影响。结果按 `[B, T, HV, BT]` 写回，每个 token 保存其所在块的一行系数。

- 64-token 融合 kernel 位于 `fla/ops/gated_delta_rule/chunk_fwd.py`，名称为 `chunk_gated_delta_rule_fwd_kkt_solve_kernel`。它将时间维拆成四个 16-token 子块，只计算四个对角块和六个非对角下三角块；对角块额外应用严格下三角掩码。这十个子矩阵保留在寄存器中，直接进入前代与合并阶段。

### 5.4 下三角求解：局部前代与块间合并

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

- **第一段：16×16 对角块前代。** 以第一个对角块为例，`b_A00` 对应严格下三角子块 $`\mathbf A_{00}`$，`b_Ai00` 最终对应 $`\mathbf A^{\mathrm{inv}}_{00}=(\mathbf I+\mathbf A_{00})^{-1}`$。对块内元素，前代关系为：

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

- **第二段：合并相邻的 16×16 块。** 以零基索引表示四行四列子块。$`\mathbf A^{\mathrm{inv}}_{rs}`$ 表示整体逆矩阵的第 (r,s) 个子块，不是非对角块单独求逆。三个紧邻对角线的子块分别为：

```math
\begin{aligned}
\mathbf A^{\mathrm{inv}}_{10}&=-\mathbf A^{\mathrm{inv}}_{11}\mathbf A_{10}\mathbf A^{\mathrm{inv}}_{00},\\
\mathbf A^{\mathrm{inv}}_{21}&=-\mathbf A^{\mathrm{inv}}_{22}\mathbf A_{21}\mathbf A^{\mathrm{inv}}_{11},\\
\mathbf A^{\mathrm{inv}}_{32}&=-\mathbf A^{\mathrm{inv}}_{33}\mathbf A_{32}\mathbf A^{\mathrm{inv}}_{22}.
\end{aligned}
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

- **第三段：合并更远的非对角块。** 第 (2,0)、(3,1) 个子块依赖相邻块结果，左下角第 (3,0) 个子块进一步依赖第 (2,0) 个结果：

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

### 5.5 W/U：共享同一个三角变换

- `fla/ops/gated_delta_rule/wy_fast.py` 的 `recompute_w_u_fwd_kernel` 使用网格 `(NT, B * HV)`，每个 program 处理一个 chunk、一个 value head。读取求解后的逆矩阵 `b_A`，分别对 V、K 执行矩阵乘法，生成 $`\widetilde{\mathbf U}_{[t]}`$ 与 $`\overleftarrow{\mathbf W}_{[t]}`$。

- **第一段：计算 value 的 UT 变换。** 由于 $`\mathbf T_{[t]}`$ 的定义已经包含 beta，此处不再额外写一次 $`\operatorname{diag}(\beta_{[t]})`$：

```math
\widetilde{\mathbf U}_{[t]}
=\mathbf T_{[t]}\mathbf V_{[t]}
=(\mathbf I+\mathbf A_{[t]})^{-1}
\left(\operatorname{diag}(\beta_{[t]})\mathbf V_{[t]}\right).
```

```python
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

- **第二段：计算衰减后的 W。** 入口状态传播到位置 r 时需要乘 $`\gamma_{[t]}^r`$，因此 K 的右端项还需按累计衰减缩放：

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

### 5.6 块间状态：残差校正与衰减写入

- `fla/ops/common/chunk_delta_h.py` 的 `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` 实现论文的块间递推：

```math
\mathbf S_{[t+1]}=\overrightarrow{\mathbf S}_{[t]}+
\left(\widetilde{\mathbf U}_{[t]}-
\overleftarrow{\mathbf W}_{[t]}\mathbf S_{[t]}^\top\right)^\top
\overrightarrow{\mathbf K}_{[t]}.
```

- 网格为 `(ceil(V / BV), N * HV)`，每个 program 处理一条序列、一个 value head 和一片 value 通道，并沿 chunk 顺序递推。下面展开 scalar gate、K=128 的源码分支：`b_h1`、`b_h2` 分别保存 $`\mathbf S_{[t]}^\top`$ 的两个 `[64, BV]` 片段，初值从 `h0` 加载，无初始状态时置零。各基址已定位到当前序列与 head；`o_k1`、`o_k2` 分别覆盖 K 维的 0–63、64–127 通道。两段代码属于同一个 `for i_t in range(NT)` 循环。

- **第一段：保存入口状态并计算写入残差。** W 乘入口状态得到此前历史在当前块的预测，从 U 中扣除后即为实际残差：

```math
\mathbf V_{\mathrm{new},[t]}=
\widetilde{\mathbf U}_{[t]}-
\overleftarrow{\mathbf W}_{[t]}\mathbf S_{[t]}^\top.
```

```python
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

- **第二段：衰减至块末并累积状态。** 对残差的第 r 行乘 $`\gamma_{[t]}^C/\gamma_{[t]}^r`$，等价于将相同衰减乘入论文中的 $`\overrightarrow{\mathbf K}_{[t]}`$。转置后的状态更新为：

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

- `v_new` 必须在乘块末衰减之前保存，因为下一节需要按每个 query 的位置计算区间衰减。状态寄存器在循环内持续更新；循环结束后，按 `STORE_FINAL_STATE` 写回最终状态 `ht`。packed 输入用 `cu_seqlens` 定位 token，用 `chunk_offsets` 定位块入口状态，各序列独立递推。

### 5.7 块内输出：历史项与局部项

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

- **第一段：沿 K 维累计两个矩阵乘积。** 两个 FP32 累加器分别保存尚未施加衰减的历史读出和块内 QK 权重：

```math
\texttt{b\_o}\ \longleftrightarrow\;
\mathbf Q_{[t]}\mathbf S_{[t]}^\top,
\qquad
\texttt{b\_A}\ \longleftrightarrow\;
\mathbf Q_{[t]}\mathbf K_{[t]}^\top.
```

```python
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

- **第二段：施加区间衰减、因果掩码并合成输出。** 历史项的第 r 行乘 $`\gamma_{[t]}^r`$，局部权重的第 (r,i) 项乘 $`\gamma_{[t]}^r/\gamma_{[t]}^i`$ 并保留 $`i\le r`$，随后与残差相乘：

```math
\begin{aligned}
\texttt{b\_o}&\ \longleftarrow\;
\overleftarrow{\mathbf Q}_{[t]}\mathbf S_{[t]}^\top,\\
\texttt{b\_A}&\ \longleftarrow\;
\mathbf Q_{[t]}\mathbf K_{[t]}^\top\odot\Gamma_{[t]},\\
\mathbf O_{[t]}&=s\left(\texttt{b\_o}+
\texttt{b\_A}\,\mathbf V_{\mathrm{new},[t]}\right).
\end{aligned}
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

### 5.8 反向传播：状态伴随与三角变换梯度

- 输出计算与块末状态都依赖 $`\mathbf V_{\mathrm{new},[t]}`$，因此残差梯度由两条路径相加：

```math
\frac{\partial\mathcal L}{\partial\mathbf V_{\mathrm{new},[t]}}
=s\left(\mathbf Q_{[t]}\mathbf K_{[t]}^\top\odot\Gamma_{[t]}\right)^\top
\frac{\partial\mathcal L}{\partial\mathbf O_{[t]}}
+\overrightarrow{\mathbf K}_{[t]}
\left(\frac{\partial\mathcal L}{\partial\mathbf S_{[t+1]}}\right)^\top.
```

- `chunk_bwd_dv_local` 计算上式第一项；`chunk_gated_delta_rule_bwd_dhu` 加入后续状态贡献，并从后向前递推状态梯度。`chunk_bwd_dqkwg` 处理输出与状态更新对 Q/K/W 和 gate 的梯度；`prepare_wy_repr_bwd` 再通过三角变换将 W/U 梯度传回 K/V、beta 和 gate。

- 前向保存 A 及必要输入，反向先重算 W/U、入口状态和 `v_new`，以计算量换取较低的中间状态存储开销。最后的反向前缀和将累计 gate 的梯度汇总到逐 token gate。

- **代码作用：组织 chunk 反向传播。** 输入前向保存量、输出梯度 `do` 与末态梯度 `dht`，重算必要中间量，再依次求状态、W/U 和原始输入的梯度。

```python
w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g=g)
h, v_new, _ = chunk_gated_delta_rule_fwd_h(
    k=k, w=w, u=u, g=g, initial_state=initial_state,
    output_final_state=False,
)
dv = chunk_bwd_dv_local(q=q, k=k, g=g, do=do, scale=scale)
dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
    q=q, k=k, w=w, g=g, h0=initial_state,
    dht=dht, do=do, dv=dv, scale=scale,
)
dq, dk, dw, dg = chunk_bwd_dqkwg(
    q=q, k=k, v=v_new, w=w, g=g, h=h,
    dv=dv, do=do, dh=dh, scale=scale,
)
dk2, dv, db, dg2 = prepare_wy_repr_bwd(
    k=k, v=v, beta=beta, g=g, A=A, dw=dw, du=dv,
)
dk.add_(dk2)
dg.add_(dg2)
dg = chunk_local_cumsum(dg, chunk_size=64, reverse=True)
```

## 6. Recurrent 算法代码解析

### 6.1 状态分块与并行网格

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

- **代码作用：配置 recurrent 状态分片。** 输入 K/V 维度与序列、head 数，确定 BK/BV、执行网格，并分配 FP32 末态缓冲区。

```python
BK = triton.next_power_of_2(K)
BV = min(8, triton.next_power_of_2(V))
NV = triton.cdiv(V, BV)
final_state = q.new_empty(N, HV, K, V, dtype=torch.float32)
grid = (NV, N * HV)
```

- 这里 N 为序列数，$`H_V`$ 为 value head 数。Q/K head 通过 `i_hv // (HV // H)` 映射，同一组 value head 共享 Q/K 输入，但各自维护独立的状态。prefill 写出的 chunk 末态与此处初始状态使用同一种缓冲区排列，二者之间不需要重新组织数学状态。

### 6.2 逐 Token 更新：衰减、残差与读出

- 以代码中的状态矩阵 $`H_t=S_t^\top`$ 表示递推。完成 Q/K 归一化后，一步更新由以下算式组成：

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

- **代码作用：执行逐 token 状态更新。** 输入当前 Q/K/V、门控与初始状态，按衰减、残差、写入、读出的顺序计算，输出 token 结果并保存末态。

```python
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

- 在数学上令 chunk 长度为 1，严格下三角矩阵 L 为零，$`A=I`$，W/U 变换直接给出单步残差。因此 chunk 的末态可以直接作为 recurrent 的初始状态，prefill 与解码之间保持同一个状态递推关系。

# Gated DeltaNet：模型调用、Chunk 与 Recurrent 实现

## 1. GDN 与 Token Mixer

- 标准因果注意力先构造 token 两两之间的相似度，再经过 softmax 归一化。以下采用不含 softmax 的点积形式表示线性注意力，其历史键值对可累积到固定大小的状态矩阵。设 Q/K/V 按 token 排成行，$`M_{ij}=\mathbf 1_{j\le i}`$ 为二值因果掩码，$`\mathcal M_{ij}`$ 在可见位置为 0、不可见位置为 $`-\infty`$，两种输出分别为：

```math
O_{\mathrm{attn}}=\operatorname{softmax}(sQK^\top+\mathcal M)V,
\qquad O_{\mathrm{linear}}=s(QK^\top\odot M)V,
\qquad s=d_k^{-1/2}.
```

- 上述并行表达式包含序列长度平方数量的成对权重。FlashAttention 可通过分块与融合避免完整权重矩阵的显存往返，但成对计算量仍随序列长度平方增长。线性注意力的优势来自其等价递推：每个 head 维护 $`d_v\times d_k`$ 的状态，单步更新与读出的计算量为 $`O(d_kd_v)`$。因果掩码不能直接移到矩阵乘法外，因此训练时需要 chunkwise 分解，将块内计算组织为矩阵乘法，将块间依赖压缩到状态传递。

- 固定大小的状态将多组键值关联叠加存储，非正交 key 会在读出时相互干扰。GDN 同时引入历史衰减 $`\alpha_t`$ 和残差写入强度 $`\beta_t`$：前者控制旧状态整体保留多少，后者控制沿当前 key 方向修正多少。二者作用于同一个状态更新，而输出门只调节本层读出的结果。

<figure style="margin: 24px 0;">
  <a href="../../assets/gdn-architecture.png"><img src="../../assets/gdn-architecture.png" alt="Gated DeltaNet 模型结构与 token mixer" width="2058" height="1122"></a>
</figure>

- Token mixer 的 Q/K/V 分支依次经过线性投影、逐通道短因果卷积和 SiLU，Q/K 再沿 head 维做 L2 归一化。短卷积提供局部时序特征，Gated Delta Rule 维护跨 token 的矩阵状态；两者在解码时分别保留卷积窗口与 recurrent state。

- 衰减门与写入系数由独立投影生成，不经过 Q/K/V 的短卷积。状态输出先按 head 归一化，再与输出门逐元素相乘，最后投影回隐藏维度。图中的结构对应论文 §3.4；第 4 节展开 Qwen3.5 的具体投影、门控参数化与算子调用。

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

- 将序列划分为长度 C 的块，块内 Q/K/V 按 token 排成行。$`S_{[t]}`$ 表示第 t 块的入口状态，$`S_{[t]}^r`$ 表示处理块内前 r 个 token 后的状态。先考虑不含衰减的 DeltaNet，并在块内省略下标 [t]。连续转移矩阵的乘积可写为论文 Eq. (4) 的 WY 表示：

```math
P_r=\prod_{i=1}^{r}(I-\beta_i k_i k_i^\top)
=I-\sum_{i=1}^{r}w_i k_i^\top,
\qquad
w_i=\beta_i\!\left(k_i-\sum_{j<i}w_j(k_j^\top k_i)\right).
```

- 乘积按 token 顺序从左向右排列。当前块产生的新增状态也具有相同结构：$`G_r^{(0)}=\sum_{i=1}^{r}u_i k_i^\top`$，其中 $`u_i=\beta_i(v_i-\sum_{j<i}u_j(k_j^\top k_i))`$。W 与 U 的递推共享相同的 key 内积系数，仅右端输入分别为 K 和 V。

- 记 $`B=\operatorname{diag}(\beta)`$、$`L_0=\operatorname{strictLower}(BKK^\top)`$。将 W/U 的向量递推按行堆叠，得到论文 Eq. (6)–(7) 的 UT 变换：

```math
\begin{aligned}
\mathcal T_0&=(I+L_0)^{-1}B,
& W_0&=\mathcal T_0K,\qquad U_0=\mathcal T_0V,\\
P_C&=I-W_0^\top K,
& S_{[t+1]}&=S_{[t]}P_C+U_0^\top K.
\end{aligned}
```

- WY 表示将转移矩阵的连乘改写为 W 与 K 的矩阵乘积；UT 变换将 W/U 的逐位置依赖改写为同一个单位下三角系统。求解仍具有因果顺序，但可以按小块前代，再通过块间矩阵乘法合并，这正是第 5.4 节的两级实现。

### 3.2 引入衰减后的三角系统

- 对每个 chunk 重新定义累计衰减 $`\gamma_i=\prod_{r=1}^{i}\alpha_r`$。D 描述入口状态传播到各位置的衰减，R 描述各位置写入量传播到块末的衰减，$`\Gamma`$ 同时编码块内的因果可见性与区间衰减：

```math
D=\operatorname{diag}(\gamma),\qquad
R=\operatorname{diag}(\gamma_C/\gamma),\qquad
\Gamma_{ij}=\begin{cases}\gamma_i/\gamma_j,&i\ge j,\\0,&i<j.\end{cases}
```

- 论文 §3.3 将块内状态展开为 $`S_{[t]}^r=S_{[t]}F_r+G_r`$，其中 $`F_r=\gamma_rP_r`$。新增状态的 WY 表示需要同时计入此前更新传播到当前位置时的衰减：

```math
G_r=\sum_{i=1}^{r}\frac{\gamma_r}{\gamma_i}\widetilde u_i k_i^\top,
\qquad
\widetilde u_i=\beta_i\!\left(v_i-\sum_{j<i}\Gamma_{ij}\widetilde u_j(k_j^\top k_i)\right).
```

- 因此，UT 系统中的 Gram 矩阵由 $`KK^\top`$ 变为 $`\Gamma\odot KK^\top`$。定义 L 为严格下三角系数，A 为单位下三角矩阵的逆，可同时求得 value 变换与入口状态变换：

```math
\begin{aligned}
L&=\operatorname{strictLower}\!\left(B(\Gamma\odot KK^\top)\right),
& A&=(I+L)^{-1},\\
\widetilde U&=ABV,
& \overleftarrow W&=ABDK=DW_0.
\end{aligned}
```

- $`\overleftarrow W=DW_0`$ 对齐论文中的箭头记号：它是无衰减 W 的逐行缩放。实现直接计算 $`A(BDK)`$，因为 $`L=DL_0D^{-1}`$，两种写法等价。A 只表示逆矩阵；右端项的 B 在生成 W/U 时相乘，不能再次计入 A 的定义。

### 3.3 块间状态与块内输出

- 令 $`E=\widetilde U-\overleftarrow W S_{[t]}^\top`$，表示扣除块入口状态预测后的写入残差。沿用论文的箭头记号，$`\overleftarrow Q=DQ`$、$`\overrightarrow K=RK`$、$`\overrightarrow S=\gamma_CS_{[t]}`$。完整 chunk 计算为：

```math
\begin{aligned}
S_{[t+1]}&=\overrightarrow S+E^\top\overrightarrow K
=\gamma_CS_{[t]}+E^\top RK,\\
O_{[t]}&=s\left[\overleftarrow Q S_{[t]}^\top
+(QK^\top\odot\Gamma)E\right].
\end{aligned}
```

- 状态更新先将历史衰减到块末，再累积当前块的残差写入。输出中的第一项读取块前历史，第二项汇总当前块内的残差；后一项使用带衰减的 $`\Gamma`$，其对角线为 1，使当前 token 的更新参与当前输出。只有在 $`\alpha_i=1`$ 时，$`\Gamma`$ 才退化为二值因果掩码 M。

- W/U 的生成可按块并行，状态更新按块顺序推进；各块入口状态得到后，输出计算再次按块并行。尾块使用最后一个有效 token 的累计衰减。第 5 节依次实现上述三角系统、残差计算、状态写入与输出读出；令 C=1 时，即得到第 6 节的单步 recurrent 更新。

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

- 对第 $`t`$ 个 chunk，记 $`H_t=S_{[t]}^\top`$，$`R=\operatorname{diag}(\gamma_C/\gamma)`$。前向计算由下三角求解、残差变换、状态递推和输出矩阵乘法组成：

```math
\begin{aligned}
A&=(I+L)^{-1},\qquad
\widetilde U=ABV,\qquad \overleftarrow W=ABDK,\\
V_{\mathrm{new}}&=\widetilde U-\overleftarrow W H_t,\\
H_{t+1}&=\gamma_C H_t+K^\top R V_{\mathrm{new}},\\
O_{[t]}&=s\left[DQH_t+(QK^\top\odot\Gamma)V_{\mathrm{new}}\right].
\end{aligned}
```

- `chunk_gated_delta_rule_fwd_intra` 构造并求解块内系统，生成 `A`、`u` 和 `w`；`chunk_gated_delta_rule_fwd_h` 引入块入口状态，计算 `v_new` 及下一块状态；`chunk_fwd_o` 完成输出计算。这里 `h` 保存所有块的入口状态，`final_state` 保存各序列的最终状态。

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

- 令 $`J`$ 为包含对角线的全 1 下三角矩阵，$`g_i=\log\alpha_i`$。累计门控等价于一次下三角线性变换：

```math
\ell=\frac{1}{\ln 2}Jg,\qquad
J_{ij}=\mathbf 1_{j\le i},\qquad
\gamma_i=2^{\ell_i},\qquad
\frac{\gamma_i}{\gamma_j}=2^{\ell_i-\ell_j}.
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

- `REVERSE` 分支计算后缀和：$`(J^\top x)_i=\sum_jx_j-(Jx)_i+x_i`$，对应前缀和的反向传播。`HAS_SCALE` 决定是否乘调用侧传入的缩放，GDN 前向传入 `RCP_LN2`。上述代码保留时间优先的访存分支。

- 每个 chunk 独立累计 gate。块前历史已经包含在入口状态中，再乘当前块的累计衰减即可；因此前缀和不需要跨 chunk 延伸。对于变长输入，块索引同时指定序列编号和序列内块编号，累计过程不能跨越序列边界。

### 5.3 KKT：构造块内递推系数

- 第 $`i`$ 个位置的 Delta 修正依赖更早位置的 key 内积。更新系数按行缩放，区间衰减逐元素作用于 Gram 矩阵，得到：

```math
L=\operatorname{strictLower}\!\left(B(\Gamma\odot KK^\top)\right),\qquad
L_{ij}=\begin{cases}\beta_i\,2^{\ell_i-\ell_j}\,k_i^\top k_j,&i>j,\\0,&i\le j.\end{cases}
```

- `fla/ops/common/chunk_scaled_dot_kkt.py` 的 `chunk_scaled_dot_kkt_fwd_kernel` 实现这一计算。网格为 `(NT, B * HV)`，每个 program 生成一个 head 的块内系数。K 维按 `BK` 分片，`tl.dot(b_k, tl.trans(b_k))` 将各片段的内积累加到 FP32 的 `[BT, BT]` 矩阵中。

- 这是 16/32-token 分步路径使用的 kernel；64-token 默认路径将相同计算与下一节的求解融合。先展开独立 KKT，能够直接对应“内积、衰减、beta、严格下三角”四个矩阵操作。

- **代码作用：构造严格下三角系数 L。** 输入 K、累计 gate 和 beta，分片累计 key 内积并施加衰减与掩码，输出 `[B, T, HV, BT]` 系数缓冲区。

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

- `i_h // (HV // H)` 将 value head 映射到共享的 Q/K head。K 的时间步长为 `H * K`，gate 和 beta 的时间步长为 `HV`；因此 head 共享改变输入索引，但每个 value head 仍生成独立的 L。

- 变长输入通过 `chunk_indices[i_t]` 取得“序列编号、序列内 chunk 编号”，再由 `cu_seqlens` 确定 `bos` 和当前序列长度。`o_t < T` 限制实际 token 范围，`o_t[:, None] > o_t[None, :]` 只保留此前位置对当前残差的影响。结果按 `[B, T, HV, BT]` 写回，每个 token 保存其所在块的一行系数。

- 64-token 融合 kernel 位于 `fla/ops/gated_delta_rule/chunk_fwd.py`，名称为 `chunk_gated_delta_rule_fwd_kkt_solve_kernel`。它将时间维拆成四个 16-token 子块，只计算四个对角块和六个非对角下三角块；对角块额外应用严格下三角掩码。这十个子矩阵保留在寄存器中，直接进入前代与合并阶段。

### 5.4 下三角求解：局部前代与块间合并

- L 严格下三角，因此 $`M=I+L`$ 的对角线为 1。标量前代与 16×16 分块前代分别满足：

```math
\begin{aligned}
A_{ii}&=1,\qquad A_{ij}=-L_{ij}-\sum_{k=j+1}^{i-1}L_{ik}A_{kj}\quad(i>j),\\
\mathcal A_{rr}&=(I+\mathcal L_{rr})^{-1},\qquad
\mathcal A_{rs}=-\mathcal A_{rr}\sum_{p=s}^{r-1}\mathcal L_{rp}\mathcal A_{ps}\quad(r>s).
\end{aligned}
```

- $`\mathcal L_{rs}`$、$`\mathcal A_{rs}`$ 分别表示 L 和 A 的子块。整体矩阵关系为：

```math
\begin{pmatrix}
I+\mathcal L_{00}&0&0&0\\
\mathcal L_{10}&I+\mathcal L_{11}&0&0\\
\mathcal L_{20}&\mathcal L_{21}&I+\mathcal L_{22}&0\\
\mathcal L_{30}&\mathcal L_{31}&\mathcal L_{32}&I+\mathcal L_{33}
\end{pmatrix}
\begin{pmatrix}
\mathcal A_{00}&0&0&0\\
\mathcal A_{10}&\mathcal A_{11}&0&0\\
\mathcal A_{20}&\mathcal A_{21}&\mathcal A_{22}&0\\
\mathcal A_{30}&\mathcal A_{31}&\mathcal A_{32}&\mathcal A_{33}
\end{pmatrix}=I.
```

- `fla/ops/utils/solve_tril.py` 采用两级分块：先分别求四个 16×16 对角块的逆，再用矩阵乘法补齐六个非对角块。`merge_16x16_to_64x64_inverse_kernel` 在一个 program 内完成两级操作。下面以零基索引表示子块；`split_blocks` 和 `assemble_blocks` 仅表示子块视图与结果组装。

- **伪代码：求解单位下三角系统。** 输入严格下三角矩阵 L；先做 16 行以内的局部前代，再按块距离合并；输出 $`A=(I+L)^{-1}`$。尾块的无效行列按零填充，写回时只保留有效 token。

```python
# 伪代码；矩阵乘法对应 kernel 中的 tl.dot。
def inverse_unit_lower(L):
    L_blocks = split_blocks(L, block_size=16)
    A_blocks = zeros_like(L_blocks)  # 4 × 4 个子块
    index = arange(16)

    # 第一级：四个对角子问题相互独立。
    for r in range(4):
        lower = strict_lower(L_blocks[r, r])
        inverse = -lower
        for i in range(2, 16):
            row = where(index < i, -lower[i], 0)
            inverse[i] = row + row @ inverse
        A_blocks[r, r] = inverse + eye(16)

    # 第二级：相邻块 → 跨一块 → 跨两块。
    for distance in range(1, 4):
        for r in range(distance, 4):
            c = r - distance
            accumulated = zeros((16, 16))
            for p in range(c, r):
                accumulated += L_blocks[r, p] @ A_blocks[p, c]
            A_blocks[r, c] = -A_blocks[r, r] @ accumulated

    return assemble_blocks(A_blocks)
```

- **局部前代。** 逆矩阵的严格下三角部分初始化为负 L。第 0 行没有下三角元素，第 1 行的结果已由初始化给出，因此循环从索引 2 开始。`row @ inverse` 对应 Triton 的 `tl.sum(row[:, None] * inverse, 0)`；当前行只引用已经求出的行，最后再加入单位对角线。

- **块间合并。** 距离为 1 时计算三个相邻子块，距离为 2 时计算两个子块，距离为 3 时计算左下角子块。例如 $`\mathcal A_{20}=-\mathcal A_{22}(\mathcal L_{20}\mathcal A_{00}+\mathcal L_{21}\mathcal A_{10})`$。每个非对角结果都由此前求出的逆矩阵子块共同决定。

- **执行路径。** 通用 `BT=16/32/64` 路径分别使用 `solve_tril_16x16_kernel` 与两种 `merge_16x16_to_*_inverse_kernel`。GDN 默认 64-token 路径则在 `chunk_gated_delta_rule_fwd_kkt_solve_kernel` 中将 KKT、局部前代和块合并融合：直接从寄存器矩阵提取当前行，省去 L 的中间写回与重新加载。

- **访存与精度。** 通用 `solve_tril` 在环境支持时使用 TMA descriptor；`FLA_TRIL_PRECISION` 默认 `ieee`，支持 TMA 时 autotune 的精度候选为 `ieee` 与用户指定值。默认 GDN 融合 kernel 使用普通指针加载，支持 TF32 时块合并采用 `tf32`，否则采用 `ieee`。两条路径的数学分解相同，访存与精度配置分别选择。

### 5.5 W/U：共享同一个三角变换

- 求得 A 后，将两个右端项分别代入同一个系统：

```math
\widetilde U=A(BV),\qquad \overleftarrow W=A(BDK),\qquad A=(I+L)^{-1}.
```

- `fla/ops/gated_delta_rule/wy_fast.py` 的 `recompute_w_u_fwd_kernel` 使用网格 `(NT, B * HV)`。每个 program 读取一个 chunk 的 A，沿 V 维和 K 维分别分片计算 U/W；A 在两组矩阵乘法之间复用。

- **伪代码：重建块内 W/U。** 输入 A、K、V、beta 和累计 gate；将 beta 乘入两个右端项，将累计衰减额外乘入 K；输出 `u` 与 `w`。`load_*`、`store_*` 表示带尾块掩码的读取与写回。

```python
# 伪代码；每个 program 处理一个 chunk、一个 value head。
chunk, head = program_id()
A, beta, ell = load_chunk_transform(chunk, head)
gamma = exp2(ell)
key_head = head // (HV // H)

for value_slice in tiles(V, BV):
    v = load_v(chunk, head, value_slice)
    rhs_v = (beta[:, None] * v).to(v.dtype)
    u = tl.dot(A, rhs_v, allow_tf32=False)
    store_u(chunk, head, value_slice, u)

for key_slice in tiles(K, BK):
    k = load_k(chunk, key_head, key_slice)
    rhs_k = beta[:, None] * gamma[:, None] * k
    w = tl.dot(A, rhs_k.to(k.dtype))
    store_w(chunk, head, key_slice, w)
```

- K 的衰减发生在乘 A 之前，对应右端项 BDK；一般情况下 D 与 A 不可交换。A 本身只包含三角逆变换，右端项的 beta 在这里相乘一次。`u` 尚未扣除块入口状态的预测，下一步通过 `u - w @ h` 形成实际写入残差。

### 5.6 块间状态：残差校正与衰减写入

- 记入口状态 $`H_t=S_{[t]}^\top`$，$`R=\operatorname{diag}(\gamma_C/\gamma)`$。一次块更新包括残差计算与状态写入：

```math
V_{\mathrm{new}}=\widetilde U-\overleftarrow W H_t,
\qquad H_{t+1}=\gamma_C H_t+K^\top R V_{\mathrm{new}}.
```

- `fla/ops/common/chunk_delta_h.py` 的 `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` 使用网格 `(ceil(V / BV), N * HV)`。每个 program 处理一条序列、一个 value head 和一片 value 通道，沿 chunk 顺序递推；K 维按 64 通道展开为多个寄存器状态片段。

- **伪代码：递推状态并保存块内残差。** 输入 K、W、U、累计 gate 与初始状态；先保存块入口状态，再计算残差并更新寄存器状态；输出所有入口状态 `h`、未乘尾部衰减的 `v_new` 和最终状态。各 `load_*`、`store_*` 均使用当前序列边界与有效 token 掩码。

```python
# 伪代码；每个 program 处理 (sequence, value_head, value_slice)。
sequence, head, value_slice = program_id()
key_head = head // (HV // H)
key_tiles = tiles(K, 64)
H_parts = load_initial_state(sequence, head, key_tiles, value_slice)

for chunk in sequence_chunks(sequence):
    ell, valid = load_cumulative_gate(chunk, head)
    ell_last = ell[last_valid_position(valid)]
    store_entry_state(chunk, head, value_slice, H_parts)

    prediction = zeros((BT, BV), dtype=float32)
    for j, key_slice in enumerate(key_tiles):
        w = load_w(chunk, head, key_slice)
        prediction += tl.dot(w, H_parts[j].to(w.dtype))
    u = load_u(chunk, head, value_slice)
    v_new = u - prediction
    store_v_new(chunk, head, value_slice, v_new)

    # 保存后再施加从当前位置到块末的衰减。
    tail_decay = where(valid, exp2(ell_last - ell), 0)
    residual = v_new * tail_decay[:, None]
    for j, key_slice in enumerate(key_tiles):
        k = load_k_transposed(chunk, key_head, key_slice)
        H_parts[j] *= exp2(ell_last)
        H_parts[j] += tl.dot(k, residual.to(k.dtype))

store_final_state(sequence, head, value_slice, H_parts)
```

- **K 维分片。** K=128 时维护两个 `[64, BV]` 片段。计算 $`\overleftarrow W H_t`$ 时，两个乘法结果先相加，再从 U 中扣除；更新状态时，同一份残差分别与两片 key 相乘。各片段共同构成一个状态矩阵，源码以相同方式支持 K≤256。

- **保存顺序。** `h` 保存块入口状态，`v_new` 保存尚未乘 R 的残差。后续输出 kernel 使用它们恢复块内各位置的结果；块末状态与乘 R 后的残差分别服务于下一块状态递推。

- **尾块与变长序列。** `ell_last` 取最后一个有效 token 的累计 gate。packed 输入使用 `cu_seqlens` 定位 token 范围，使用 `chunk_offsets` 定位入口状态缓冲区；两种偏移分别按 token 数和 chunk 数累计。每条序列从自己的初始状态开始，结束后独立保存末态。

### 5.7 块内输出：历史项与局部项

- 各块的入口状态与残差生成后，输出可按块并行。记 $`P=s(QK^\top\odot\Gamma)`$，则：

```math
O_{[t]}=sDQH_t+PV_{\mathrm{new}},\qquad P=s(QK^\top\odot\Gamma).
```

- `fla/ops/common/chunk_o.py` 的 `chunk_fwd_kernel_o` 使用网格 `(ceil(V / BV), NT, B * HV)`。沿 K 维循环时，同时累加 `[BT, BV]` 的历史读出与 `[BT, BT]` 的块内权重；前者与后者共享同一份 query 分片。

- **伪代码：合成当前块输出。** 输入 Q/K、块入口状态、`v_new` 与累计 gate；分别计算历史项和块内项，应用衰减、因果掩码及 query 缩放；输出当前 value 分片的 O。`load_*` 屏蔽越界行列并补零。

```python
# 伪代码；每个 program 处理 (chunk, value_head, value_slice)。
chunk, head, value_slice = program_id()
key_head = head // (HV // H)
history = zeros((BT, BV), dtype=float32)
scores = zeros((BT, BT), dtype=float32)

for key_slice in tiles(K, BK):
    q = load_q(chunk, key_head, key_slice)          # [BT, BK]
    k = load_k_transposed(chunk, key_head, key_slice)  # [BK, BT]
    h = load_entry_state(chunk, head, key_slice, value_slice)
    history += tl.dot(q, h)                       # [BT, BV]
    scores += tl.dot(q, k)                        # [BT, BT]

ell, valid = load_cumulative_gate(chunk, head)
history *= exp2(ell)[:, None]
scores *= exp2(ell[:, None] - ell[None, :])
causal = lower_triangle(BT, include_diagonal=True)
scores = where(causal & valid[:, None] & valid[None, :], scores, 0)
v_new = load_v_new(chunk, head, value_slice)
output = scale * (history + tl.dot(scores.to(v_new.dtype), v_new))
store_output(chunk, head, value_slice, output)
```

- 历史项中的 D 表示入口状态传播到各位置的衰减；局部项中的 $`\Gamma`$ 表示块内残差在不同位置之间的传播。输出掩码包含对角线，使当前 token 读到本步更新后的状态；KKT 掩码则严格下三角，只表示此前位置对当前残差的影响。

- 变长输入保留全局 chunk 编号以读取入口状态，同时使用序列内 chunk 编号读取 Q/K/V。源码的 `i_tg` 与 `i_t` 分别承担这两种索引。局部变量 `b_A` 在此处表示 QK 权重，与三角求解的 A 无关。

### 5.8 反向传播：状态伴随与三角变换梯度

- 令 $`E=V_{\mathrm{new}}`$，$`G_O=\partial\mathcal L/\partial O`$，$`G_{H_{t+1}}=\partial\mathcal L/\partial H_{t+1}`$。块内输出和块间状态两条路径共同形成 E 的梯度；入口状态的伴随则按逆序递推：

```math
\begin{aligned}
G_E&=P^\top G_O+RK\,G_{H_{t+1}},\\
G_{H_t}&=\gamma_C G_{H_{t+1}}
+sQ^\top D G_O-\overleftarrow W^\top G_E,\\
G_{\widetilde U}&=G_E,\qquad
G_{\overleftarrow W}=-G_EH_t^\top,\\
G_L&=\operatorname{strictLower}\!\left(-A^\top G_AA^\top\right).
\end{aligned}
```

- `chunk_bwd_dv_local` 计算局部输出贡献 $`P^\top G_O`$；`chunk_gated_delta_rule_bwd_dhu` 加入来自后续状态的贡献，并从后向前递推状态梯度。`chunk_bwd_dqkwg` 处理输出与状态更新对 Q/K/W 和 gate 的梯度；`prepare_wy_repr_bwd` 再通过三角变换将 W/U 梯度传回 K/V、beta 和 gate。

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

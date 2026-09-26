# Gated DeltaNet：模型调用、Chunk 与 Recurrent 实现

## 1. GDN 与 Token Mixer

Gated DeltaNet 在 Delta Rule 中引入数据相关的衰减门，同时控制历史状态的保留与当前键值对的写入。其 token mixer 由输入投影、短因果卷积、状态更新、输出归一化与门控组成。

<figure style="margin: 24px 0;">
  <a href="../../assets/gdn-architecture.png"><img src="../../assets/gdn-architecture.png" alt="Gated DeltaNet 论文模型图：左侧为 H1，中间为 H2，右侧为 token mixer 的投影、卷积、L2 归一化、Gated Delta Rule 与输出门控" width="2058" height="1122"></a>
  <figcaption style="text-align: center;">图 1．Gated DeltaNet-H1、H2 与 token mixer 结构（<a href="https://arxiv.org/pdf/2412.06464#page=7">论文 Figure 1</a>）。</figcaption>
</figure>

图右侧的 Q/K/V 分支经过线性投影、短卷积与 SiLU，Q/K 进一步进行 L2 归一化。衰减门与更新系数由输入的独立投影产生，分别作用于历史状态和 value 残差。状态读出结果经过归一化后与输出门逐元素相乘，再投影回模型隐藏维度。输出门只调节本层输出，不参与状态递推。

## 2. Gated Delta Rule 的计算公式

对单个 head，令 $`q_t,k_t\in\mathbb R^{d_k}`$，$`v_t\in\mathbb R^{d_v}`$，状态 $`S_t\in\mathbb R^{d_v\times d_k}`$。状态转移及其残差形式为：

```math
\begin{aligned}
S_t
&=\alpha_t S_{t-1}(I-\beta_t k_tk_t^\top)+\beta_t v_tk_t^\top\\
&=\alpha_t S_{t-1}
  +\beta_t\bigl(v_t-\alpha_t S_{t-1}k_t\bigr)k_t^\top,\\
o_t&=sS_tq_t,\qquad s=d_k^{-1/2}.
\end{aligned}
```

$`\alpha_t`$ 对历史状态施加整体衰减，$`\beta_t`$ 控制沿当前 key 方向的更新幅度。残差中的旧 value 由衰减后的状态读取，因此衰减必须先于残差计算。写入完成后，再用 query 读取更新后的状态。

## 3. Chunk 的矩阵表示

将序列划分为长度为 $`C`$ 的块，块内 Q/K/V 按 token 排成行。累计衰减与因果衰减矩阵定义为：

```math
\gamma_i=\prod_{r=1}^{i}\alpha_r,\qquad
\Gamma_{ij}=\begin{cases}\gamma_i/\gamma_j,&i\ge j,\\0,&i<j.\end{cases}
```

WY 表示将连续的 Delta 更新压缩为两个变换矩阵；UT 变换进一步将其递推关系写成单位下三角系统。记 $`B=\operatorname{diag}(\beta)`$、$`D=\operatorname{diag}(\gamma)`$，则：

```math
\begin{aligned}
L&=\operatorname{strictLower}\!\left(B(\Gamma\odot KK^\top)\right),\\
(I+L)\widetilde U&=BV,\qquad
(I+L)\overleftarrow W=BDK.
\end{aligned}
```

$`L`$ 编码当前块内部较早 key 对后续位置的影响。求解同一个下三角系统，可以同时得到 value 变换 $`\widetilde U`$ 和作用于块入口状态的变换 $`\overleftarrow W`$。扣除入口状态的预测后，得到真正参与状态写入与输出计算的 $`V_{\mathrm{new}}`$。

## 4. 模型调用方式：Qwen3.5 GatedDeltaNet

### 4.1 输入投影

设输入为 $`X\in\mathbb R^{T\times d_{\mathrm{model}}}`$。Q/K 的总投影维度为 $`d_Q=H_Kd_k`$，V 的总投影维度为 $`d_V=H_Vd_v`$。四个线性映射分别产生卷积输入、衰减参数、写入参数和输出门：

```math
\begin{aligned}
P&=XW_{qkv}^{\top},
& W_{qkv}&\in\mathbb R^{(2d_Q+d_V)\times d_{\mathrm{model}}},\\
a&=XW_a^{\top},\quad b=XW_b^{\top},
& W_a,W_b&\in\mathbb R^{H_V\times d_{\mathrm{model}}},\\
Z&=XW_z^{\top},
& W_z&\in\mathbb R^{d_V\times d_{\mathrm{model}}}.
\end{aligned}
```

`Qwen3_5MoeGatedDeltaNet` 将 Q/K/V 合并为一次投影，随后按通道拆分。衰减参数和写入参数只需要每个 value head 一个标量，因此 `in_proj_a`、`in_proj_b` 的输出维度为 `num_v_heads`；输出门需要逐通道调节读出结果，其投影维度与 V 相同。

```python
self.in_proj_qkv = nn.Linear(self.hidden_size, 2 * self.key_dim + self.value_dim, bias=False)
self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
```

### 4.2 短卷积、门控与 Head 映射

令 $`\mathcal C`$ 表示按通道独立计算的因果卷积。卷积后的 Q/K 尚未归一化，记为 $`Q^0,K^0`$。该阶段计算：

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

短卷积沿时间维混合局部上下文，各通道之间不发生卷积混合。合并投影的输出在卷积时采用 `[batch, channels, time]` 排列，卷积后恢复时间优先布局，再拆分为多头 Q/K/V。

`A_log` 和 `dt_bias` 是每个 value head 的可学习参数。上述参数化保证 `g` 为负值，从而将历史衰减限制在 0 与 1 之间；`beta` 经 sigmoid 控制残差写入强度。Grouped Value Attention 通过 head 映射使多个 value head 共享 Q/K，模型代码使用 `repeat_interleave` 显式实现该映射。

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

有缓存的单 token 分支使用增量因果卷积维护短卷积窗口；它与 GDN 的矩阵状态分别保存。前者提供当前 token 的局部 Q/K/V 特征，后者累积更长范围的序列信息。

### 4.3 状态算子与输出门控

Q/K 在进入状态更新前沿 head 维归一化。记 $`\mathcal D`$ 为 Gated Delta Rule 算子，则输入归一化、状态计算和输出映射可写为：

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

Chunk 与 recurrent 实现同一个状态算子。模型根据输入长度和已有状态选择计算方式：完整序列使用 chunk；存在缓存且输入长度为 1 时使用 recurrent。两者均返回当前输出和序列末态，末态回写缓存，供后续输入接续计算。

输出归一化独立作用于每个 value head。`self.norm` 先执行 RMSNorm，再乘 `SiLU(z)`；head 维合并后，`out_proj` 将结果映射回模型隐藏维度。以下代码合并了两条分支中相同的调用参数。

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

对第 $`t`$ 个 chunk，记 $`H_t=S_{[t]}^\top`$，$`R=\operatorname{diag}(\gamma_C/\gamma)`$。前向计算由下三角求解、残差变换、状态递推和输出矩阵乘法组成：

```math
\begin{aligned}
A&=(I+L)^{-1},\qquad
\widetilde U=ABV,\qquad \overleftarrow W=ABDK,\\
V_{\mathrm{new}}&=\widetilde U-\overleftarrow W H_t,\\
H_{t+1}&=\gamma_C H_t+K^\top R V_{\mathrm{new}},\\
O_{[t]}&=s\left[DQH_t+(QK^\top\odot\Gamma)V_{\mathrm{new}}\right].
\end{aligned}
```

`chunk_gated_delta_rule_fwd_intra` 构造并求解块内系统，生成 `A`、`u` 和 `w`；`chunk_gated_delta_rule_fwd_h` 引入块入口状态，计算 `v_new` 及下一块状态；`chunk_fwd_o` 完成输出计算。这里 `h` 保存所有块的入口状态，`final_state` 保存各序列的最终状态。

默认 64-token 路径将 KKT 构造与下三角求解融合执行，随后单独生成 W/U。块内变换可以跨 chunk 并行；状态更新仍按 chunk 顺序递推；入口状态生成后，各块输出又可以并行计算。下面保留固定长度、无 context parallelism 的调用主干。

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

令 $`J`$ 为包含对角线的全 1 下三角矩阵，$`g_i=\log\alpha_i`$。累计门控等价于一次下三角线性变换：

```math
\ell=\frac{1}{\ln 2}Jg,\qquad
J_{ij}=\mathbf 1_{j\le i},\qquad
\gamma_i=2^{\ell_i},\qquad
\frac{\gamma_i}{\gamma_j}=2^{\ell_i-\ell_j}.
```

`chunk_local_cumsum` 在每个 chunk 内沿时间轴执行前缀和。累积在 FP32 中完成，并通过 `RCP_LN2` 将自然对数转换为以 2 为底的对数。后续 kernel 使用 `exp2` 还原整体衰减和区间衰减，避免在各阶段重复计算连乘。

```python
b_g = tl.cumsum(b_log_alpha, axis=0) * RCP_LN2
```

每个 chunk 独立累计 gate。块前历史已经包含在入口状态中，再乘当前块的累计衰减即可；因此前缀和不需要跨 chunk 延伸。对于变长输入，块索引同时指定序列编号和序列内块编号，累计过程不能跨越序列边界。

### 5.3 KKT：构造块内递推系数

第 $`i`$ 个位置的 Delta 修正依赖更早位置的 key 内积。加入衰减和更新系数后，这些依赖形成严格下三角矩阵：

```math
L_{ij}=\begin{cases}
\beta_i\,2^{\ell_i-\ell_j}\,k_i^\top k_j,&i>j,\\
0,&i\le j,
\end{cases}
\qquad
L=\operatorname{strictLower}\!\left(B(\Gamma\odot KK^\top)\right).
```

64×64 的块被划分为 16×16 子块。`tl.dot` 首先计算各子块的 key 内积；对角子块应用严格下三角掩码，非对角下三角子块则全部保留。随后乘相应的区间衰减，并以行向量 `beta` 缩放。较大的 key 维度通过多个 K 分块累加，下面保留一组 K 分块的核心计算。

```python
b_A00 = tl.dot(b_k0, tl.trans(b_k0))
b_A10 = tl.dot(b_k1, tl.trans(b_k0))

m_d = o_i[:, None] > o_i[None, :]
b_A00 *= tl.where(
    m_d & m_tc0[:, None] & m_tc0[None, :],
    exp2(b_g0[:, None] - b_g0[None, :]), 0.,
)
b_A10 *= tl.where(
    m_tc1[:, None] & m_tc0[None, :],
    exp2(b_g1[:, None] - b_g0[None, :]), 0.,
)
b_A00 *= b_b0[:, None]
b_A10 *= b_b1[:, None]
```

`b_A00` 对应第一个对角子块，`b_A10` 对应其下方子块。`m_tc0/m_tc1` 同时屏蔽越界的行和列，使尾块填充位置不进入递推系数。源码在这里使用 `b_A*` 命名工作区，但其内容仍是 $`L`$；只有完成下一步求解后，输出缓冲区才保存逆矩阵 $`A`$。

### 5.4 下三角求解：局部前代与块间合并

由于 $`L`$ 严格下三角，$`M=I+L`$ 为单位下三角矩阵。求解 $`MA=I`$ 可以按行前代完成：

```math
A_{ii}=1,\qquad A_{ij}=0\ (i<j),\qquad
A_{ij}=-L_{ij}-\sum_{k=j+1}^{i-1}L_{ik}A_{kj}\ (i>j).
```

该递推只使用已经求得的行，不需要通用矩阵求逆。实现采用两级分块：先分别求四个 16×16 对角块的逆，再通过分块矩阵乘法补齐 64×64 逆矩阵的非对角部分。默认 GDN 路径将这两级操作与前一节的 KKT 一起放入 `chunk_gated_delta_rule_fwd_kkt_solve_kernel`，KKT 子块可以直接在寄存器中参与求解。

以下保留第一个对角块的前代过程。`b_Ai00` 初始为负严格下三角部分，其第二行已等于求解结果，因此循环从第 3 行开始。每次迭代提取当前行，与此前求出的逆矩阵行组合；完成后加上单位对角线。

```python
b_Ai00 = -b_A00
for i in range(2, min(BC, T - i_tc0)):
    b_a00 = tl.sum(tl.where((o_i == i)[:, None], -b_A00, 0.), 0)
    b_a00 = tl.where(o_i < i, b_a00, 0.)
    b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
    b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
b_Ai00 += o_i[:, None] == o_i[None, :]
```

对角块求解完成后，利用分块矩阵乘法恢复整体逆矩阵。记 $`A_{ij}`$ 为逆矩阵的 16×16 子块，则与下面两段计算对应的关系为：

```math
\begin{aligned}
A_{ii}&=(I+L_{ii})^{-1},\\
A_{10}&=-A_{11}L_{10}A_{00},\\
A_{20}&=-A_{22}\left(L_{20}A_{00}+L_{21}A_{10}\right).
\end{aligned}
```

第一条非对角关系合并相邻的两个子块；第二条同时累计从第 0 块直接传入和经第 1 块传入的依赖。剩余子块按相同的前代顺序计算。逐元素的串行求解因此被限制在 16 行以内，跨子块的大部分运算转为 `tl.dot`。

```python
b_Ai10 = -tl.dot(
    tl.dot(b_Ai11, b_A10, input_precision=SOLVE_TRIL_DOT_PRECISION),
    b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION,
)
b_Ai20 = -tl.dot(
    b_Ai22,
    tl.dot(b_A20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
    + tl.dot(b_A21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
    input_precision=SOLVE_TRIL_DOT_PRECISION,
)
```

通用 `solve_tril.py` 对 16、32、64 的块长分别提供 `solve_tril_16x16_kernel`、`merge_16x16_to_32x32_inverse_kernel` 和 `merge_16x16_to_64x64_inverse_kernel`；其中 32/64 路径同样在单个 kernel 内完成局部求逆与合并。运行环境支持 TMA 时，该通用实现使用 tensor descriptor 加载和存储子块。其 `FLA_TRIL_PRECISION` 默认值为 `ieee`；支持 TMA 时，autotune 的精度候选由 `ieee` 和用户指定值组成。

默认 GDN 64-token 融合路径位于 `chunk_fwd.py`，使用普通指针加载；块合并精度由 `SOLVE_TRIL_DOT_PRECISION` 决定，支持 TF32 时取 `tf32`，否则取 `ieee`。融合减少了 KKT 中间矩阵的全局内存往返，但同时需要保留更多寄存器中间量，其性能取决于具体维度和硬件资源。

### 5.5 W/U：共享同一个三角变换

求得 $`A`$ 后，将两个右端项分别代入同一个线性系统：

```math
\begin{aligned}
(I+L)\,[\widetilde U\mid\overleftarrow W]
  &=B[V\mid DK],\\
\widetilde U&=A(BV),\qquad
\overleftarrow W=A(BDK).
\end{aligned}
```

`recompute_w_u_fwd_kernel` 读取逆矩阵 `A`，将 beta 按行乘到 V 和 K 上，再执行两组矩阵乘法。K 分支还需要先乘累计衰减，得到右端项 `BDK`。这个缩放发生在乘 A 之前，因为一般情况下，对角门控矩阵 D 与下三角逆矩阵 A 不能交换。

```python
b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
b_u = tl.dot(b_A, b_vb, allow_tf32=False)

b_kb = b_k * b_b[:, None]
b_kb *= exp2(b_g)[:, None]
b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
```

W/U 只依赖块内输入，可以按 chunk 独立生成。`u` 已包含当前块内部的 Delta 修正，但尚未扣除块入口状态对 value 的预测；`w` 表示该预测应经过的线性变换。两者在下一节通过 `u - w @ h` 合成实际残差。

### 5.6 块间状态：残差校正与衰减写入

记块入口状态为 $`H_t=S_{[t]}^\top`$，$`R=\operatorname{diag}(\gamma_C/\gamma)`$。单次块更新为：

```math
\begin{aligned}
V_{\mathrm{new}}&=\widetilde U-\overleftarrow W H_t,\\
H_{t+1}&=\gamma_C H_t+K^\top R V_{\mathrm{new}}.
\end{aligned}
```

第一式从块内变换结果中减去入口状态的预测，得到各位置的写入残差。第二式将入口状态衰减到块末，并把各位置残差经过剩余时间步的衰减后累积进去。代码将 R 乘在 `v_new` 上，再与按 `[K, C]` 加载的 key 相乘，等价于公式中的 $`K^\top R V_{\mathrm{new}}`$。

```python
# 保存当前块入口状态，后续输出计算会读取它。
tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=m_h)

# V_new = U_tilde - W_left @ S_in.T
b_v_new = b_u - tl.dot(b_w, b_h.to(b_w.dtype))
tl.store(p_v_new, b_v_new.to(p_v_new.dtype.element_ty), mask=m_v)

# 将本块残差衰减到块末，再写入状态。
b_v = b_v_new * tl.where(
    m_t, exp2(b_g_last - b_g), 0.,
)[:, None]
b_h *= exp2(b_g_last)
b_h += tl.dot(b_k, b_v.to(b_k.dtype))
```

状态 kernel 沿 chunk 顺序执行。每次迭代先保存入口状态，再保存尚未乘 R 的 `v_new`，最后更新寄存器中的状态。输出 kernel 需要这两个中间量重建块内各位置的输出，因此不能用块末状态替代入口状态，也不能提前把尾部衰减合入已保存的 `v_new`。

`b_g_last` 取当前块最后一个有效 token 的累计 gate；尾块不足 64 时，整体衰减和残差衰减均按实际长度计算。序列结束后，寄存器状态写入 `final_state`，作为后续解码的初始状态。每条 packed 序列独立维护状态，序列边界通过 `cu_seqlens` 和 chunk 索引确定。

### 5.7 块内输出：历史项与局部项

所有块的入口状态和残差生成后，输出计算可以按块并行。定义包含缩放的局部权重 $`P=s(QK^\top\odot\Gamma)`$，则：

```math
O_{[t]}
=\underbrace{sDQH_t}_{\text{history}}
+\underbrace{P V_{\mathrm{new}}}_{\text{within chunk}},
\qquad P=s(QK^\top\odot\Gamma).
```

第一项读取块前历史，累计衰减 D 反映入口状态传播到各位置的强度。第二项汇总块内残差，$`\Gamma`$ 同时规定可见的历史位置及其传播衰减。kernel 分别计算 `Q @ h` 和 `Q @ K.T`，完成门控和因果掩码后，再将局部权重乘以 `v_new`。

```python
# b_h 是当前块入口状态；b_v 是上一阶段保存的 v_new。
b_o = tl.dot(b_q, b_h)
b_A = tl.dot(b_q, b_k)

b_o *= exp2(b_g)[:, None]
b_A *= exp2(b_g[:, None] - b_g[None, :])
m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
b_A = tl.where(m_A, b_A, 0.)
b_o = (b_o + tl.dot(b_A.to(b_v.dtype), b_v)) * scale
```

输出的因果掩码包含对角线，因为当前 token 的残差已经参与当前状态更新。KKT 的严格下三角掩码只编码此前位置对当前残差的影响，两者承担不同的计算作用。代码中的 `b_A` 是当前输出 kernel 的局部 QK 权重，与前面保存的三角逆矩阵 A 属于不同的中间量。

### 5.8 反向传播：状态伴随与三角变换梯度

令 $`E=V_{\mathrm{new}}`$，$`G_O=\partial\mathcal L/\partial O`$，$`G_{H_{t+1}}=\partial\mathcal L/\partial H_{t+1}`$。块内输出和块间状态两条路径共同形成 E 的梯度；入口状态的伴随则按逆序递推：

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

`chunk_bwd_dv_local` 计算局部输出贡献 $`P^\top G_O`$；`chunk_gated_delta_rule_bwd_dhu` 加入来自后续状态的贡献，并从后向前递推状态梯度。`chunk_bwd_dqkwg` 处理输出与状态更新对 Q/K/W 和 gate 的梯度；`prepare_wy_repr_bwd` 再通过三角变换将 W/U 梯度传回 K/V、beta 和 gate。

前向保存 A 及必要输入，反向先重算 W/U、入口状态和 `v_new`，以计算量换取较低的中间状态存储开销。最后的反向前缀和将累计 gate 的梯度汇总到逐 token gate。

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

Recurrent kernel 将每个序列、每个 value head 的状态沿 value 维划分。对于 scalar gate 分支，线程块尺寸与输出状态缓冲区为：

```math
\begin{aligned}
B_K&=2^{\lceil\log_2 d_k\rceil},\qquad
B_V=\min\!\left(8,2^{\lceil\log_2 d_v\rceil}\right),\\
N_V&=\left\lceil d_v/B_V\right\rceil,\qquad
\mathrm{grid}=(N_V,NH_V),\\
\mathcal H_{\mathrm{final}}&\in\mathbb R^{N\times H_V\times d_k\times d_v}.
\end{aligned}
```

每个 program 维护一个 `[BK, BV]` 状态片段，并沿时间维串行更新。不同 value 片段可以独立推进，因为每个片段只需要完整 key 向量和对应的 value 子向量。较小的 BV 限制单个 program 的状态寄存器用量；最终状态使用 FP32 保存，供下一次调用接续。

```python
BK = triton.next_power_of_2(K)
BV = min(8, triton.next_power_of_2(V))
NV = triton.cdiv(V, BV)
final_state = q.new_empty(N, HV, K, V, dtype=torch.float32)
grid = (NV, N * HV)
```

这里 N 为序列数，$`H_V`$ 为 value head 数。Q/K head 通过 `i_hv // (HV // H)` 映射，同一组 value head 共享 Q/K 输入，但各自维护独立的状态。prefill 写出的 chunk 末态与此处初始状态使用同一种缓冲区排列，二者之间不需要重新组织数学状态。

### 6.2 逐 Token 更新：衰减、残差与读出

以代码中的状态矩阵 $`H_t=S_t^\top`$ 表示递推。完成 Q/K 归一化后，一步更新由以下算式组成：

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

对 K 维的归约实现 $`\bar H_t^\top k_t`$，读出衰减后的旧 value；外积 $`k_te_t^\top`$ 将残差写入当前 key 方向；最后的归约实现状态对 query 的读出。完整 key 维包含在每个 program 内，因此这些归约不需要跨 program 通信。

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

代码在循环前加载初始状态，在循环后写回末态。逐 token 解码时通常只有一次迭代，Q/K/V 指针按各自的 head 数前进，gate 和 beta 则按 value head 前进。状态更新发生在寄存器中，输出与末态分别写回对应缓冲区。

该循环由逐元素乘法、外积和归约组成，没有矩阵乘法指令。Chunk 将多个时间步的依赖组织为三角求解与矩阵乘法，以提高整段序列计算的并行度；recurrent 则直接执行单步状态转移。当前 fused recurrent 实现未提供 backward，训练由 chunk 路径完成。

在数学上令 chunk 长度为 1，严格下三角矩阵 L 为零，$`A=I`$，W/U 变换直接给出单步残差。因此 chunk 的末态可以直接作为 recurrent 的初始状态，prefill 与解码之间保持同一个状态递推关系。

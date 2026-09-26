# Gated DeltaNet：模型调用、Chunk 与 Recurrent 实现

## 1. GDN 与 Token Mixer

Gated DeltaNet 将门控衰减与 Delta Rule 结合：前者控制历史状态的保留，后者沿当前 key 的方向修正 value 预测。它用固定大小的矩阵状态混合序列信息。

Token mixer 的输入先经过线性投影、短因果卷积和 SiLU，得到 Q/K/V；Q/K 还需 L2 归一化。衰减门和更新系数由输入的独立投影产生。Gated Delta Rule 计算出的输出再经过 RMSNorm、SiLU 输出门和线性投影。**状态更新中的门控与最后的输出门是两个不同的环节。**

Qwen3.5 将 GDN 层与标准注意力层组合使用。下面沿 GDN 层的实际调用顺序展开：第 4 节生成算子输入并选择执行路径，第 5 节处理整段序列，第 6 节接续缓存完成逐 token 解码。

## 2. Gated Delta Rule 的计算公式

对单个 head，$`q_t,k_t\in\mathbb R^{d_k}`$，$`v_t\in\mathbb R^{d_v}`$，状态 $`S_t\in\mathbb R^{d_v\times d_k}`$。GDN 的更新及其残差形式为：

```math
\begin{aligned}
S_t
&=\alpha_t S_{t-1}(I-\beta_t k_tk_t^\top)+\beta_t v_tk_t^\top\\
&=\alpha_t S_{t-1}
  +\underbrace{\beta_t(v_t-\alpha_t S_{t-1}k_t)}_{\text{value residual}}k_t^\top,\\
o_t&=S_tq_t.
\end{aligned}
```

计算顺序是：衰减状态 → 读取当前 key 对应的旧 value → 写入残差 → 用 query 读出。$`\alpha_t=1`$ 时退化为 Delta Rule；一般情况下，残差必须基于**已经衰减的状态**计算。

FLA 默认将状态按 `[K, V]` 存储，因此代码中的 `h` 对应 $`S^\top`$。公式省略 query 的默认缩放 $`d_k^{-1/2}`$，代码中由 `scale` 处理。

## 3. Chunk 的核心矩阵公式

将序列分成长度为 $`C`$ 的块。$`S_{[t]}`$ 表示第 $`t`$ 块的入口状态，块内 Q/K/V 按 token 排成行。省略块下标后，定义累计衰减及因果衰减矩阵：

```math
\gamma_i=\prod_{r=1}^{i}\alpha_r,\qquad
\Gamma_{ij}=\begin{cases}\gamma_i/\gamma_j,&i\ge j,\\0,&i<j.\end{cases}
```

WY 表示将连续的 Delta 更新组织成 W/U，UT 变换再把它们的逐行递推改写为下三角求解。令 $`B=\operatorname{diag}(\beta)`$，$`D=\operatorname{diag}(\gamma)`$，FLA 使用的矩阵为：

```math
\begin{aligned}
L&=\operatorname{strictLower}\!\left(B(\Gamma\odot KK^\top)\right),
& A&=(I+L)^{-1},\\
\widetilde U&=ABV,
&\overleftarrow W&=ABDK,\\
\overleftarrow Q&=DQ,
&\overrightarrow K&=\operatorname{diag}(\gamma_C/\gamma)K.
\end{aligned}
```

$`\widetilde U`$ 包含当前块内部的 Delta 修正；$`\overleftarrow W`$ 用于扣除块入口状态对 value 的预测。它对应论文中经过累计衰减缩放的 W，FLA 直接将这个量存入 `w`。

以论文的分块表达式为主线，状态与输出由同一个修正后的 value 块计算：

```math
\begin{aligned}
V_{\mathrm{new}}
  &=\widetilde U-\overleftarrow W S_{[t]}^\top,\\
S_{[t+1]}
  &=\gamma_C S_{[t]}+V_{\mathrm{new}}^\top\overrightarrow K,\\
O_{[t]}
  &=\overleftarrow Q S_{[t]}^\top
   +(QK^\top\odot\Gamma)V_{\mathrm{new}}.
\end{aligned}
```

这三行分别对应代码中的 `v_new`、块间状态更新和输出计算。输出第一项读取块前历史，第二项处理块内 token；后者的掩码 $`\Gamma`$ 同时包含因果约束与衰减，不能只保留 0/1 因果掩码。

## 4. 模型调用方式：Qwen3.5 GatedDeltaNet

### 4.1 投影与状态的维度

模型侧入口是 Transformers 的 `Qwen3_5MoeGatedDeltaNet.forward()`。Q/K 有 `num_k_heads` 个 head，V 有 `num_v_heads` 个 head。两类 head 的总维度分别为 `key_dim` 和 `value_dim`，核心投影层是：

```python
self.in_proj_qkv = nn.Linear(self.hidden_size, 2 * self.key_dim + self.value_dim, bias=False)
self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
```

`in_proj_qkv` 产生送入卷积的 Q/K/V；`in_proj_a` 与 `in_proj_b` 产生每个 value head 的衰减门和更新系数；`in_proj_z` 产生输出门。短卷积按通道独立计算，`groups` 等于 Q/K/V 的总通道数。每个 value head 维护一个 `[head_k_dim, head_v_dim]` 的状态矩阵。

### 4.2 从 hidden states 得到算子输入

下面保留无缓存的完整序列分支。代码摘录省略 padding 处理等外围逻辑，`batch_size`、`seq_len` 来自 `hidden_states.shape`。

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

`A_log` 和 `dt_bias` 是每个 value head 的可学习衰减参数。这里 `g = log(alpha)`，传入算子的不是 `alpha` 本身。负的 `g` 经指数还原为衰减系数；使用对数表示后，chunk 内的连续衰减可以通过前缀和计算。`beta` 则已完成 sigmoid，算子直接使用。

最后的 head 扩展实现 Grouped Value Attention：多个 value head 共享同一组 Q/K。当前 Transformers 仍显式复制 Q/K；FLA kernel 本身也支持用 head 索引映射处理共享关系。

有缓存的单 token 卷积分支通常调用 `causal_conv1d_update`，直接更新短卷积状态。它维护的是卷积窗口；下文 `recurrent_state` 保存的才是 GDN 矩阵状态，两者都需要跨解码步骤保留。

### 4.3 选择 Chunk 或 Recurrent，再完成输出

两种路径使用同一组输入和同一种状态布局。将源码中的重复调用参数合并后，分流及缓存更新的主干如下：

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

`use_qk_l2norm_in_kernel=True` 要求算子归一化 Q/K。名称中的 `torch_` 表示这里有 PyTorch fallback；启用对应的 FLA kernel 后，由 Transformers 的 kernel 分派机制替换实现。

整段训练和 prefill 通常进入 chunk；已有状态且 `seq_len == 1` 时进入 recurrent。两条路径返回的 `core_attn_out` 都按 value head 做 RMSNorm，再乘 `SiLU(z)`，最后经 `out_proj` 回到模型隐藏维度。这个输出门不会修改保存的 GDN 状态。

接下来先展开 chunk 如何得到 `core_attn_out` 和 `last_recurrent_state`，再看 recurrent 如何接着使用后者。

## 5. Chunk-wise 算法代码解析

### 5.1 前向传播总流程

FLA 的公共入口处理 Q/K 归一化及 `scale` 后，进入 `chunk_gated_delta_rule_fwd()`。下面保留默认 `chunk_size=64`、已计算好 `g`、无 context parallelism 的主干：

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

这里有三组必须区分的量：`A` 是下三角系统的逆，`u/w` 是其生成的变换结果，`v_new` 才是扣除入口状态预测后的 value。`h` 保存各块的**入口状态**，`final_state` 只保存整条序列的末态。

从算法看，`fwd_intra` 包含 KKT、下三角求解、W/U 三步。FLA v0.5.2 在默认 64-token 路径中将前两步融合到同一个 kernel，W/U 仍单独计算；16/32-token 路径保留分开的 KKT 与 `solve_tril` 调用。下面按这三步的数学含义分别说明。

以下 Triton 片段保留运算主干，省略地址计算、启动参数和重复的 K/V 分块；局部变量作必要简写，不是完整的可独立启动 kernel。

### 5.2 累积门控：chunk_local_cumsum

第 4 节传来的 `g` 是逐 token 的自然对数衰减。`chunk_local_cumsum` 在每块内部累计，再乘 `RCP_LN2`，后面的 kernel 用 `exp2` 还原：

```python
b_g = tl.cumsum(b_log_alpha, axis=0) * RCP_LN2
```

处理后的 `exp2(b_g[i])` 对应 $`\gamma_i`$，`exp2(b_g[i] - b_g[j])` 对应区间衰减。这份累计 gate 同时供 KKT、W/U、状态更新和输出使用；它在每个 chunk 开头重新累计，块前历史由 `h` 承接。

### 5.3 KKT：构造严格下三角矩阵 L

这一步对应第 3 节的 $`L`$，记录块内较早的 key 对当前位置 Delta 更新的影响：先计算 key 两两内积，乘区间衰减，再按行乘 `beta`。默认的 `chunk_gated_delta_rule_fwd_kkt_solve_kernel` 将 64×64 矩阵划分为 16×16 子块。下面是第一个对角块和相邻下三角块的核心计算：

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

`b_b0/b_b1` 是对应行的 beta，`m_tc0/m_tc1` 标记有效 token。对角子块用 `>` 去掉对角线；`b_A10` 整块位于下三角，只需处理序列边界。其余子块同样构造。

此时名字中的 `A` 仍是待求解的严格下三角部分，即公式中的 $`L`$。下一步才将它变成 $`(I+L)^{-1}`$。

### 5.4 solve_tril：从 L 得到 A

先分别求四个 16×16 对角块的逆。以下保留第一个子块的原始前代逻辑：

```python
b_Ai00 = -b_A00
for i in range(2, min(BC, T - i_tc0)):
    b_a00 = tl.sum(tl.where((o_i == i)[:, None], -b_A00, 0.), 0)
    b_a00 = tl.where(o_i < i, b_a00, 0.)
    b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
    b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
b_Ai00 += o_i[:, None] == o_i[None, :]
```

`BC=16`。初始化的负严格下三角部分已经包含第二行的结果，因此循环从第 3 行开始；最后加上单位对角线，得到这个对角子系统的逆。

对角块求解完成后，用矩阵乘法补齐整体逆矩阵的非对角块。下面保留最相邻的块和跨越两个子块的情况：

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

`b_Ai20` 同时包含经过第 0 块与第 1 块的依赖。这里是在合成整个下三角矩阵的逆，不能分别对 `b_A10`、`b_A20` 求逆。最终写出的 `A` 对应第 3 节的 $`(I+L)^{-1}`$，供下一步共同生成 W/U。

### 5.5 recompute_w_u_fwd：生成论文中的 W/U

`recompute_w_u_fwd_kernel` 读取求解后的 `A`。第一组乘法计算 $`\widetilde U=ABV`$，第二组计算已经吸收累计衰减的 $`\overleftarrow W=ABDK`$：

```python
b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
b_u = tl.dot(b_A, b_vb, allow_tf32=False)

b_kb = b_k * b_b[:, None]
b_kb *= exp2(b_g)[:, None]
b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
```

`u` 与 `w` 可以对各块独立计算，因为它们只依赖块内的 K/V、beta 和 gate。它们尚未使用块入口状态，所以 `u` 还不是最终写入状态的残差。下一步才计算 `v_new = u - w @ h`。

### 5.6 chunk_gated_delta_rule_fwd_h：递推块间状态

状态 kernel 沿 chunk 顺序推进。第一个块从 `initial_state` 开始，没有初始状态则置零；每个块先保存自己的入口状态，再产生下一块的状态。以下是默认 `[K, V]` 布局下单次迭代的主干：

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

这里 `b_k` 按 `[K, C]` 加载，因而最后一次 `tl.dot` 对应 $`\overrightarrow K^\top V_{\mathrm{new}}`$；转回论文的状态布局，就是 $`V_{\mathrm{new}}^\top\overrightarrow K`$。`b_g_last` 来自当前块最后一个有效 token，尾块不足 64 时也按实际长度计算衰减。

这一步需要保存两份不同用途的数据：入口 `h` 供本块输出读取，未乘尾部衰减的 `v_new` 供块内计算使用。更新后的 `b_h` 继续传给下一块；全部块处理完后，它成为第 4 节写入缓存的 `last_recurrent_state`。

对于 packed 输入，`cu_seqlens` 划定独立序列，chunk 索引负责定位每条序列的块。门控累积和状态递推都必须在序列边界重新开始，不能把前一条序列的末态传给后一条。

### 5.7 chunk_fwd_o：合并历史项与块内项

状态 kernel 完成后，各块的入口 `h` 和 `v_new` 已经就绪，输出 kernel 可以按块并行执行。它直接对应第 3 节的输出公式：先算历史项与 QK 点积，再分别施加衰减，最后乘 `v_new` 并相加。

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

这里 `b_A` 是输出 kernel 内的 QK 权重，不是前面保存的三角逆矩阵。`b_o` 的第一项对应 $`\overleftarrow Q S_{[t]}^\top`$，第二项对应 $`(QK^\top\odot\Gamma)V_{\mathrm{new}}`$。

输出掩码使用 `>=`，保留当前 token；KKT 阶段使用 `>`，只计算来自更早位置的修正。输出结果 `o` 返回第 4 节，继续经过输出门控和投影。

### 5.8 反向传播如何接回前向

训练时，`chunk_gated_delta_rule_bwd()` 先利用保存的 `A` 重算 W/U、块入口状态和 `v_new`，再沿前向依赖反向传播。下面保留固定长度、无 context parallelism 的调用主干：

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

`chunk_bwd_dv_local` 处理输出的块内项；`bwd_dhu` 从后往前传播状态依赖；`prepare_wy_repr_bwd` 再将 W/U 的梯度传回 K/V、beta 和 gate。最后的反向累积对应第 5.2 节的前缀和。这里的 checkpoint 策略是保存三角变换所需数据，反向重算较大的状态中间量。

## 6. Recurrent 算法代码解析

### 6.1 从 Chunk 的末态接续解码

prefill 完成后，第 5.6 节得到的 `final_state` 已写入缓存。下一个 token 到来时，第 4.3 节将它作为 `initial_state` 传给 `fused_recurrent_gated_delta_rule`。单 token 没有块内历史需要做三角求解，直接执行第 2 节的状态更新即可。

FLA 的 recurrent wrapper 按 value head 和 value 维度分块启动 kernel。默认状态布局下，缓冲区和 grid 的核心定义是：

```python
BK = triton.next_power_of_2(K)
BV = min(8, triton.next_power_of_2(V))
NV = triton.cdiv(V, BV)
final_state = q.new_empty(N, HV, K, V, dtype=torch.float32)
grid = (NV, N * HV)
```

这里展示 `output_final_state=True`、标量 gate 的分支。`N` 是序列数，`H/HV` 分别是 Q/K 与 V 的 head 数。每个 program 维护一个 `[BK, BV]` 状态片段；value head 对应的 Q/K head 用 `i_hv // (HV // H)` 定位。

GDN 缓存的大小随 head 数及 head 维度变化，不随已经生成的 token 数增长。它保存的是累计状态，另有短卷积窗口缓存；Qwen3.5 中标准注意力层的 KV cache 仍需另外维护。

### 6.2 Kernel：衰减、残差写入与读出

`fused_recurrent_gated_delta_rule_fwd_kernel` 先读入 `initial_state`，然后沿输入时间步执行以下循环。下面保留 scalar gate、headwise beta 和默认状态布局：

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

中间四行与第 2 节公式逐项对应：`exp(b_g)` 衰减状态；对 K 维归约读取旧 value；乘 beta 的残差通过外积写回；最后对 query 归约得到输出。解码时通常 `T=1`，读入一次缓存，完成一次更新，再写回新的缓存。

这个循环使用逐元素运算、外积和归约，没有 `tl.dot`。长序列的逐 token 串行依赖由 chunk 算法改写为块内矩阵乘法；单 token 解码则无需构造 KKT、W/U 等中间量。当前 FLA 的 fused recurrent 实现未提供 backward，训练使用前一节的 chunk 路径。

当 chunk 长度缩为 1 时，严格下三角部分为空，三角求解退化为单位变换，`v_new` 就是这一循环写入的 value 残差。因此 prefill 的 chunk 末态可以直接接续 recurrent；两者改变的是计算组织方式，维护的是同一个 GDN 状态。

---

参考：[原文](https://zhuanlan.zhihu.com/p/2007937984738129405)、[Gated Delta Networks 论文](https://arxiv.org/abs/2412.06464)。模型代码对照 [Transformers v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py)（[Apache-2.0](https://github.com/huggingface/transformers/blob/v5.17.0/LICENSE)），算子代码对照 [FLA v0.5.2](https://github.com/fla-org/flash-linear-attention/tree/v0.5.2/fla/ops/gated_delta_rule)（[MIT](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/LICENSE)）。代码摘录保留上述执行分支的主干，核对日期：2026-09-26。

# Gated DeltaNet：模型调用、Chunk 与 Recurrent 实现

## 1. GDN 与 Token Mixer

- Gated DeltaNet 在 Delta Rule 中引入数据相关的衰减门，同时控制历史状态的保留与当前键值对的写入。其 token mixer 由输入投影、短因果卷积、状态更新、输出归一化与门控组成。

<figure style="margin: 24px 0;">
  <a href="../../assets/gdn-architecture.png"><img src="../../assets/gdn-architecture.png" alt="Gated DeltaNet 论文模型图：左侧为 H1，中间为 H2，右侧为 token mixer 的投影、卷积、L2 归一化、Gated Delta Rule 与输出门控" width="2058" height="1122"></a>
</figure>

- 图右侧的 Q/K/V 分支经过线性投影、短卷积与 SiLU，Q/K 进一步进行 L2 归一化。衰减门与更新系数由输入的独立投影产生，分别作用于历史状态和 value 残差。状态读出结果经过归一化后与输出门逐元素相乘，再投影回模型隐藏维度。输出门只调节本层输出，不参与状态递推。

## 2. Gated Delta Rule 的计算公式

- 对单个 head，令 $`q_t,k_t\in\mathbb R^{d_k}`$，$`v_t\in\mathbb R^{d_v}`$，状态 $`S_t\in\mathbb R^{d_v\times d_k}`$。状态转移及其残差形式为：

```math
\begin{aligned}
S_t
&=\alpha_t S_{t-1}(I-\beta_t k_tk_t^\top)+\beta_t v_tk_t^\top\\
&=\alpha_t S_{t-1}
  +\beta_t\bigl(v_t-\alpha_t S_{t-1}k_t\bigr)k_t^\top,\\
o_t&=sS_tq_t,\qquad s=d_k^{-1/2}.
\end{aligned}
```

- $`\alpha_t`$ 对历史状态施加整体衰减，$`\beta_t`$ 控制沿当前 key 方向的更新幅度。残差中的旧 value 由衰减后的状态读取，因此衰减必须先于残差计算。写入完成后，再用 query 读取更新后的状态。

## 3. Chunk 的矩阵表示

- 将序列划分为长度为 $`C`$ 的块，块内 Q/K/V 按 token 排成行。累计衰减与因果衰减矩阵定义为：

```math
\gamma_i=\prod_{r=1}^{i}\alpha_r,\qquad
\Gamma_{ij}=\begin{cases}\gamma_i/\gamma_j,&i\ge j,\\0,&i<j.\end{cases}
```

- WY 表示将连续的 Delta 更新压缩为两个变换矩阵；UT 变换进一步将其递推关系写成单位下三角系统。记 $`B=\operatorname{diag}(\beta)`$、$`D=\operatorname{diag}(\gamma)`$，则：

```math
\begin{aligned}
L&=\operatorname{strictLower}\!\left(B(\Gamma\odot KK^\top)\right),\\
(I+L)\widetilde U&=BV,\qquad
(I+L)\overleftarrow W=BDK.
\end{aligned}
```

- $`L`$ 编码当前块内部较早 key 对后续位置的影响。求解同一个下三角系统，可以同时得到 value 变换 $`\widetilde U`$ 和作用于块入口状态的变换 $`\overleftarrow W`$。扣除入口状态的预测后，得到真正参与状态写入与输出计算的 $`V_{\mathrm{new}}`$。

## 4. 模型调用方式：Qwen3.5 GatedDeltaNet

### 4.1 输入投影

- 设输入为 $`X\in\mathbb R^{T\times d_{\mathrm{model}}}`$。Q/K 的总投影维度为 $`d_Q=H_Kd_k`$，V 的总投影维度为 $`d_V=H_Vd_v`$。四个线性映射分别产生卷积输入、衰减参数、写入参数和输出门：

```math
P=XW_{qkv}^{\top},\qquad a=XW_a^{\top},\quad b=XW_b^{\top},\qquad Z=XW_z^{\top}.
```

- `Qwen3_5MoeGatedDeltaNet` 将 Q/K/V 合并为一次投影，随后按通道拆分。衰减参数和写入参数只需要每个 value head 一个标量，因此 `in_proj_a`、`in_proj_b` 的输出维度为 `num_v_heads`；输出门需要逐通道调节读出结果，其投影维度与 V 相同。

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

- 上一步得到的 L 是严格下三角部分，因此 $`M=I+L`$ 的对角线为 1。求解 $`MA=I`$ 时，逐行前代只需使用此前已求出的行；无需调用通用矩阵求逆。标量递推与分块矩阵关系为：

```math
\begin{aligned}
A_{ii}&=1,\qquad
A_{ij}=-L_{ij}-\sum_{k=j+1}^{i-1}L_{ik}A_{kj}\quad(i>j),\\
\mathcal A_{rr}&=(I+\mathcal L_{rr})^{-1},\qquad
\mathcal A_{rs}=-\mathcal A_{rr}\sum_{p=s}^{r-1}\mathcal L_{rp}\mathcal A_{ps}\quad(r>s).
\end{aligned}
```

- $`\mathcal L_{rs}`$、$`\mathcal A_{rs}`$ 分别表示 L 和 A 的 16×16 子块。以四块下三角矩阵为例，整体求解关系为：

```math
\begin{pmatrix}
I+\mathcal L_{11}&0&0&0\\
\mathcal L_{21}&I+\mathcal L_{22}&0&0\\
\mathcal L_{31}&\mathcal L_{32}&I+\mathcal L_{33}&0\\
\mathcal L_{41}&\mathcal L_{42}&\mathcal L_{43}&I+\mathcal L_{44}
\end{pmatrix}
\begin{pmatrix}
\mathcal A_{11}&0&0&0\\
\mathcal A_{21}&\mathcal A_{22}&0&0\\
\mathcal A_{31}&\mathcal A_{32}&\mathcal A_{33}&0\\
\mathcal A_{41}&\mathcal A_{42}&\mathcal A_{43}&\mathcal A_{44}
\end{pmatrix}=I.
```

- `fla/ops/utils/solve_tril.py` 采用两级分块：先分别求四个 16×16 对角块的逆，再用矩阵乘法补齐六个非对角块。`merge_16x16_to_64x64_inverse_kernel` 在同一个 program 内完成两级操作；下面保留普通指针加载分支。输入参数 `A` 保存 L，输出参数 `Ai` 保存 $`(I+L)^{-1}`$，其上三角区域由调用侧预置为零。

```python
@triton.jit(do_not_specialize=['T'])
def merge_16x16_to_64x64_inverse_kernel(
    A, Ai, cu_seqlens, chunk_indices, T, H: tl.constexpr, BT: tl.constexpr,
    USE_TMA: tl.constexpr, IS_VARLEN: tl.constexpr, DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    o_t = (i_t * BT + o_i).to(tl.int64)
    p_A_11 = A + o_t[:, None] * (H*BT) + o_i[None, :]
    p_A_22 = A + (o_t[:, None] + 16) * (H*BT) + (o_i[None, :] + 16)
    p_A_33 = A + (o_t[:, None] + 32) * (H*BT) + (o_i[None, :] + 32)
    p_A_44 = A + (o_t[:, None] + 48) * (H*BT) + (o_i[None, :] + 48)
    b_Ai_11 = tl.load(p_A_11, mask=(o_t[:, None] < T), other=0.0).to(tl.float32)
    b_Ai_22 = tl.load(p_A_22, mask=(o_t[:, None] + 16 < T), other=0.0).to(tl.float32)
    b_Ai_33 = tl.load(p_A_33, mask=(o_t[:, None] + 32 < T), other=0.0).to(tl.float32)
    b_Ai_44 = tl.load(p_A_44, mask=(o_t[:, None] + 48 < T), other=0.0).to(tl.float32)

    # [16, 16]
    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)
    b_Ai_33 = -tl.where(m_A, b_Ai_33, 0)
    b_Ai_44 = -tl.where(m_A, b_Ai_44, 0)

    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(A + (i_t * BT + i) * H*BT + o_i)
        b_a_11 = tl.where(o_i < i, b_a_11, 0.)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 16)
        b_a_22 = tl.where(o_i < i - 16, b_a_22, 0.)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
    for i in range(32 + 2, min(48, T - i_t * BT)):
        b_a_33 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 32)
        b_a_33 = tl.where(o_i < i - 32, b_a_33, 0.)
        b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)
        b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a_33, b_Ai_33)
    for i in range(48 + 2, min(64, T - i_t * BT)):
        b_a_44 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 48)
        b_a_44 = tl.where(o_i < i - 48, b_a_44, 0.)
        b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)
        b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a_44, b_Ai_44)
    b_Ai_11 += m_I
    b_Ai_22 += m_I
    b_Ai_33 += m_I
    b_Ai_44 += m_I

    p_A_21 = A + (o_t[:, None] + 16) * (H*BT) + o_i[None, :]
    p_A_31 = A + (o_t[:, None] + 32) * (H*BT) + o_i[None, :]
    p_A_32 = A + (o_t[:, None] + 32) * (H*BT) + (o_i[None, :] + 16)
    p_A_41 = A + (o_t[:, None] + 48) * (H*BT) + o_i[None, :]
    p_A_42 = A + (o_t[:, None] + 48) * (H*BT) + (o_i[None, :] + 16)
    p_A_43 = A + (o_t[:, None] + 48) * (H*BT) + (o_i[None, :] + 32)
    b_A_21 = tl.load(p_A_21, mask=(o_t[:, None] + 16 < T), other=0.0).to(tl.float32)
    b_A_31 = tl.load(p_A_31, mask=(o_t[:, None] + 32 < T), other=0.0).to(tl.float32)
    b_A_32 = tl.load(p_A_32, mask=(o_t[:, None] + 32 < T), other=0.0).to(tl.float32)
    b_A_41 = tl.load(p_A_41, mask=(o_t[:, None] + 48 < T), other=0.0).to(tl.float32)
    b_A_42 = tl.load(p_A_42, mask=(o_t[:, None] + 48 < T), other=0.0).to(tl.float32)
    b_A_43 = tl.load(p_A_43, mask=(o_t[:, None] + 48 < T), other=0.0).to(tl.float32)

    b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION), b_Ai_11, input_precision=DOT_PRECISION)
    b_Ai_32 = -tl.dot(tl.dot(b_Ai_33, b_A_32, input_precision=DOT_PRECISION), b_Ai_22, input_precision=DOT_PRECISION)
    b_Ai_43 = -tl.dot(tl.dot(b_Ai_44, b_A_43, input_precision=DOT_PRECISION), b_Ai_33, input_precision=DOT_PRECISION)

    b_Ai_31 = -tl.dot(
        b_Ai_33,
        tl.dot(b_A_31, b_Ai_11, input_precision=DOT_PRECISION) +
        tl.dot(b_A_32, b_Ai_21, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai_42 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_42, b_Ai_22, input_precision=DOT_PRECISION) +
        tl.dot(b_A_43, b_Ai_32, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai_41 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_41, b_Ai_11, input_precision=DOT_PRECISION) +
        tl.dot(b_A_42, b_Ai_21, input_precision=DOT_PRECISION) +
        tl.dot(b_A_43, b_Ai_31, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )

    p_Ai_11 = Ai + o_t[:, None] * (H*BT) + o_i[None, :]
    p_Ai_22 = Ai + (o_t[:, None] + 16) * (H*BT) + (o_i[None, :] + 16)
    p_Ai_33 = Ai + (o_t[:, None] + 32) * (H*BT) + (o_i[None, :] + 32)
    p_Ai_44 = Ai + (o_t[:, None] + 48) * (H*BT) + (o_i[None, :] + 48)
    p_Ai_21 = Ai + (o_t[:, None] + 16) * (H*BT) + o_i[None, :]
    p_Ai_31 = Ai + (o_t[:, None] + 32) * (H*BT) + o_i[None, :]
    p_Ai_32 = Ai + (o_t[:, None] + 32) * (H*BT) + (o_i[None, :] + 16)
    p_Ai_41 = Ai + (o_t[:, None] + 48) * (H*BT) + o_i[None, :]
    p_Ai_42 = Ai + (o_t[:, None] + 48) * (H*BT) + (o_i[None, :] + 16)
    p_Ai_43 = Ai + (o_t[:, None] + 48) * (H*BT) + (o_i[None, :] + 32)
    tl.store(p_Ai_11, b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"), mask=(o_t[:, None] < T))
    tl.store(p_Ai_22, b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"), mask=(o_t[:, None] + 16 < T))
    tl.store(p_Ai_33, b_Ai_33.to(p_Ai_33.dtype.element_ty, fp_downcast_rounding="rtne"), mask=(o_t[:, None] + 32 < T))
    tl.store(p_Ai_44, b_Ai_44.to(p_Ai_44.dtype.element_ty, fp_downcast_rounding="rtne"), mask=(o_t[:, None] + 48 < T))
    tl.store(p_Ai_21, b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"), mask=(o_t[:, None] + 16 < T))
    tl.store(p_Ai_31, b_Ai_31.to(p_Ai_31.dtype.element_ty, fp_downcast_rounding="rtne"), mask=(o_t[:, None] + 32 < T))
    tl.store(p_Ai_32, b_Ai_32.to(p_Ai_32.dtype.element_ty, fp_downcast_rounding="rtne"), mask=(o_t[:, None] + 32 < T))
    tl.store(p_Ai_41, b_Ai_41.to(p_Ai_41.dtype.element_ty, fp_downcast_rounding="rtne"), mask=(o_t[:, None] + 48 < T))
    tl.store(p_Ai_42, b_Ai_42.to(p_Ai_42.dtype.element_ty, fp_downcast_rounding="rtne"), mask=(o_t[:, None] + 48 < T))
    tl.store(p_Ai_43, b_Ai_43.to(p_Ai_43.dtype.element_ty, fp_downcast_rounding="rtne"), mask=(o_t[:, None] + 48 < T))
```

- **第一级：16×16 对角块前代。** `b_Ai_11` 等矩阵初始化为负严格下三角部分，随后从索引 2 开始逐行更新：索引 0 没有下三角元素，索引 1 的结果已经由初始化给出。`tl.sum(b_a[:, None] * b_Ai, 0)` 汇总已知行的贡献，`tl.where` 将新结果写入当前行，最后加上单位矩阵。四个对角子问题相互独立，但每个子问题内部仍有逐行依赖。

- **第二级：合并为 64×64 逆矩阵。** 先计算相邻子块 `b_Ai_21`、`b_Ai_32`、`b_Ai_43`，再计算跨一个子块的 `b_Ai_31`、`b_Ai_42`，最后计算 `b_Ai_41`。例如 $`\mathcal A_{31}=-\mathcal A_{33}(\mathcal L_{31}\mathcal A_{11}+\mathcal L_{32}\mathcal A_{21})`$，同时包含直接依赖和经第 2 块传递的依赖。非对角位置保存的是整体逆矩阵的子块，并非对输入子块各自求逆。

- **执行路径。** `BT=16` 使用 `solve_tril_16x16_kernel`，`BT=32` 使用 `merge_16x16_to_32x32_inverse_kernel`，通用 `BT=64` 使用上面的 kernel。GDN 默认 64-token 路径则在 `chunk_gated_delta_rule_fwd_kkt_solve_kernel` 内直接求解寄存器中的 KKT 子块：逐行数据通过 `tl.sum(tl.where(...))` 提取，不再从全局内存重新加载；局部前代与块合并的矩阵关系不变。

- **访存与精度。** 通用 `solve_tril` 在环境支持时使用 TMA descriptor 加载、存储 16×16 子块；`FLA_TRIL_PRECISION` 默认 `ieee`，支持 TMA 时 autotune 的精度候选为 `ieee` 与用户指定值。默认 GDN 融合 kernel 使用普通指针加载，合并精度由 `SOLVE_TRIL_DOT_PRECISION` 决定：支持 TF32 时为 `tf32`，否则为 `ieee`。融合省去 L 的中间写回和重新加载，同时增加寄存器中间量，不能据此直接推断所有形状下都更快。

### 5.5 W/U：共享同一个三角变换

- 求得 $`A`$ 后，将两个右端项分别代入同一个线性系统：

```math
\widetilde U=A(BV),\qquad \overleftarrow W=A(BDK),\qquad A=(I+L)^{-1}.
```

- 算子位于 `fla/ops/gated_delta_rule/wy_fast.py`。`recompute_w_u_fwd_kernel` 的网格为 `(NT, B * HV)`，每个 program 读取一个块的逆矩阵 `A`，将 beta 按行乘到 V 和 K 上，再执行两组矩阵乘法。K 分支还需要先乘累计衰减，得到右端项 `BDK`。这个缩放发生在乘 A 之前，因为一般情况下，对角门控矩阵 D 与下三角逆矩阵 A 不能交换。

```python
@triton.jit(do_not_specialize=['T'])
def recompute_w_u_fwd_kernel(
    k, v, beta, w, u, A, g, cu_seqlens, chunk_indices, T, H: tl.constexpr, HV: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_G: tl.constexpr, IS_VARLEN: tl.constexpr,
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

- `b_A` 在 program 内复用：V 维按 `BV` 遍历，逐片计算并写回 `u`；K 维按 `BK` 遍历，逐片计算并写回 `w`。`b_vb`、`b_kb` 分别是右端项 BV、BDK。A 本身不含右端项的 beta，因此此处只需乘一次。

- W/U 只依赖块内输入，可以按 chunk 独立生成。`u` 已包含当前块内部的 Delta 修正，但尚未扣除块入口状态对 value 的预测；`w` 表示该预测应经过的线性变换。两者在下一节通过 `u - w @ h` 合成实际残差。

### 5.6 块间状态：残差校正与衰减写入

- 记块入口状态为 $`H_t=S_{[t]}^\top`$，$`R=\operatorname{diag}(\gamma_C/\gamma)`$。单次块更新为：

```math
\begin{aligned}
V_{\mathrm{new}}&=\widetilde U-\overleftarrow W H_t,\\
H_{t+1}&=\gamma_C H_t+K^\top R V_{\mathrm{new}}.
\end{aligned}
```

- 第一式从块内变换结果中减去入口状态的预测，得到各位置的写入残差。第二式将入口状态衰减到块末，并把各位置残差经过剩余时间步的衰减后累积进去。代码将 R 乘在 `v_new` 上，再与按 `[K, C]` 加载的 key 相乘，等价于公式中的 $`K^\top R V_{\mathrm{new}}`$。

- 算子位于 `fla/ops/common/chunk_delta_h.py`。网格为 `(ceil(V / BV), N * HV)`：每个 program 负责一条序列、一个 value head 和一片 value 通道，并沿 chunk 顺序递推。下面展开 `K=128`、`STATE_V_FIRST=False` 的标量 gate 分支，保留两个 `[64, BV]` 状态片段；参数 `v` 实际接收上一阶段的 `u`，`SAVE_NEW_VALUE=True`。

```python
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k, v, w, v_new, g, gk, h, h0, ht, cu_seqlens, chunk_offsets, T, H: tl.constexpr,
    HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr,
    USE_G: tl.constexpr, USE_GK: tl.constexpr, USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr, SAVE_NEW_VALUE: tl.constexpr,
    STATE_V_FIRST: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    b_h2 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += (boh * HV + i_h).to(tl.int64) * K*V
    v += (bos * HV + i_h).to(tl.int64) * V
    k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    v_new += (bos * HV + i_h).to(tl.int64) * V

    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K*V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K*V

    # load initial state
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    o_k3 = 128 + o_k1
    m_k3 = o_k3 < K
    o_k4 = 192 + o_k1
    m_k4 = o_k4 < K
    if USE_INITIAL_STATE:
        p_h0_1 = h0 + o_k1[:, None] * V + o_v[None, :]
        m_h0_1 = m_k1[:, None] & m_v[None, :]
        b_h1 += tl.load(p_h0_1, mask=m_h0_1, other=0.0).to(tl.float32)
        p_h0_2 = h0 + o_k2[:, None] * V + o_v[None, :]
        m_h0_2 = m_k2[:, None] & m_v[None, :]
        b_h2 += tl.load(p_h0_2, mask=m_h0_2, other=0.0).to(tl.float32)

    # main recurrence
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

    if STORE_FINAL_STATE:
        p_ht = ht + o_k1[:, None] * V + o_v[None, :]
        m_ht = m_k1[:, None] & m_v[None, :]
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), mask=m_ht)
        p_ht = ht + o_k2[:, None] * V + o_v[None, :]
        m_ht = m_k2[:, None] & m_v[None, :]
        tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), mask=m_ht)
```

- `b_h1`、`b_h2` 覆盖 key 维的两个 64 通道片段。计算 `w @ h` 时，两次 `tl.dot` 的结果先相加，再从 u 中扣除；状态写入时，相同的残差分别与两片 key 相乘。这是沿 K 维展开的同一个矩阵乘法，两个状态片段不是独立的时间递推。源码以相同方式扩展到 `K ≤ 256`。

- 状态 kernel 沿 chunk 顺序执行。每次迭代先保存入口状态，再保存尚未乘 R 的 `v_new`，最后更新寄存器中的状态。输出 kernel 需要这两个中间量重建块内各位置的输出，因此不能用块末状态替代入口状态，也不能提前把尾部衰减合入已保存的 `v_new`。

- `b_g_last` 取当前块最后一个有效 token 的累计 gate；尾块不足 64 时，整体衰减和残差衰减均按实际长度计算。序列结束后，寄存器状态写入 `final_state`，作为后续解码的初始状态。每条 packed 序列独立维护状态：`cu_seqlens` 给出 token 起止位置，`chunk_offsets` 给出此前序列占用的 chunk 数。`boh` 用于定位该序列在入口状态缓冲区中的起点，与 token 偏移 `bos` 分别计算。

### 5.7 块内输出：历史项与局部项

- 所有块的入口状态和残差生成后，输出计算可以按块并行。定义包含缩放的局部权重 $`P=s(QK^\top\odot\Gamma)`$，则：

```math
O_{[t]}
=\underbrace{sDQH_t}_{\text{history}}
+\underbrace{P V_{\mathrm{new}}}_{\text{within chunk}},
\qquad P=s(QK^\top\odot\Gamma).
```

- 第一项读取块前历史，累计衰减 D 反映入口状态传播到各位置的强度。第二项汇总块内残差，$`\Gamma`$ 同时规定可见的历史位置及其传播衰减。算子位于 `fla/ops/common/chunk_o.py`，网格为 `(ceil(V / BV), NT, B * HV)`。`chunk_fwd_kernel_o` 分别计算 `Q @ h` 和 `Q @ K.T`，完成门控和因果掩码后，再将局部权重乘以 `v_new`。

```python
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_o(
    q, k, v, h, g, g_gamma, o, cu_seqlens, chunk_indices, scale, T, H: tl.constexpr,
    HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
    BV: tl.constexpr, USE_G: tl.constexpr, USE_G_GAMMA: tl.constexpr,
    STATE_V_FIRST: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    q += (bos * H + i_h // (HV // H)) * K
    k += (bos * H + i_h // (HV // H)) * K
    v += (bos * HV + i_h) * V
    o += (bos * HV + i_h) * V
    h += (i_tg * HV + i_h).to(tl.int64) * K*V

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

- K 维循环同时累加历史项 `[BT, BV]` 和局部权重 `[BT, BT]`。Q 按 `[BT, BK]`、K 按 `[BK, BT]`、入口状态按 `[BK, BV]` 加载，使两个乘法共享同一份 Q。`v` 指向前一阶段保存的 `v_new`，局部权重完成门控后再与它相乘；这里只展开 GDN 的标量 gate 分支。

- 变长输入中，`i_tg` 保留全局 chunk 编号以读取入口状态，`i_t` 则转换为序列内 chunk 编号以读取 Q/K/V。二者不能混用，否则第二条及后续序列会读取错误的入口状态。

- 输出的因果掩码包含对角线，因为当前 token 的残差已经参与当前状态更新。KKT 的严格下三角掩码只编码此前位置对当前残差的影响，两者承担不同的计算作用。代码中的 `b_A` 是当前输出 kernel 的局部 QK 权重，与前面保存的三角逆矩阵 A 属于不同的中间量。

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

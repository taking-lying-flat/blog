# Gated Delta Rule 与 Chunk 算法实现

Gated Delta Rule 用门控衰减历史状态，再沿当前 key 的方向写入预测残差。逐 token 递推用于解码；chunk 算法在块间递推状态，将块内计算转为矩阵乘法。

## 1. Gated Delta Rule

对单个 head，状态为 $`S_t\in\mathbb R^{d_v\times d_k}`$，更新规则为：

```math
\begin{aligned}
S_t &= \alpha_t S_{t-1}
  + \beta_t\bigl(v_t-\alpha_t S_{t-1}k_t\bigr)k_t^\top,\\
o_t &= s\,S_tq_t,\qquad s=d_k^{-1/2}.
\end{aligned}
```

$`\alpha_t`$ 控制衰减，$`\beta_t`$ 控制残差写入强度。FLA 默认以 `[K, V]` 布局保存转置状态 `h = S.T`。`fused_recurrent_gated_delta_rule_fwd_kernel` 的核心是：

```python
b_q = b_q * scale
b_h *= exp(b_g)
b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
b_h += b_k[:, None] * b_v
b_o = tl.sum(b_h * b_q[:, None], 0)
```

这里 `b_g = log(alpha)`，`b_beta` 是已激活的更新系数。先衰减状态，再计算 value 残差，最后写入并读出。以下代码均保留这一标量 gate 分支，省略地址计算、加载存储及重复的维度分块；局部变量按计算含义简写。

## 2. Chunk 的矩阵形式

每块包含 $`C`$ 个 token，$`Q,K\in\mathbb R^{C\times d_k}`$，$`V\in\mathbb R^{C\times d_v}`$。块内累计衰减与因果衰减矩阵定义为：

```math
\begin{gathered}
\gamma_i=\prod_{r=1}^{i}\alpha_r,\qquad
\Gamma_{ij}=\begin{cases}\gamma_i/\gamma_j,&i\ge j,\\0,&i<j,\end{cases}\\
B=\operatorname{diag}(\beta),\quad
D=\operatorname{diag}(\gamma),\quad
R=\operatorname{diag}(\gamma_C/\gamma).
\end{gathered}
```

WY／UT 表示将块内递推归结为单位下三角求解。令 $`\operatorname{tril}_{-1}`$ 只保留严格下三角，整个 chunk 的计算为：

```math
\begin{aligned}
L &= \operatorname{tril}_{-1}\!\left(B(KK^\top\odot\Gamma)\right),
& A&=(I+L)^{-1},\\
U &= ABV,
& W&=ABDK,\\
E &= U-WS_{\mathrm{in}}^\top,\\
S_{\mathrm{out}} &= \gamma_C S_{\mathrm{in}}+E^\top RK,\\
O &= s\left[DQS_{\mathrm{in}}^\top+(QK^\top\odot\Gamma)E\right].
\end{aligned}
```

这里的 $`W`$ 已包含门控缩放，对应 FLA 的 `w`；$`U`$ 对应 `u`，$`E`$ 对应 `v_new`。块间只传递状态，块内的下三角求解与矩阵乘法可并行计算。

## 3. Chunk 核心代码

FLA v0.5.2 默认 `chunk_size=64`。执行顺序为：门控累积 → KKT 与下三角求解 → W/U → 状态更新 → 输出。默认路径融合 KKT 与下三角求解，W/U 仍由独立 kernel 计算。

### 3.1 门控累积

`chunk_local_cumsum` 对每块的 `log(alpha)` 做前缀和，并换为以 2 为底，后续统一使用 `exp2`：

```python
b_g = tl.cumsum(b_log_alpha, axis=0) * RCP_LN2
```

因此 `exp2(b_g[i])` 对应块内的累计衰减 `gamma[i]`，每个 chunk 重新开始累计。

### 3.2 KKT 与下三角求解

`chunk_gated_delta_rule_fwd_kkt_solve_kernel` 将 64×64 矩阵划分为 16×16 子块。下面保留第一个对角块和相邻下三角块的构造；`b_g0/b_g1` 是对应的累计 gate，`b_b0/b_b1` 是 beta，`m_tc0/m_tc1` 标记有效 token。

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

此时 `b_A00` 保存 $`L`$ 的第一个对角子块。先在寄存器中逐行求解，得到 `b_Ai00 = (I + b_A00)^{-1}`：

```python
b_Ai00 = -b_A00
for i in range(2, min(BC, T - i_tc0)):
    b_a00 = tl.sum(tl.where((o_i == i)[:, None], -b_A00, 0.), 0)
    b_a00 = tl.where(o_i < i, b_a00, 0.)
    b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
    b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
b_Ai00 += o_i[:, None] == o_i[None, :]
```

其余对角块同样求解，再用矩阵乘法补齐逆矩阵的下三角子块。例如：

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

`b_A00/b_A10` 是 $`L`$ 的子块，`b_Ai00/b_Ai10` 是整体逆矩阵 $`A`$ 的对应子块；非对角子块不能单独求逆。求解结果写回后，由 W/U kernel 读取。

### 3.3 计算 W/U

`recompute_w_u_fwd_kernel` 的两次矩阵乘法分别对应 $`U=ABV`$ 与 $`W=ABDK`$。此处 `b_A` 已是求解后的逆矩阵，`b_g` 是累计 gate 的对数值。

```python
b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
b_u = tl.dot(b_A, b_vb, allow_tf32=False)

b_kb = b_k * b_b[:, None]
b_kb *= exp2(b_g)[:, None]
b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
```

### 3.4 更新块间状态

`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` 沿 chunk 顺序推进。`b_h` 是块入口状态，`b_k` 在这里按 `[K, C]` 加载，`b_g_last` 取块内最后一个有效 token 的累计 gate。

```python
b_v_new = b_u - tl.dot(b_w, b_h.to(b_w.dtype))

b_v = b_v_new * tl.where(
    m_t, exp2(b_g_last - b_g), 0.,
)[:, None]
b_h *= exp2(b_g_last)
b_h += tl.dot(b_k, b_v.to(b_k.dtype))
```

`v_new` 必须在乘尾部衰减前保存；输出 kernel 使用它和当前块的入口状态，更新后的 `b_h` 则传给下一块。

### 3.5 计算块内输出

`chunk_fwd_kernel_o` 将块入口状态的贡献与块内 token 的贡献相加。这里的 `b_h` 为入口状态，`b_v` 为 `v_new`，`b_k` 仍按 `[K, C]` 加载。

```python
b_o = tl.dot(b_q, b_h)
b_A = tl.dot(b_q, b_k)

b_o *= exp2(b_g)[:, None]
b_A *= exp2(b_g[:, None] - b_g[None, :])
m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
b_A = tl.where(m_A, b_A, 0.)
b_o = (b_o + tl.dot(b_A.to(b_v.dtype), b_v)) * scale
```

KKT 阶段使用严格下三角，输出阶段保留对角线：当前 token 的残差已写入状态，应参与当前位置的输出。

---

参考：[原文](https://zhuanlan.zhihu.com/p/2007937984738129405)、[Gated Delta Networks 论文](https://arxiv.org/abs/2412.06464)。代码按 [FLA v0.5.2](https://github.com/fla-org/flash-linear-attention/tree/v0.5.2/fla/ops/gated_delta_rule)（[MIT 许可](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/LICENSE)）摘取并简化；模型侧已对照 [Transformers v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)。核对日期：2026-09-26。

# Qwen3.5 Gated DeltaNet：从递推公式到 FLA 内核

Qwen3.5 的 Gated DeltaNet 用一个状态矩阵保存历史。每个 token 先衰减旧状态，再用当前 key 读出预测 value，把预测误差写回，最后用 query 得到输出。本文从 `Qwen3_5GatedDeltaNet.forward()` 出发，沿着这四步进入 FLA：先看逐 token 的 recurrent kernel，再推导它如何变成 chunk kernel 中的下三角求解和矩阵乘法。

代码固定到截至 2026-09-26 的最新正式版 **Transformers v5.17.0 / FLA v0.5.2**。正文展示真实源码节选，省略的相邻分支通过链接查看。FLA 部分采用默认 `chunk_size=64`、`state_v_first=False`，不展开 context parallel 分支。

## 1. 模型入口：准备 Q、K、V 和三个门

`forward()` 收到的 `hidden_states` 形状为 `[B, T, D]`。输入先经过四组线性投影：一组生成拼接的 QKV，另外三组生成输出门 `z`、写入门的原始值 `b`、衰减门的原始值 `a`。

[modeling_qwen3_5.py · L565–L572](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L565-L572)

```python
mixed_qkv = self.in_proj_qkv(hidden_states)
mixed_qkv = mixed_qkv.transpose(1, 2)

z = self.in_proj_z(hidden_states)
z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

b = self.in_proj_b(hidden_states)
a = self.in_proj_a(hidden_states)
```

`mixed_qkv` 接着经过逐通道的因果卷积和 SiLU；`z`、`a`、`b` 不经过这层卷积。训练或 prefill 使用整段卷积；常规单 token 缓存解码（`record_past=False`）用 `causal_conv1d_update()` 更新卷积窗口。卷积之后拆开 QKV，恢复 head 维度：

[modeling_qwen3_5.py · L602–L622](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L602-L622)

```python
mixed_qkv = mixed_qkv.transpose(1, 2)
query, key, value = torch.split(
    mixed_qkv,
    [
        self.key_dim,
        self.key_dim,
        self.value_dim,
    ],
    dim=-1,
)

query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

beta = b.sigmoid()
# If the model is loaded in fp16, without the .float() here, A might be -inf
g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
if self.num_v_heads // self.num_k_heads > 1:
    query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
    key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
```

这里有两个与后面内核直接相关的细节。

第一，`g` 已经是**对数衰减**，真正乘在状态上的系数是 $`\alpha_t=\exp(g_t)`$。`A_log` 是可学习参数；先对它取指数，再取负号乘 softplus，使 $`g_t\leq 0`$。`beta` 则经过 sigmoid，控制本次误差写入的强度。

```math
\beta_t=\sigma(b_t),\qquad
 g_t=-\exp(A_{\log})\,\operatorname{softplus}(a_t+\mathrm{dt\_bias}),
\qquad \alpha_t=\exp(g_t).
```

第二，Q/K head 数可以少于 V head 数。以 [Qwen3.5-9B 的配置](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/config.json)为例，`D=4096`，Q/K 有 16 个 head，V 有 32 个 head，每个 head 的 K/V 维度都是 128。上面的 `repeat_interleave()` 把 Q/K 扩展到 32 个 head。

| 张量 | 模型侧形状（Q/K 扩展前） | FLA 输入形状或用途 |
| --- | --- | --- |
| `query`、`key` | `[B, T, 16, 128]` | `[B, T, 32, 128]` |
| `value` | `[B, T, 32, 128]` | `[B, T, 32, 128]` |
| `z` | `[B, T, 32, 128]` | 留在模型侧，用于输出门控 |
| `g`、`beta` | `[B, T, 32]` | `[B, T, 32]` |
| `recurrent_state` | — | `[B, 32, 128, 128]` |

FLA 自身也支持 Q/K head 共享，内核中的 `i_h // (HV // H)` 就是对应的 head 映射；不过这个 Transformers 入口已经显式重复了 Q/K。

接下来选择计算路径：**已有状态且本次只有一个 token**，走 recurrent；其余情况走 chunk。两个分支都要求在算子内做 Q/K 的 L2 归一化。

[modeling_qwen3_5.py · L624–L650](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L624-L650)

```python
recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0] if use_precomputed_states else None
if use_precomputed_states and seq_len == 1:
    core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=recurrent_state,
        output_final_state=cache_params is not None,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=kwargs.pop("cu_seq_lens_q", None),
        **kwargs,
    )
else:
    core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=recurrent_state,
        output_final_state=cache_params is not None,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=kwargs.pop("cu_seq_lens_q", None),
        **kwargs,
    )
```

函数名中的 `torch_` 不代表一定执行 PyTorch 参考实现。这些函数由 Transformers 的 [kernel 分派装饰器](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/integrations/hub_kernels.py#L829-L896)包装，可以使用本地 FLA、指定的 Hub kernel 或回退实现。下面展开本地 FLA 路径。

## 2. Recurrent：一个 token 如何修改状态

先只考虑一个 batch 中的一个 head。用列向量表示 $`q_t,k_t\in\mathbb R^{d_k}`$、$`v_t\in\mathbb R^{d_v}`$，状态布局跟随源码默认值：

```math
S_t\in\mathbb R^{d_k\times d_v}.
```

### 2.1 归一化与状态衰减

`fused_recurrent_gated_delta_rule_fwd_kernel()` 在进入循环前，将初始状态加载到 FP32 的 `b_h`。循环首先加载当前 token 的 QKV，然后对 Q/K 做 L2 归一化，并对 query 乘 $`d_k^{-1/2}`$：

[fused_recurrent.py · L112–L119](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gated_delta_rule/fused_recurrent.py#L112-L119)

```python
for _ in tl.range(0, T):
    b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale
```

下文的 $`q_t`$ 都表示已经归一化并缩放后的 query，$`k_t`$ 表示归一化后的 key。随后处理衰减：

[fused_recurrent.py · L129–L136](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gated_delta_rule/fused_recurrent.py#L129-L136)

```python
if USE_G:
    b_g = tl.load(p_g).to(tl.float32)
    if USE_GATE_IN_KERNEL:
        b_A = tl.load(A_log + i_hv).to(tl.float32)
        if HAS_DT_BIAS:
            b_g = b_g + tl.load(dt_bias + i_hv).to(tl.float32)
        b_g = -exp(b_A) * softplus(b_g)
    b_h *= exp(b_g)
```

Qwen 的入口已经算好了 `g`，所以这里 `USE_GATE_IN_KERNEL=False`，直接执行最后一行。这时 `b_h` 从 $`S_{t-1}`$ 变成 $`\bar S_t=\alpha_t S_{t-1}`$。

### 2.2 读出误差、写回、查询

默认 `[K, V]` 状态布局的核心更新如下：

[fused_recurrent.py · L157–L159](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gated_delta_rule/fused_recurrent.py#L157-L159)

```python
b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
b_h += b_k[:, None] * b_v
b_o = tl.sum(b_h * b_q[:, None], 0)
```

第一行沿 K 维求和，得到 $`\bar S_t^\top k_t`$，也就是衰减后的旧状态对当前 value 的预测。`b_v` 被覆盖成加权预测误差。第二行把这个误差与 key 做外积，写回状态；第三行用 query 查询**更新后的**状态。

```math
\begin{aligned}
\bar S_t &= \alpha_t S_{t-1},\\
e_t &= \beta_t\left(v_t-\bar S_t^\top k_t\right),\\
S_t &= \bar S_t+k_te_t^\top,\\
o_t &= S_t^\top q_t.
\end{aligned}
```

展开状态更新，可以看到 Delta Rule 对旧状态做了什么：

```math
S_t=\alpha_t\left(I-\beta_t k_tk_t^\top\right)S_{t-1}
       +\beta_t k_tv_t^\top.
```

$`\alpha_t`$ 对整个状态做衰减；$`I-\beta_t k_tk_t^\top`$ 修正 key 指定的方向。与直接累加 $`k_tv_t^\top`$ 相比，这次写入会扣掉状态已经记住的部分。

这个 kernel 沿序列时间循环，在 batch、head 和 V 的分片上并行。单 token decode 只执行一次循环，也不需要构造 token 间的注意力矩阵。长序列训练则不能只依赖这条逐 token 的状态链：下一节将把块内依赖改写成一个可求解的矩阵系统。

## 3. 从递推到 Chunk：下三角系统是怎样来的

取连续的 $`C`$ 个 token 作为一个块，把块入口状态记为 $`S_{\mathrm{in}}`$。本节所有下标都相对于当前块，先定义块内累计衰减：

```math
G_i=\sum_{r=1}^{i}g_r,\qquad \gamma_i=\exp(G_i),\qquad
D_{ij}=\begin{cases}
\exp(G_i-G_j), & j\leq i,\\
0, & j>i.
\end{cases}
```

$`\gamma_i`$ 是从块入口到第 $`i`$ 个 token 的衰减，$`D_{ij}`$ 是第 $`j`$ 次写入传播到第 $`i`$ 个位置时的衰减。将第二节的状态递推展开：

```math
S_i=\gamma_i S_{\mathrm{in}}+\sum_{j\leq i}D_{ij}k_je_j^\top.
```

计算第 $`i`$ 次写入误差时，只使用前面的写入，因此

```math
e_i^\top
=\beta_i v_i^\top
-\beta_i\gamma_i k_i^\top S_{\mathrm{in}}
-\sum_{j<i}\beta_i D_{ij}(k_i^\top k_j)e_j^\top.
```

把所有 $`e_i^\top`$ 堆成矩阵 $`E\in\mathbb R^{C\times d_v}`$，把 K、V 按 token 堆成行。再定义 $`B_\beta=\operatorname{diag}(\beta)`$、$`\Gamma=\operatorname{diag}(\gamma)`$，上面的逐行关系就变成：

```math
L=\operatorname{tril}\!\left(B_\beta\left(KK^\top\odot D\right),-1\right),
\qquad
(I+L)E=B_\beta V-B_\beta\Gamma K S_{\mathrm{in}}.
```

**这里必须是严格下三角。** 当前 token 的误差不能依赖自己刚写入的状态，所以 $`L`$ 不含对角线。$`I+L`$ 的对角线全为 1，可以直接做前代求解。令

```math
A=(I+L)^{-1},\qquad
U=A B_\beta V,\qquad
W=A B_\beta\Gamma K,
```

就得到 FLA 后续计算使用的形式：

```math
\boxed{E=U-W S_{\mathrm{in}}.}
```

`u` 和 `w` 只依赖当前块的输入，所有块可以并行计算；块入口状态到达后，再用一次矩阵乘法得到 `v_new`，即这里的 $`E`$。源码中称为 WY representation 的中间量就在这里。

| 数学量 | FLA 变量 | 单个块、单个 head 的形状 |
| --- | --- | --- |
| $`A=(I+L)^{-1}`$ | `A` | `[C, C]` |
| $`U=A B_\beta V`$ | `u` | `[C, d_v]` |
| $`W=A B_\beta\Gamma K`$ | `w` | `[C, d_k]` |
| $`S_{\mathrm{in}}`$ | `h` 中当前块的状态 | `[d_k, d_v]` |
| $`E=U-W S_{\mathrm{in}}`$ | `v_new` | `[C, d_v]` |

后面的源码按这个顺序执行：`chunk_local_cumsum` 准备 $`G`$，`chunk_gated_delta_rule_fwd_intra` 生成 `A/w/u`，`chunk_gated_delta_rule_fwd_h` 计算 `h/v_new`，最后 `chunk_fwd_o` 生成输出。

## 4. 块内计算：累计衰减、KKT 和三角求解

### 4.1 Cumsum 为什么乘 1 / ln 2

`chunk_gated_delta_rule_fwd()` 的普通 gate 分支先调用：

[chunk.py · L63–L69](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gated_delta_rule/chunk.py#L63-L69)

```python
g = chunk_local_cumsum(
    g,
    chunk_size=chunk_size,
    scale=RCP_LN2,
    cu_seqlens=cu_seqlens,
    chunk_indices=chunk_indices,
)
```

传给 cumsum 的 `scale=RCP_LN2`，因此保存在张量中的实际上是 $`\widetilde G_i=G_i/\ln 2`$。后续 Triton 内核使用 `exp2()`，两者组合满足

```math
2^{\widetilde G_i}=e^{G_i},\qquad
2^{\widetilde G_i-\widetilde G_j}=e^{G_i-G_j}.
```

对应 scalar kernel 的主要工作如下。每个 program 只扫描一个 chunk，累计和在每个块的起点重新开始：

[cumsum.py · L65–L73](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/utils/cumsum.py#L65-L73)

```python
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

前向的 `REVERSE=False`；反向会复用同一个 kernel 做后缀和。

### 4.2 先形成严格下三角的 L

默认 `C=64` 时，FLA 将 $`64\times64`$ 的矩阵拆成四个 $`16\times16`$ 对角块和六个下三角非对角块。`chunk_gated_delta_rule_fwd_kkt_solve_kernel()` 把 KKT 和三角求解融合到一个 kernel。下面是前两个子块的 KKT：

[chunk_fwd.py · L137–L150](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gated_delta_rule/chunk_fwd.py#L137-L150)

```python
for i_k in range(tl.cdiv(K, BK)):
    o_k = i_k * BK + tl.arange(0, BK)
    p_k0 = k + (i_tc0 + o_i)[:, None] * (H*K) + o_k[None, :]
    b_k0 = tl.load(p_k0, mask=m_tc0[:, None] & (o_k[None, :] < K), other=0.0)
    # diagonal block 0
    b_A00 += tl.dot(b_k0, tl.trans(b_k0))

    if i_tc1 < T:
        p_k1 = k + (i_tc1 + o_i)[:, None] * (H*K) + o_k[None, :]
        b_k1 = tl.load(p_k1, mask=m_tc1[:, None] & (o_k[None, :] < K), other=0.0)
        # diagonal block 1
        b_A11 += tl.dot(b_k1, tl.trans(b_k1))
        # off-diagonal (1,0)
        b_A10 += tl.dot(b_k1, tl.trans(b_k0))
```

`b_A00` 是 $`K_0K_0^\top`$，`b_A10` 是 $`K_1K_0^\top`$。这里只计算下三角区域需要的十个子块。随后乘上累计 gate 差值的指数，再按行乘 beta。以第一个对角块为例：

[chunk_fwd.py · L179–L183](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gated_delta_rule/chunk_fwd.py#L179-L183)

```python
m_d = o_i[:, None] > o_i[None, :]
m_I = o_i[:, None] == o_i[None, :]

if USE_G:
    b_A00 *= tl.where(m_d & m_tc0[:, None] & m_tc0[None, :], exp2(b_g0[:, None] - b_g0[None, :]), 0.)
```

[chunk_fwd.py · L200–L204](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gated_delta_rule/chunk_fwd.py#L200-L204)

```python
# diagonal blocks: scaled by beta
b_A00 = b_A00 * b_b0[:, None]
b_A11 = b_A11 * b_b1[:, None]
b_A22 = b_A22 * b_b2[:, None]
b_A33 = b_A33 * b_b3[:, None]
```

此时的 `b_A00` 对应 $`L`$ 的一个严格下三角子块，还不是逆矩阵 $`A`$。`m_d` 去掉对角线和上三角；`m_tc0` 同时屏蔽尾块越界的行、列。非对角子块已经完整位于下三角区域，因此只需要边界 mask。

### 4.3 在寄存器中做前代，再合并子块

先看第一个 $`16\times16`$ 对角块的求逆过程：

[chunk_fwd.py · L222–L231](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gated_delta_rule/chunk_fwd.py#L222-L231)

```python
b_Ai00 = -b_A00
b_Ai11 = -b_A11
b_Ai22 = -b_A22
b_Ai33 = -b_A33

for i in range(2, min(BC, T - i_tc0)):
    b_a00 = tl.sum(tl.where((o_i == i)[:, None], -b_A00, 0.), 0)
    b_a00 = tl.where(o_i < i, b_a00, 0.)
    b_a00 = b_a00 + tl.sum(b_a00[:, None] * b_Ai00, 0)
    b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
```

用 $`L_{00}`$ 表示这块严格下三角矩阵，目标是 $`A_{00}=(I+L_{00})^{-1}`$。由 $`(I+L_{00})A_{00}=I`$，第 $`i`$ 行满足

```math
(A_{00})_{i,:}=I_{i,:}-\sum_{j<i}(L_{00})_{ij}(A_{00})_{j,:}.
```

代码先把 `b_Ai00` 设为 $`-L_{00}`$，暂时不存对角线上的 1。循环中，`b_a00` 取出当前行的 $`-L_{00}`$；`tl.sum(b_a00[:, None] * b_Ai00, 0)` 累加已经求出的前面各行。最后再加回单位阵：

[chunk_fwd.py · L248–L251](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gated_delta_rule/chunk_fwd.py#L248-L251)

```python
b_Ai00 += m_I
b_Ai11 += m_I
b_Ai22 += m_I
b_Ai33 += m_I
```

循环从第 2 行开始，是因为从 0 编号时，第 0 行没有严格下三角元素，第 1 行只有一项 $`-L_{10}`$，初始化已经得到这两行的结果。

四个对角子块求完之后，还要补出逆矩阵的六个非对角子块。对于相邻的两个块，块矩阵前代给出

```math
A_{10}=-A_{11}L_{10}A_{00}.
```

源码直接用两次矩阵乘法实现：

[chunk_fwd.py · L257–L261](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gated_delta_rule/chunk_fwd.py#L257-L261)

```python
b_Ai10 = -tl.dot(
    tl.dot(b_Ai11, b_A10, input_precision=SOLVE_TRIL_DOT_PRECISION),
    b_Ai00,
    input_precision=SOLVE_TRIL_DOT_PRECISION
)
```

再往下的 $`A_{20}`$ 要累计经过第 0、1 个块的贡献：

```math
A_{20}=-A_{22}(L_{20}A_{00}+L_{21}A_{10}).
```

[chunk_fwd.py · L273–L278](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gated_delta_rule/chunk_fwd.py#L273-L278)

```python
b_Ai20 = -tl.dot(
    b_Ai22,
    tl.dot(b_A20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION) +
    tl.dot(b_A21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
    input_precision=SOLVE_TRIL_DOT_PRECISION,
)
```

其余子块按同一规则合并，最终写出的 `A` 才是完整的 $`(I+L)^{-1}`$。默认路径的融合边界也由此明确：KKT 的临时结果和子块求解留在同一个 kernel 内；生成 W/U 仍是后续 kernel。`chunk_size=16/32` 则走分开的 KKT 与 `solve_tril`，可见 [Python 调度代码](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gated_delta_rule/chunk_fwd.py#L368-L427)。

### 4.4 用 A 生成 U 和 W

`recompute_w_u_fwd_kernel()` 先加载刚求出的 `A`。U 的计算就是先给 V 乘 beta，再左乘 A：

[wy_fast.py · L78–L86](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gated_delta_rule/wy_fast.py#L78-L86)

```python
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

W 的计算多了一项从块入口到当前位置的累计衰减：

[wy_fast.py · L88–L102](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gated_delta_rule/wy_fast.py#L88-L102)

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

这对应第三节的 $`W=A B_\beta\Gamma K`$。`exp2(g)` 必须在乘 A **之前**作用于 K 的各行；它一般不能移到 A 左侧，因为 A 混合的是不同时间位置的数据。

到这里，每个 chunk 已经独立得到 `A/w/u`。下一个 kernel 才沿时间传播状态。

## 5. 块间状态：先减去旧记忆，再推进到块末

`chunk_gated_delta_rule_fwd_h()` 的每个 program 负责一个序列、一个 V head 和一片 V 通道，并在内部顺序遍历 chunk。循环开始时的 `b_h` 就是当前块的入口状态；它先被写入 `h`，供稍后的输出 kernel 使用。

K 维以 64 为一片，Qwen3.5-9B 的 `d_k=128` 因而用到 `b_h1`、`b_h2`。下面两段点积累加得到 $`W S_{\mathrm{in}}`$：

[chunk_delta_h.py · L200–L212](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/common/chunk_delta_h.py#L200-L212)

```python
p_w = w + o_t[:, None] * (HV*K) + o_k1[None, :]
b_w = tl.load(p_w, mask=m_t[:, None] & m_k1[None, :], other=0.0)
if STATE_V_FIRST:
    b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
else:
    b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
if K > 64:
    p_w = w + o_t[:, None] * (HV*K) + o_k2[None, :]
    b_w = tl.load(p_w, mask=m_t[:, None] & m_k2[None, :], other=0.0)
    if STATE_V_FIRST:
        b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
    else:
        b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
```

这里 `v` 参数实际传入的是上一节的 `u`。加载 U 后减掉刚才的结果，就得到本块每次写入的误差 E，并存入 `v_new`：

[chunk_delta_h.py · L227–L232](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/common/chunk_delta_h.py#L227-L232)

```python
p_v = v + o_t[:, None] * (HV*V) + o_v[None, :]
b_v = tl.load(p_v, mask=m_t[:, None] & m_v[None, :], other=0.0) - b_v

if SAVE_NEW_VALUE:
    p_v = v_new + o_t[:, None] * (HV*V) + o_v[None, :]
    tl.store(p_v, b_v.to(p_v.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])
```

### 5.1 状态如何从块入口走到块末

将第三节的状态展开式取在块末位置 $`C`$：

```math
S_{\mathrm{out}}
=\gamma_C S_{\mathrm{in}}
+K^\top\operatorname{diag}\!\left(\gamma_C/\gamma\right)E.
```

第一项是旧状态衰减到块末，第二项是各次新写入也衰减到块末后再求和。代码恰好按这两项执行：

[chunk_delta_h.py · L234–L247](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/common/chunk_delta_h.py#L234-L247)

```python
last_idx = min((i_t + 1) * BT, T) - 1
if USE_G:
    b_g_last = tl.load(g + (bos * HV + last_idx * HV + i_h).to(tl.int64)).to(tl.float32)
    p_g = g + (bos * HV + i_h).to(tl.int64) + o_t * HV
    b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
    b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
    b_g_last = exp2(b_g_last)
    b_h1 *= b_g_last
    if K > 64:
        b_h2 *= b_g_last
    if K > 128:
        b_h3 *= b_g_last
    if K > 192:
        b_h4 *= b_g_last
```

`b_v` 在保存到 `v_new` 时还是 E；这里才乘上 $`\gamma_C/\gamma_i`$，用于更新块末状态。`last_idx` 取本块最后一个**有效** token，尾块不足 64 个 token 时不会把填充位置当成终点。

接着加载 K，累加写入项。下面展示第一片 K 的更新，其他 K 分片执行相同运算：

[chunk_delta_h.py · L277–L284](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/common/chunk_delta_h.py#L277-L284)

```python
b_v = b_v.to(k.dtype.element_ty)

p_k = k + o_k1[:, None] + o_t[None, :] * (H*K)
b_k = tl.load(p_k, mask=m_k1[:, None] & m_t[None, :], other=0.0)
if STATE_V_FIRST:
    b_h1 += tl.trans(tl.dot(b_k, b_v))
else:
    b_h1 += tl.dot(b_k, b_v)
```

`b_h1` 等状态累加器保持 FP32，参与 `tl.dot` 时按输入类型转换；块入口快照 `h` 按 K 的 dtype 存储，返回的 `final_state` 使用 FP32。这里同时存在寄存器中的工作状态、供输出使用的块入口快照和最终缓存，读源码时需要区分它们。

这一阶段仍然有块间依赖，但序列上的迭代次数已经从 T 次变成约 $`T/C`$ 次；每次更新中的 $`WS`$ 和 $`K^\top E`$ 都可以使用矩阵乘法。

### 5.2 Packed 序列怎样保持独立

packed 输入把多条序列拼进 `[1, T_total, H, ...]`，但每条序列的 chunk 划分和初始状态仍然独立。`cu_seqlens` 给出 token 边界，`chunk_indices` 给出每个全局 chunk 对应的“序列编号、序列内 chunk 编号”。输出 kernel 的解码如下：

[chunk_o.py · L70–L80](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/common/chunk_o.py#L70-L80)

```python
if IS_VARLEN:
    i_tg = i_t
    i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    NT = tl.cdiv(T, BT)
else:
    NT = tl.cdiv(T, BT)
    i_tg = i_b * NT + i_t
    bos, eos = i_b * T, i_b * T + T

```

`bos` 用来定位 Q/K/V 的 token 区间；`i_tg` 则用于定位 `h` 中的块入口状态。二者的单位不同：前者数 token，后者数 chunk。每条序列分别分块，不能将整个 packed token 流直接切成连续的 64-token 块，否则一个块可能跨过样本边界。

## 6. 输出：历史状态与块内写入相加

每个位置的输出是 $`o_i=S_i^\top q_i`$。把第三节展开的状态代入，按 token 堆成矩阵：

```math
\boxed{O=\Gamma Q S_{\mathrm{in}}+(QK^\top\odot D)E.}
```

第一项查询进入本块之前的历史；第二项查询本块已经发生的写入。计算 E 时的 $`L`$ 是严格下三角，而这里 $`D`$ **包含对角线**，因为输出要读到当前 token 刚写入的内容。

`chunk_fwd_kernel_o()` 先分别累加 $`QS_{\mathrm{in}}`$ 和 $`QK^\top`$：

[chunk_o.py · L111–L117](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/common/chunk_o.py#L111-L117)

```python
# [BT, BK] @ [BK, BV] -> [BT, BV]
if STATE_V_FIRST:
    b_o += tl.dot(b_q, tl.trans(b_h))
else:
    b_o += tl.dot(b_q, b_h)
# [BT, BK] @ [BK, BT] -> [BT, BT]
b_A += tl.dot(b_q, b_k)
```

再为两项乘上各自的 gate：

[chunk_o.py · L119–L124](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/common/chunk_o.py#L119-L124)

```python
if USE_G:
    g += bos * HV + i_h
    p_g = g + o_t * HV
    b_g = tl.load(p_g, mask=m_t, other=0.0)
    b_o = b_o * exp2(b_g)[:, None]
    b_A = b_A * exp2(b_g[:, None] - b_g[None, :])
```

最后加因果 mask，加载 `v_new` 并生成输出：

[chunk_o.py · L130–L140](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/common/chunk_o.py#L130-L140)

```python
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

传入这个 kernel 的 `v` 就是 E。注意 `scale` 在写回前同时乘给两项；FLA 的 chunk 路径在这里应用 $`d_k^{-1/2}`$，等价于本文公式预先缩放 query，不能重复乘一次。

### 6.1 回到 Qwen：输出门 z 在哪里使用

GDN 内核返回 `[B, T, HV, d_v]` 的结果。Transformers 将它和 `z` 都展平成每行一个 V head，送入 `Qwen3_5RMSNormGated`：

[modeling_qwen3_5.py · L225–L234](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L225-L234)

```python
def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    # Norm before gate
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    hidden_states = self.weight * hidden_states.to(input_dtype)
    hidden_states = hidden_states * ACT2FN[self.activation](gate.to(torch.float32))

    return hidden_states.to(input_dtype)
```

因此这里的操作是

```math
\widetilde o_t=\operatorname{RMSNorm}(o_t)\odot\operatorname{SiLU}(z_t).
```

RMSNorm 沿每个 V head 的通道做归一化，之后乘 SiLU 输出门。它和前面的 $`\alpha`$、$`\beta`$ 作用在不同位置：前两者决定状态如何演化，`z` 决定当前输出通过多少。最后把所有 V head 拼接，经过 `out_proj` 返回模型隐藏维度；有 cache 时，同时把 `last_recurrent_state` 存回这一层，供下一次 decode 使用。

## 7. 反向：为什么再次出现 W、U 和状态扫描

chunk 训练路径由 `ChunkGatedDeltaRuleFunction` 接管 autograd。它在前向保存归一化后的 Q/K、原始 V、累计 gate、beta、A 和初始状态等，但没有保存 W、U、所有块入口状态与 `v_new`。因此反向一开始先重算 W/U：

[chunk.py · L147–L155](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gated_delta_rule/chunk.py#L147-L155)

```python
w, u = recompute_w_u_fwd(
    k=k,
    v=v,
    beta=beta,
    A=A,
    g=g,
    cu_seqlens=cu_seqlens,
    chunk_indices=chunk_indices,
)
```

然后重新调用状态前向，得到 `h/v_new`，并先算输出对块内写入的局部梯度：

[chunk.py · L160–L181](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gated_delta_rule/chunk.py#L160-L181)

```python
h, v_new, _ = chunk_gated_delta_rule_fwd_h(
    k=k,
    w=w,
    u=u,
    g=g,
    initial_state=initial_state,
    output_final_state=False,
    cu_seqlens=cu_seqlens,
    chunk_indices=chunk_indices,
    state_v_first=state_v_first,
    chunk_size=chunk_size,
)
dv = chunk_bwd_dv_local(
    q=q,
    k=k,
    g=g,
    do=do,
    scale=scale,
    cu_seqlens=cu_seqlens,
    chunk_indices=chunk_indices,
    chunk_size=chunk_size,
)
```

这对应一个直接的依赖关系：由

```math
O=\Gamma Q S_{\mathrm{in}}+M E,\qquad M=QK^\top\odot D,
```

得到当前块输出贡献的 $`\nabla_E=M^\top\nabla_O`$。但 E 也参与块末状态更新，会影响后面所有 chunk，所以反向还要调用 [`chunk_gated_delta_rule_bwd_dhu()`](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gated_delta_rule/chunk.py#L202-L216)，从后往前传播状态梯度，再由 `chunk_bwd_dqkwg()` 计算输出和状态路径的 Q/K/W/g 梯度。

最后通过 W/U 的构造继续反传到原始 K/V/beta/g，并合并 K、g 的两路贡献：

[chunk.py · L233–L246](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gated_delta_rule/chunk.py#L233-L246)

```python
dk2, dv, db, dg2 = prepare_wy_repr_bwd(
    k=k,
    v=v,
    beta=beta,
    g=g,
    A=A,
    dw=dw,
    du=dv,
    cu_seqlens=cu_seqlens,
    chunk_indices=chunk_indices,
)
dk.add_(dk2)
dg.add_(dg2)
dg = chunk_local_cumsum(dg, chunk_size=chunk_size, reverse=True, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
```

最后一行是反向 cumsum。前向有 $`G_i=\sum_{j\leq i}g_j`$，因此每个原始 gate 的梯度需要累加所有后续累计 gate 的贡献：

```math
\frac{\partial\mathcal L}{\partial g_j}
=\sum_{i\geq j}\frac{\partial\mathcal L}{\partial G_i}.
```

`exp2` 与 `1 / ln 2` 的换底因子在整条链式求导中抵消；这里需要的就是块内后缀和。Q/K 的 L2 归一化梯度随后由 autograd 包装层计算，再返回模型入口，继续传到卷积、线性投影和 gate 参数。训练因而保留完整的状态依赖，同时通过重计算减少前向中间张量的保存。

---

本文依据[原文的 GDN 推导与 kernel 解析](https://zhuanlan.zhihu.com/p/2007937984738129405)重新组织，并对照当前源码展开。源码节选来自 [Transformers（Apache-2.0；© 2025 The Qwen Team and The HuggingFace Inc. team）](https://github.com/huggingface/transformers/blob/v5.17.0/LICENSE)及 [FLA（MIT；© 2023–2026 Songlin Yang, Yu Zhang, Zhiyuan Li）](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/LICENSE)，各段链接固定到正式版。另核对了 Transformers [`27166ea`](https://github.com/huggingface/transformers/tree/27166ea03f12c940f23176a904ab1d2ff1a3dcbb) 与 FLA [`954438d`](https://github.com/fla-org/flash-linear-attention/tree/954438d1fcb5e1bb05c22f9908de9c5c2df74ae5) 主分支；本文所推导的状态递推与 chunk 分解不变。

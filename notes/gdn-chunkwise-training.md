# GDN Chunkwise Training

# Input Projections and Gating

GDN 的 chunkwise 计算需要两类输入：描述当前内容的 QKV，以及控制状态更新的衰减门与写入门。Qwen3.5 的实现先在外层完成投影和短卷积，再进入 chunk 内核完成归一化、分块与累计衰减计算。

<figure class="lake-figure" style="width: 640px;">
  <img src="../../assets/gdn-architecture.png" alt="Gated DeltaNet 模型与 token mixer 结构，Yang et al., Figure 1，CC BY 4.0" width="2058" height="1122">
</figure>

设进入 GDN 层的单个 token 表示为 $`\boldsymbol{x}_t`$。Qwen3.8-Flash-Next 技术报告 §2.1.1 将 QKV 的特征生成写为投影、短因果卷积与激活的复合映射；Q/K 还需沿 head 的特征维归一化。用 $`\widehat{\boldsymbol{q}}_t`$ 表示尚未乘 head dimension scaling 的归一化 query：

```math
\begin{aligned}
\widehat{\boldsymbol{q}}_t
&=\operatorname{L2Norm}\left(\operatorname{SiLU}\left(\operatorname{ShortConv}(\mathbf{W}_q\boldsymbol{x}_t)\right)\right),\\
\boldsymbol{k}_t
&=\operatorname{L2Norm}\left(\operatorname{SiLU}\left(\operatorname{ShortConv}(\mathbf{W}_k\boldsymbol{x}_t)\right)\right),\\
\boldsymbol{v}_t
&=\operatorname{SiLU}\left(\operatorname{ShortConv}(\mathbf{W}_v\boldsymbol{x}_t)\right).
\end{aligned}\tag{1}
```

源码将三个投影合并为 `in_proj_qkv`，并单独投影出输出门 `z`、写入参数 `b` 和衰减参数 `a`。短卷积缓存保存局部历史输入，递推缓存保存 GDN 的记忆矩阵；`z` 留待状态读出之后使用，不参与记忆更新。

```python
hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
batch_size, seq_len, _ = hidden_states.shape
if cache_params is not None:
    conv_state = cache_params.get_conv_state(self.layer_idx, state_idx=0)
    recurrent_state = cache_params.get_recurrent_state(self.layer_idx, state_idx=0)
else:
    conv_state, recurrent_state = None, None

mixed_qkv = self.in_proj_qkv(hidden_states)
mixed_qkv = mixed_qkv.transpose(1, 2)
z = self.in_proj_z(hidden_states)
z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
b = self.in_proj_b(hidden_states)
a = self.in_proj_a(hidden_states)
```

短因果卷积为 QKV 提供局部时序归纳偏置，使当前位置的内容特征在本层显式融合邻近位置的信息。卷积缓存的作用是维持这一局部感受野的连续性：当输入被拆成多次调用时，新输入仍能访问调用边界左侧的投影特征，使分段处理与连续处理具有相同的卷积上下文。它保留有限窗口内的具体特征与相对顺序，保存长度由卷积核宽度决定。

递推缓存承担更长时间尺度上的关联记忆。每个 head 用固定大小的矩阵汇总历史键值关系，并通过衰减与 delta 更新调整保留内容，使后续 query 能从历史关联中读出信息。这里存在明确的容量取舍：存储量不随历史长度增长，有限状态也无法逐项保存任意长历史中的全部关联。卷积缓存维持局部特征提取所需的时序窗口，递推缓存延续内容相关的历史记忆；续算时只保留其中一种，另一条计算路径就会在调用边界失去历史条件。chunk 划分仅改变递推的计算组织，块边界本身不构成记忆重置点。

```python
if cache_params is not None:
    mixed_qkv = cache_params.update_conv_state(
        mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size)
mixed_qkv = causal_conv1d_fn(
    mixed_qkv, self.conv1d.weight.squeeze(1), self.conv1d.bias,
    activation=self.activation, **kwargs)
if cache_params is not None:
    mixed_qkv = mixed_qkv[:, :, -seq_len:]

mixed_qkv = mixed_qkv.transpose(1, 2)
query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
```

两个状态门按 token、按 value head 生成标量。记 $`A_h`$ 和 $`d_h`$ 分别对应 `A_log` 与 `dt_bias` 在 head $`h`$ 上的参数，则

```math
\beta_t=\sigma(b_t),\qquad
g_t=-\exp(A_h)\operatorname{softplus}(a_t+d_h),\qquad
\alpha_t=\exp(g_t).
\tag{2}
```

`g` 直接存储 $`\log\alpha`$。负指数参数化使实际衰减因子处于 $`(0,1)`$，`beta` 则控制当前键值关联的更新强度。当 value head 数更多时，这一实现重复 Q/K 的 head，使每个 value head 有对应的 query 和 key。

```python
beta = b.sigmoid()
g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
if self.num_v_heads // self.num_k_heads > 1:
    query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
    key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
```

对单个 head，衰减因子 $`\alpha_t`$ 控制历史状态的整体保留比例，更新强度 $`\beta_t`$ 控制沿当前 key 方向的旧关联消除与新关联写入；key 的范数因此直接影响状态更新的尺度。其状态更新为

```math
\mathbf{S}_t=\mathbf{S}_{t-1}\left(\alpha_t\left(\mathbf{I}-\beta_t\boldsymbol{k}_t\boldsymbol{k}_t^{\intercal}\right)\right)+\beta_t\boldsymbol{v}_t\boldsymbol{k}_t^{\intercal}\in\mathbb{R}^{d_v\times d_k}.\tag{3}
```

# Normalization and Chunkwise Decay

`torch_chunk_gated_delta_rule` 按 head 独立组织序列计算。这条 PyTorch 实现将核心中间量转换为 FP32，以减小归一化、累计衰减与状态更新中的舍入误差。Q/K 的归一化分别作用于每个 token、每个 head 内的特征向量，归约范围仅限特征维，不引入不同 token 或 head 之间的信息混合。

```python
initial_dtype = query.dtype
batch_size, sequence_length, _, k_head_dim = key.shape
num_v_heads, v_head_dim = value.shape[-2:]
recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
padded_output_shape = (batch_size, num_v_heads, -1, v_head_dim)
decay = g
query, key, value, beta, decay = [
    x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
    for x in (query, key, value, beta, decay)
]
```

L2 归一化使 Q/K 的关联匹配主要取决于特征方向，从而降低上游投影幅值波动的影响。对 key，范数约束使沿当前方向的擦除比例主要受写入门控制；对 query，它约束查询向量对记忆读出的放大作用。数值稳定项 $`\varepsilon`$ 避免分母在原点退化，使近零输入处的映射保持连续，零向量仍映射为零。归一化后施加的 $`d_k^{-1/2}`$ 用于调整 query 的读出尺度，不参与状态更新：

```math
\operatorname{L2Norm}_{\varepsilon}(\boldsymbol{x})
=\frac{\boldsymbol{x}}{\sqrt{\sum_{r=1}^{d_k}x_r^2+\varepsilon}},
\qquad
\boldsymbol{q}_t=\frac{\widehat{\boldsymbol{q}}_t}{\sqrt{d_k}},
\qquad \varepsilon=10^{-6}.
\tag{4}
```

结合式（3），归一化保证 $`\|\boldsymbol{k}_t\|_2\leq1`$，使秩一修正算子 $`\mathbf{I}-\beta_t\boldsymbol{k}_t\boldsymbol{k}_t^{\intercal}`$ 沿 key 方向的特征值 $`1-\beta_t\|\boldsymbol{k}_t\|_2^2`$ 位于 $`[0,1]`$，在其正交子空间上的特征值为 1。因此，对固定的 key 与门控，该算子不会放大已有状态的扰动；进一步乘以 $`\alpha_t`$ 则施加整体衰减。这一性质约束历史状态的传播增益，不限制每一步新写入的 value 项的幅度。

```python
def l2norm(x, dim=-1, eps=1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm

if use_qk_l2norm_in_kernel:
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)
scaling = query.shape[-1] ** -0.5
query = query * scaling
```

随后按默认块长 $`C=64`$ 划分序列；长度 $`T=128`$ 对应两个完整 chunk，不能整除时仅在序列尾部补齐。预先计算 $`\beta K`$ 与 $`\beta V`$，使逐 token 的写入权重显式进入后续块内线性系统：`v_beta` 作为第一组三角求解的右端项，`k_beta` 参与构造系统系数，并在乘累计衰减后构成另一右端项。补齐位置取 $`\beta=0`$、$`g=0`$，分别对应零写入强度与单位衰减因子，因此不会改变有效序列结束处的递推状态。

```python
pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
beta, decay = (F.pad(x, (0, pad_size)) for x in (beta, decay))
total_sequence_length = sequence_length + pad_size
num_chunks = total_sequence_length // chunk_size
v_beta = value * beta.unsqueeze(-1)
k_beta = key * beta.unsqueeze(-1)
# query/key/k_beta: [B, H, num_chunks, chunk_size, d_k]
# v_beta: [B, H, num_chunks, chunk_size, d_v]
query, key, k_beta, v_beta = [
    x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
    for x in (query, key, k_beta, v_beta)
]
# decay: [B, H, num_chunks, chunk_size]
decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)
```

对固定的 batch、head 和 chunk，使用块内下标 $`i=1,\ldots,C`$。累计衰减 $`\gamma_i`$ 与代码中的累计对数衰减 $`G_i`$ 满足

```math
\gamma_i=\prod_{r=1}^{i}\alpha_r,
\qquad
G_i=\sum_{r=1}^{i}g_r=\log\gamma_i,
\qquad \gamma_0=1,\quad G_0=0.
\tag{5}
```

衰减因果矩阵在下三角使用前缀衰减的比值，上三角为零。行 $`i`$ 表示当前读出位置，列 $`j`$ 表示历史写入位置；限制到一个 chunk 后，矩阵为

```math
\boldsymbol{\Gamma}=
\begin{pmatrix}
1 & 0 & \cdots & 0\\
\gamma_2/\gamma_1 & 1 & \cdots & 0\\
\vdots & \vdots & \ddots & \vdots\\
\gamma_C/\gamma_1 & \gamma_C/\gamma_2 & \cdots & 1
\end{pmatrix},
\qquad
\Gamma_{ij}=
\begin{cases}
\displaystyle\frac{\gamma_i}{\gamma_j}
=\prod_{r=j+1}^{i}\alpha_r
=\exp(G_i-G_j), & i\geq j,\\
0, & i<j.
\end{cases}
\tag{6}
```

**对数空间把累计连乘变成求和，把前缀积相除变成累计和相减。** $`\gamma_i/\gamma_j`$ 消去写入位置及其之前的公共前缀，只保留从 $`j+1`$ 到 $`i`$ 的衰减；当 $`i=j`$ 时，空乘积为 1。实现直接计算 $`G_i-G_j`$，避免分别生成可能已经下溢的两个前缀积，再进行不稳定的相除。

每个 chunk 独立建立累计衰减的参照点，跨块历史仍通过入口 `recurrent_state` 传递。由于 $`g\leq0`$，下三角差值非正；先把上三角置为 $`-\infty`$ 再取指数，可同时实现因果屏蔽并避免未来位置的正指数溢出。所得 `pairwise_decay` 对应式（6）的 $`\boldsymbol{\Gamma}`$，对角线为 1，上三角为 0。

```python
strictly_upper_mask = torch.ones(
    chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
cum_decay = decay.cumsum(dim=3)  # 在每个 chunk 内累计
# pairwise_decay: [B, H, num_chunks, chunk_size, chunk_size]
pairwise_decay = cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)
pairwise_decay = pairwise_decay.masked_fill(strictly_upper_mask, float("-inf"))
pairwise_decay = pairwise_decay.exp()
```

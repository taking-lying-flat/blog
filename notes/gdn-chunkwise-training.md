# GDN Chunkwise Training

# 🦊 Input Projections and Gating

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

# 🦉 Normalization and Chunkwise Decay

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

每个 chunk 独立建立累计衰减的参照点，跨块历史仍通过入口 `recurrent_state` 传递。由于 $`g\leq0`$，下三角差值非正；先把上三角置为 $`-\infty`$ 再取指数，可同时实现因果屏蔽并避免未来位置的正指数溢出。所得 `pairwise_decay` 对应式（6）的 $`\boldsymbol{\Gamma}`$，对角线为 1，上三角为 0。

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

```python
strictly_upper_mask = torch.ones(
    chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
cum_decay = decay.cumsum(dim=3)  # 在每个 chunk 内累计
# pairwise_decay: [B, H, num_chunks, chunk_size, chunk_size]
pairwise_decay = cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)
pairwise_decay = pairwise_decay.masked_fill(strictly_upper_mask, float("-inf"))
pairwise_decay = pairwise_decay.exp()
```

# 🐨 Mamba2: Linear Attention with Decay

**在不考虑归一化操作以及查询和键的激活函数时，线性 Transformer 可以表述为如下线性递推形式**

```math
S_t=S_{t-1}+v_tk_t^\top\in\mathbb{R}^{d_v\times d_k}, \quad o_t=S_tq_t\in\mathbb{R}^{d_v}
\tag{7}
```

- 其中 $`d_k`$ 和 $`d_v`$ 分别表示查询/键和值的注意力头维度。通过展开上述递推关系，分别得到向量形式和矩阵形式

```math
o_t=\sum_{i=1}^{t}(v_ik_i^\top)q_t =\sum_{i=1}^{t}v_i(k_i^\top q_t)\in\mathbb{R}^{d_v}, \quad O=(QK^\top\odot M)V\in\mathbb{R}^{L\times d_v}
\tag{8}
```

- 其中 $`L`$ 表示序列长度， $`M\in\mathbb{R}^{L\times L}`$ 为因果掩码：当 $`i<j`$ 时 $`M_{ij}=0`$；否则 $`M_{ij}=1`$

**基础的线性注意力在语言建模任务上的性能远逊于标准 Transformer。为解决这一问题，一种常见做法是引入衰减项，以遗忘历史信息。以 Mamba2 为例，其可以表示为如下线性递推形式**

```math
S_t=\alpha_tS_{t-1}+v_tk_t^\top, \quad o_t=S_tq_t
\tag{9}
```

- 其中 $`\alpha_t\in(0,1)`$ 是一个随时间步 $`t`$ 变化、依赖于输入数据的标量衰减项。定义累积衰减乘积 $`\gamma_j=\prod_{i=1}^{j}\alpha_i`$。通过展开上述递推关系，可以分别得到如下向量形式和矩阵并行形式

```math
o_t=\sum_{i=1}^{t}\frac{\gamma_t}{\gamma_i}v_ik_i^\top q_t =\sum_{i=1}^{t}v_i\left(\frac{\gamma_t}{\gamma_i}k_i^\top q_t\right), \quad O=\bigl((QK^\top)\odot\Gamma\bigr)V.
\tag{10}
```

```math
\Gamma_{ij}= \begin{cases} \frac{\gamma_i}{\gamma_j}, & i\ge j\\ 0, & i<j \end{cases}
\tag{11}
```

- $`\Gamma\in\mathbb{R}^{L\times L}`$ 是一个考虑衰减效应的因果掩码。上述并行形式与递推形式之间的等价关系也被称为状态空间对偶性

- 类似的递推结构还出现在 Gated RFA、xLSTM 和 Gated RetNet 等多种架构中。当 $`\gamma_t`$ 与输入数据无关时，该形式可退化为 RetNet 和 Lightning Attention

**为实现硬件友好的线性时间训练，通常采用分块并行形式。分块并行方法将输入和输出划分为若干大小为** $`C`$ **的分块，并依据前一个分块的最终状态以及当前分块的查询、键和值矩阵计算当前分块的输出**

- 定义 $`Q_{[t]}=[q_{tC+1},\ldots,q_{(t+1)C}]^\top`$ 为第 $`t`$ 个分块的查询矩阵，并定义 $`q_{[t]}^r:=q_{tC+r}`$ 为第 $`t`$ 个分块内的第 $`r`$ 个查询向量。第 $`t`$ 个分块的初始状态定义为 $`S_{[t]}:=S_{[t]}^0=S_{[t-1]}^C`$。先对无衰减的线性注意力递推进行部分展开，可以得到

```math
S_{[t]}^r =S_{[t]}+\sum_{i=1}^{r}v_{[t]}^i{k_{[t]}^i}^{\top} \in\mathbb{R}^{d_v\times d_k}, \quad o_{[t]}^r =S_{[t]}^rq_{[t]}^r =S_{[t]}q_{[t]}^r +\sum_{i=1}^{r}v_{[t]}^i \left({k_{[t]}^i}^{\top}q_{[t]}^r\right) \in\mathbb{R}^{d_v}
\tag{12}
```

```math
S_{[t+1]} =S_{[t]}+V_{[t]}^\top K_{[t]} \in\mathbb{R}^{d_v\times d_k}, \quad O_{[t]} =Q_{[t]}S_{[t]}^\top +\bigl(Q_{[t]}K_{[t]}^\top\odot M\bigr)V_{[t]} \in\mathbb{R}^{C\times d_v}
\tag{13}
```

- 其中 $`M\in\mathbb{R}^{C\times C}`$ 是分块内部的因果掩码。上述方程主要由矩阵乘法运算构成，因此可以充分利用张量核心进行硬件层面的优化。**这种分块算法也可以方便地扩展至包含衰减项的线性注意力**

```math
S_{[t+1]} =\overrightarrow{S}_{[t]} +V_{[t]}^\top\overrightarrow{K}_{[t]} \in\mathbb{R}^{d_v\times d_k}, \quad O_{[t]} =\overleftarrow{Q}_{[t]}S_{[t]}^\top +\bigl(Q_{[t]}K_{[t]}^\top\odot\Gamma_{[t]}\bigr)V_{[t]} \in\mathbb{R}^{C\times d_v}.
\tag{14}
```

```math
(\Gamma_{[t]})_{ij} = \begin{cases}\gamma_{[t]}^i/\gamma_{[t]}^j,&i\ge j,\\0,&i<j,\end{cases}\qquad \gamma_{[t]}^j = \prod_{\ell=tC+1}^{tC+j}\alpha_\ell.
\tag{15}
```

- 使用左箭头 $`\overleftarrow{(\,\cdot\,)}`$ 和右箭头 $`\overrightarrow{(\,\cdot\,)}`$，分别表示将变量衰减至每个分块的首位置和末位置

```math
\begin{aligned}
{\color{blue}\overleftarrow{\boldsymbol{q}_{[t]}^r}} &= {\color{blue}\gamma_{[t]}^r} \boldsymbol{q}_{[t]}^r && \text{decaying each vector to the first position of chunk } t \\
{\color{blue}\overrightarrow{\boldsymbol{k}_{[t]}^r}} &= \frac{{\color{blue}\gamma_{[t]}^C}}{{\color{blue}\gamma_{[t]}^r}} \boldsymbol{k}_{[t]}^r && \text{decaying each vector to the last position of chunk } t \\
{\color{blue}\overrightarrow{\mathbf{S}_{[t]}}} &= {\color{blue}\gamma_{[t]}^C} \mathbf{S}_{[t]} && \text{decaying the state matrix over the entire chunk } t
\end{aligned}
\tag{16}
```

**首先将线性注意力及其带衰减形式表示为等价的递推与并行矩阵计算，并进一步通过 chunkwise 将逐 token 串行更新改写为块内少量大规模 GEMM、块间仅传递紧凑状态矩阵** $`S_{[t]}`$ **，从而在保持线性复杂度和计算等价性的同时充分利用 GPU Tensor Core**

# 🦌 Delta Networks: Linear Attention with Delta Rule

**Delta 更新规则会动态擦除与当前输入键** $`k_t`$ **相关联的旧值** $`v_t^{\mathrm{old}}`$ **，写入新值** $`v_t^{\mathrm{new}}`$ **。新值由当前输入值与旧值根据写入强度** $`\beta_t\in(0,1)`$ **进行线性组合得到**

```math
S_t=S_{t-1}+\beta_t(v_t-S_{t-1}k_t)k_t^{\top}=S_{t-1}(I-\beta_t k_t k_t^{\top})+\beta_t v_t k_t^{\top}.\tag{17}
```

- **DeltaNet 实现一种以广义 Householder 矩阵** $`I-\beta_tk_tk_t^\top`$ **为状态转移矩阵的一阶线性递推。尽管 DeltaNet 在关联回忆和语言建模任务中表现出色，但由于计算效率较低，其长期以来并未受到广泛关注**

**Chunkwise parallel form：分块并行与 WY 表示法**

```math
\mathbf{S}_{[t]}^r = \mathbf{S}_{[t]} \underbrace{\left( \prod_{i=1}^r \left(\mathbf{I} - \beta_{[t]}^i \boldsymbol{k}_{[t]}^i {\boldsymbol{k}_{[t]}^i}^\top\right) \right)}_{:=\mathbf{P}_{[t]}^r} + \underbrace{\sum_{i=1}^r \left( \beta_{[t]}^i \boldsymbol{v}_{[t]}^i {\boldsymbol{k}_{[t]}^i}^\top \prod_{j=i+1}^r \left( \mathbf{I} - \beta_{[t]}^j \boldsymbol{k}_{[t]}^j {\boldsymbol{k}_{[t]}^j}^\top \right) \right)}_{:=\mathbf{H}_{[t]}^r}
\tag{18}
```

- **WY 表示法：将有序矩阵连乘改写为外积和。** $`P_{[t]}^r`$ 中逐 token 累乘的状态转移矩阵可写成 $`I-\sum_i w_{[t]}^i{k_{[t]}^i}^{\top}`$，块内累计写入则写成 $`\sum_i u_{[t]}^i{k_{[t]}^i}^{\top}`$。这使历史状态变换与新增写入分别由 $`W`$、$`U`$ 表示，为矩阵化计算提供代数形式；此时 $`w^r`$、$`u^r`$ 的定义仍包含对先前位置的递推依赖

```math
\mathbf{P}_{[t]}^r = \mathbf{I} - \sum_{i=1}^r \mathbf{w}_{[t]}^i {\boldsymbol{k}_{[t]}^i}^\top \in \mathbb{R}^{d_k \times d_k} \qquad \mathbf{w}_{[t]}^r = \beta_{[t]}^r \left( \boldsymbol{k}_{[t]}^r - \sum_{i=1}^{r-1} \left( \mathbf{w}_{[t]}^i ({\boldsymbol{k}_{[t]}^i}^\top \boldsymbol{k}_{[t]}^r) \right) \right) \in \mathbb{R}^{d_k}
\tag{19}
```

```math
\mathbf{H}_{[t]}^r = \sum_{i=1}^r \mathbf{u}_{[t]}^i {\boldsymbol{k}_{[t]}^i}^\top \in \mathbb{R}^{d_v \times d_k} \qquad \mathbf{u}_{[t]}^r = \beta_{[t]}^r \left( \boldsymbol{v}_{[t]}^r - \sum_{i=1}^{r-1} \left( \mathbf{u}_{[t]}^i ({\boldsymbol{k}_{[t]}^i}^\top \boldsymbol{k}_{[t]}^r) \right) \right) \in \mathbb{R}^{d_v}
\tag{20}
```

- **UT 变换：将 W、U 的递推依赖组织为共享系数的下三角线性系统。** 系统系数由块内 key 的 Gram 矩阵与写入门构成，两个右端项分别为 $`\operatorname{diag}(\beta)K`$ 与 $`\operatorname{diag}(\beta)V`$。式（22）以 $`T`$ 表示同一求解算子，使 W、U 可通过块级三角求解和矩阵乘法共同计算；逆矩阵是代数记号，实现无需显式构造一般矩阵的逆。**WY 给出紧凑表示，UT 给出计算该表示的块级方法**

```math
\mathbf{P}_{[t]} = \mathbf{I} - \mathbf{W}_{[t]}^\top \mathbf{K}_{[t]} \in \mathbb{R}^{d_k \times d_k}, \qquad \mathbf{H}_{[t]} = \mathbf{U}_{[t]}^\top \mathbf{K}_{[t]} \in \mathbb{R}^{d_v \times d_k}
\tag{21}
```

```math
\mathbf{T}_{[t]} = \left[ \mathbf{I} + \text{strictLower} \left( \text{diag}(\beta_{[t]}) \mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top \right) \right]^{-1} \text{diag} (\beta_{[t]}) \in \mathbb{R}^{C \times C}
\tag{22}
```

```math
\mathbf{W}_{[t]} = \mathbf{T}_{[t]} \mathbf{K}_{[t]} \in \mathbb{R}^{C \times d_k}, \qquad \mathbf{U}_{[t]} = \mathbf{T}_{[t]} \mathbf{V}_{[t]} \in \mathbb{R}^{C \times d_v}
\tag{23}
```

- 将 WY 表示与 UT 变换得到的 $`W`$、$`U`$ 代回后，$`U-WS_{[t]}^{\top}`$ 表示当前块相对于入口状态的有效写入。块内输出和块末状态更新由矩阵乘法完成，从而利用 Tensor Core；块间仍通过 $`S_{[t]}`$ 传递历史状态

```math
\mathbf{S}_{[t+1]} = \mathbf{S}_{[t]}\mathbf{P}_{[t]} + \mathbf{H}_{[t]} = \mathbf{S}_{[t]} + \left(\mathbf{U}_{[t]} - \mathbf{W}_{[t]}\mathbf{S}_{[t]}^\top \right)^\top \mathbf{K}_{[t]} \in \mathbb{R}^{d_v \times d_k}
\tag{24}
```

```math
\mathbf{O}_{[t]} = \mathbf{Q}_{[t]}\mathbf{S}_{[t]}^\top + \left(\mathbf{Q}_{[t]}\mathbf{K}_{[t]}^\top \odot \mathbf{M} \right) \left(\mathbf{U}_{[t]} - \mathbf{W}_{[t]}\mathbf{S}_{[t]}^\top \right) \in \mathbb{R}^{C \times d_v}
\tag{25}
```

# 🐧 WY Representation: Appendix B.1

*Parallelizing Linear Transformers with the Delta Rule over Sequence Length* 的完整会议版在附录 **B Method Continued** 中分别证明 WY 表示与 UT 变换。B.1 处理的障碍是：DeltaNet 的写入依赖此前状态，直接沿序列并行展开会产生逐位置的二维中间矩阵。WY 利用每次更新的秩一结构，将这些累计变换表示为向量外积之和，使分块计算只需传递块边界状态，并保存各位置的向量系数。

**先证明入口状态的累计变换。** 当 $`r=1`$ 时，取 $`\mathbf{w}_{[t]}^1=\beta_{[t]}^1\boldsymbol{k}_{[t]}^1`$，式（19）成立。假设它对 $`r-1`$ 成立，按附录 B.1 依次代入归纳假设、展开乘积，再合并以当前 key 为右因子的项：

```math
\begin{aligned}
\mathbf{P}_{[t]}^r
&=\mathbf{P}_{[t]}^{r-1}\left(\mathbf I-\beta_{[t]}^r\boldsymbol{k}_{[t]}^r{\boldsymbol{k}_{[t]}^r}^{\top}\right)\\
&=\left(\mathbf I-\sum_{i=1}^{r-1}\mathbf w_{[t]}^i{\boldsymbol k_{[t]}^i}^{\top}\right)
  \left(\mathbf I-\beta_{[t]}^r\boldsymbol k_{[t]}^r{\boldsymbol k_{[t]}^r}^{\top}\right)\\
&=\mathbf I-\sum_{i=1}^{r-1}\mathbf w_{[t]}^i{\boldsymbol k_{[t]}^i}^{\top}
  -\beta_{[t]}^r\boldsymbol k_{[t]}^r{\boldsymbol k_{[t]}^r}^{\top}\\
&\quad+\beta_{[t]}^r\sum_{i=1}^{r-1}\mathbf w_{[t]}^i
  \left({\boldsymbol k_{[t]}^i}^{\top}\boldsymbol k_{[t]}^r\right){\boldsymbol k_{[t]}^r}^{\top}\\
&=\mathbf I-\sum_{i=1}^{r-1}\mathbf w_{[t]}^i{\boldsymbol k_{[t]}^i}^{\top}
  -\underbrace{\beta_{[t]}^r\left(\boldsymbol k_{[t]}^r-\sum_{i=1}^{r-1}\mathbf w_{[t]}^i
  \left({\boldsymbol k_{[t]}^i}^{\top}\boldsymbol k_{[t]}^r\right)\right)}_{\mathbf w_{[t]}^r}{\boldsymbol k_{[t]}^r}^{\top}\\
&=\mathbf I-\sum_{i=1}^{r}\mathbf w_{[t]}^i{\boldsymbol k_{[t]}^i}^{\top}.
\end{aligned}
\tag{26}
```

- 归纳过程中没有交换矩阵乘法的顺序。此前所有更新对当前 key 的影响进入 $`\mathbf w_{[t]}^r`$，因此外积和仍保留原有序乘积的结果。这正是式（19）中 w 递推的来源。

**块内写入项使用相同的归纳方法。** B.1 对零初态状态的证明，对应前文式（18）的 $`\mathbf H_{[t]}^r`$。从 $`\mathbf u_{[t]}^1=\beta_{[t]}^1\boldsymbol v_{[t]}^1`$ 出发，假设 $`\mathbf H_{[t]}^{r-1}=\sum_{i<r}\mathbf u_{[t]}^i{\boldsymbol k_{[t]}^i}^{\top}`$，则

```math
\begin{aligned}
\mathbf H_{[t]}^r
&=\mathbf H_{[t]}^{r-1}\left(\mathbf I-\beta_{[t]}^r\boldsymbol k_{[t]}^r{\boldsymbol k_{[t]}^r}^{\top}\right)
  +\beta_{[t]}^r\boldsymbol v_{[t]}^r{\boldsymbol k_{[t]}^r}^{\top}\\
&=\left(\sum_{i=1}^{r-1}\mathbf u_{[t]}^i{\boldsymbol k_{[t]}^i}^{\top}\right)
  \left(\mathbf I-\beta_{[t]}^r\boldsymbol k_{[t]}^r{\boldsymbol k_{[t]}^r}^{\top}\right)
  +\beta_{[t]}^r\boldsymbol v_{[t]}^r{\boldsymbol k_{[t]}^r}^{\top}\\
&=\sum_{i=1}^{r-1}\mathbf u_{[t]}^i{\boldsymbol k_{[t]}^i}^{\top}
  -\beta_{[t]}^r\sum_{i=1}^{r-1}\mathbf u_{[t]}^i
  \left({\boldsymbol k_{[t]}^i}^{\top}\boldsymbol k_{[t]}^r\right){\boldsymbol k_{[t]}^r}^{\top}
  +\beta_{[t]}^r\boldsymbol v_{[t]}^r{\boldsymbol k_{[t]}^r}^{\top}\\
&=\sum_{i=1}^{r-1}\mathbf u_{[t]}^i{\boldsymbol k_{[t]}^i}^{\top}
  +\underbrace{\beta_{[t]}^r\left(\boldsymbol v_{[t]}^r-\sum_{i=1}^{r-1}\mathbf u_{[t]}^i
  \left({\boldsymbol k_{[t]}^i}^{\top}\boldsymbol k_{[t]}^r\right)\right)}_{\mathbf u_{[t]}^r}{\boldsymbol k_{[t]}^r}^{\top}\\
&=\sum_{i=1}^{r}\mathbf u_{[t]}^i{\boldsymbol k_{[t]}^i}^{\top}.
\end{aligned}
\tag{27}
```

- 两个归纳证明同时给出式（19）、（20）的表示与构造方法：$`\mathbf W_{[t]}`$ 表示本块对入口状态的累计修改，$`\mathbf U_{[t]}`$ 表示本块自身的累计写入。它们使式（24）、（25）可以使用矩阵乘法，但系数向量的构造仍依赖先前位置，尚未解决块内计算的串行性。

# 🐢 UT Transform: Appendix B.2

**UT 处理的是 W、U 的构造依赖。** 正文 §3.2 指出，WY 的向量递推按原形式执行时无法直接利用 Tensor Core。附录 B.2 因而从同一递推逐行整理出下三角系统，使系数构造能够使用三角求解与矩阵乘法。以下保留附录的逐行表达、系数定义和移项步骤。

W、U 的第 r 行分别是 $`{\mathbf w_{[t]}^r}^{\top}`$、$`{\mathbf u_{[t]}^r}^{\top}`$。将式（19）、（20）的向量递推写成行形式，得到 B.2 的起点：

```math
\begin{aligned}
\mathbf W_{[t]}[r,:]
&=\beta_{[t]}^r\mathbf K_{[t]}[r,:]
 -\beta_{[t]}^r\sum_{i=1}^{r-1}\mathbf W_{[t]}[i,:]
 \left(\mathbf K_{[t]}[i,:]\mathbf K_{[t]}[r,:]^{\top}\right),\\
\mathbf U_{[t]}[r,:]
&=\beta_{[t]}^r\mathbf V_{[t]}[r,:]
 -\beta_{[t]}^r\sum_{i=1}^{r-1}\mathbf U_{[t]}[i,:]
 \left(\mathbf K_{[t]}[i,:]\mathbf K_{[t]}[r,:]^{\top}\right).
\end{aligned}
\tag{28}
```

附录定义的两个矩阵为

```math
\mathbf B_{[t]}=\operatorname{diag}(\beta_{[t]}),\qquad
\mathbf L_{[t]}=\operatorname{tril}\left(\mathbf B_{[t]}\mathbf K_{[t]}\mathbf K_{[t]}^{\top},-1\right).
\tag{29}
```

**下三角矩阵恰好收集递推中每个先前位置的系数。** 它的第 r 行、第 i 列在 $`i<r`$ 时为 $`\beta_{[t]}^r{\boldsymbol k_{[t]}^r}^{\top}\boldsymbol k_{[t]}^i`$，其余位置为零。内积为标量，可以交换两个 key 的次序，因此

```math
\begin{aligned}
(\mathbf L_{[t]}\mathbf W_{[t]})[r,:]
&=\sum_{i=1}^{r-1}\beta_{[t]}^r
 \left(\mathbf K_{[t]}[r,:]\mathbf K_{[t]}[i,:]^{\top}\right)\mathbf W_{[t]}[i,:]\\
&=\beta_{[t]}^r\sum_{i=1}^{r-1}\mathbf W_{[t]}[i,:]
 \left(\mathbf K_{[t]}[i,:]\mathbf K_{[t]}[r,:]^{\top}\right).
\end{aligned}
\tag{30}
```

- 式（30）就是式（28）右侧被减去的求和项。将它移到左侧，并将所有行同时写出，得到附录中的 $`\mathbf W_{[t]}+\mathbf L_{[t]}\mathbf W_{[t]}=\mathbf B_{[t]}\mathbf K_{[t]}`$；对 U 完全相同。两组方程可并列写为

```math
(\mathbf I+\mathbf L_{[t]})
\begin{bmatrix}\mathbf W_{[t]}&\mathbf U_{[t]}\end{bmatrix}
=\mathbf B_{[t]}
\begin{bmatrix}\mathbf K_{[t]}&\mathbf V_{[t]}\end{bmatrix}.
\tag{31}
```

**三角求解与递推等价，关键是每一行的方程完全相同。** $`\mathbf I+\mathbf L_{[t]}`$ 的对角线均为 1，第 r 行只依赖此前已经确定的行，因而前向代入得到唯一解。B.2 随后左乘该系数矩阵的逆，得到前文式（22）中的 $`\mathbf T_{[t]}`$：

```math
\begin{aligned}
\mathbf T_{[t]}&=(\mathbf I+\mathbf L_{[t]})^{-1}\mathbf B_{[t]},\\
\mathbf W_{[t]}&=(\mathbf I+\mathbf L_{[t]})^{-1}\mathbf B_{[t]}\mathbf K_{[t]}
 =\mathbf T_{[t]}\mathbf K_{[t]},\\
\mathbf U_{[t]}&=(\mathbf I+\mathbf L_{[t]})^{-1}\mathbf B_{[t]}\mathbf V_{[t]}
 =\mathbf T_{[t]}\mathbf V_{[t]}.
\end{aligned}
\tag{32}
```

- 附录最后一步所说的以 V 替换 K，作用于**右端输入**；系数矩阵仍由 K 与写入门决定。求解 W 是计算入口状态如何被本块修改，求解 U 是计算当前 value 如何形成块内写入。二者共用一个系数矩阵，属于两组右端项；可以分别求解，也可以拼接右端项后一起求解。逆矩阵写法表达其代数解，实际实现可使用前向代入。

**W、U 的作用在代回状态更新时才合在一起。** 正文 §3.2 将 WY 表示代入式（18），逐项整理为

```math
\begin{aligned}
\mathbf S_{[t]}^r
&=\mathbf S_{[t]}\left(\mathbf I-\sum_{i=1}^{r}\mathbf w_{[t]}^i{\boldsymbol k_{[t]}^i}^{\top}\right)
 +\sum_{i=1}^{r}\mathbf u_{[t]}^i{\boldsymbol k_{[t]}^i}^{\top}\\
&=\mathbf S_{[t]}+\sum_{i=1}^{r}
 \left(\mathbf u_{[t]}^i-\mathbf S_{[t]}\mathbf w_{[t]}^i\right){\boldsymbol k_{[t]}^i}^{\top}.
\end{aligned}
\tag{33}
```

- $`\mathbf u_{[t]}^i`$ 表示当前块从零状态出发的写入系数，$`\mathbf S_{[t]}\mathbf w_{[t]}^i`$ 表示本块从入口记忆中消除的对应分量。两者之差才是实际叠加到状态中的增量。按行堆叠后，它正是式（24）、（25）共同使用的 $`\mathbf U_{[t]}-\mathbf W_{[t]}\mathbf S_{[t]}^{\top}`$，既决定块末状态，也决定块内输出。

- 分别准备 W、U，使两组系数都只依赖当前块的 K、V 与写入门，可以在不同 chunk 上并行计算。入口状态到达后，再计算上述修正。如果只保留 U，非零入口状态中的旧关联就不会经历本块应有的消除；若将入口状态提前合并到右端项，系数准备又会依赖前一个 chunk。**两组求解将块内系数计算与跨块状态传递分开。**

# 🦦 UT Construction and Triangular Solves

接下来进入 `torch_chunk_gated_delta_rule` 的块内求解。GDN §3.3 将 WY 系数扩展到带衰减的更新：历史写入从位置 i 传播到位置 r 时乘 $`\gamma_{[t]}^r/\gamma_{[t]}^i`$。该节给出的 value 系数递推为

```math
\widetilde{\mathbf u}_{[t]}^r
=\beta_{[t]}^r\left(
\boldsymbol v_{[t]}^r-\sum_{i=1}^{r-1}\widetilde{\mathbf u}_{[t]}^i
\left(\frac{\gamma_{[t]}^r}{\gamma_{[t]}^i}
{\boldsymbol k_{[t]}^i}^{\top}\boldsymbol k_{[t]}^r\right)\right).
\tag{34}
```

- 这一递推的归纳证明见 *Gated Delta Networks: Improving Mamba2 with Delta Rule* 附录 A：将此前的衰减外积和代入单步更新，再用 $`\alpha_{[t]}^r\gamma_{[t]}^{r-1}=\gamma_{[t]}^r`$ 合并系数，即可得到当前的 $`\widetilde{\mathbf u}_{[t]}^r`$。相较式（28），变化在于先前位置的系数增加累计衰减比值，因此三角系统的严格下三角部分相应加入 $`\Gamma_{[t]}`$。

本段源码需要得到两组系数，入口状态留到下一步使用：

| 源码结果 | 对应的量 | 在状态更新中的作用 |
| --- | --- | --- |
| `new_values` | $`\widetilde{\mathbf U}_{[t]}`$ | 表示已计入块内纠正与衰减关系的 value 系数 |
| `k_cumdecay` | $`\overleftarrow{\mathbf W}_{[t]}`$ | 作用于入口状态，计算需要扣除的历史关联分量 |

**系统系数、块内读出权重与入口衰减分别准备。** `ut_system` 的严格下三角部分编码式（34）的历史依赖；`intra_chunk_attn` 保存当前 Q/K 之间带衰减的因果读出权重，留供后续输出计算；`decayed_k_beta` 则将入口状态传播到各位置所需的累计衰减乘入第二组右端项。`intra_chunk_attn` 在此处已经生成，后面对 `query`、`key` 的重新赋值不会改变它。

```python
# ut_system/intra_chunk_attn: [B, H, num_chunks, chunk_size, chunk_size]
ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
# decayed_k_beta: [B, H, num_chunks, chunk_size, d_k]
decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)
```

**第一组右端项是 `v_beta`。** 对式（34）移项并堆叠各行，GDN §3.3 给出的 UT 形式为

```math
\widetilde{\mathbf U}_{[t]}
=\left[\mathbf I+\operatorname{strictLower}\left(
\operatorname{diag}(\beta_{[t]})
\left(\Gamma_{[t]}\odot\mathbf K_{[t]}\mathbf K_{[t]}^{\top}\right)\right)\right]^{-1}
\operatorname{diag}(\beta_{[t]})\mathbf V_{[t]}.
\tag{35}
```

- 常规执行分支直接求解该系统。`upper=False` 选择下三角，`unitriangular=True` 将对角线视为 1，并忽略 `ut_system` 中原有的对角元素。因此实际系数矩阵是 $`\mathbf I+\operatorname{strictLower}(\texttt{ut\_system})`$，无需显式加单位矩阵或构造一般矩阵的逆。所得 `new_values` 已处理当前块内部的更新依赖，尚未使用入口递推状态。

```python
new_values = torch.linalg.solve_triangular(
    ut_system, v_beta, upper=False, unitriangular=True)
```

**第二组右端项是 `decayed_k_beta`，使用相同的系统系数。** GDN §3.3 定义 $`\overleftarrow{\mathbf w}_{[t]}^r=\gamma_{[t]}^r\mathbf w_{[t]}^r`$。将前文式（19）的 w 递推乘以 $`\gamma_{[t]}^r`$，并把历史项写回带箭头的系数，得到

```math
\begin{aligned}
\overleftarrow{\mathbf w}_{[t]}^r
&=\gamma_{[t]}^r\mathbf w_{[t]}^r\\
&=\beta_{[t]}^r\gamma_{[t]}^r\boldsymbol k_{[t]}^r
-\beta_{[t]}^r\sum_{i=1}^{r-1}\overleftarrow{\mathbf w}_{[t]}^i
\left(\frac{\gamma_{[t]}^r}{\gamma_{[t]}^i}
{\boldsymbol k_{[t]}^i}^{\top}\boldsymbol k_{[t]}^r\right).
\end{aligned}
\tag{36}
```

- 式（34）、（36）对历史位置的依赖系数相同，区别仅在右端：前者是 $`\beta_{[t]}^r\boldsymbol v_{[t]}^r`$，后者是 $`\beta_{[t]}^r\gamma_{[t]}^r\boldsymbol k_{[t]}^r`$。这说明第二次调用可以复用 `ut_system`，而 `k_cumdecay` 的额外累计衰减必须保留：它描述入口记忆传播到当前写入位置时的影响。

```python
k_cumdecay = torch.linalg.solve_triangular(
    ut_system, decayed_k_beta, upper=False, unitriangular=True)
```

两次求解完成后，各 chunk 的 $`\widetilde{\mathbf U}_{[t]}`$ 与 $`\overleftarrow{\mathbf W}_{[t]}`$ 均已准备好。下一步才引入入口状态，以 $`\widetilde{\mathbf U}_{[t]}-\overleftarrow{\mathbf W}_{[t]}\mathbf S_{[t]}^{\top}`$ 形成 `v_new`，用于块内输出和块末状态更新。

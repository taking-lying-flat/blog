# Qwen3.5 Gated DeltaNet

`Qwen3.5` 的 `Gated DeltaNet` 将历史压缩为一个固定大小的矩阵：每个 token 先衰减旧状态，再读取当前 key 已经对应的 value，最后把预测误差写回状态。推理时，这个过程可以逐 token 执行；训练时，同一递推被改写成分块矩阵运算，让大部分工作交给 Tensor Core

本文沿着[原文的数学推导与 kernel 解析](https://zhuanlan.zhihu.com/p/2007937984738129405)，重新梳理 `Linear Attention → Delta Rule → Gated Delta Rule → Chunkwise / Recurrent`，并对照当前代码更新实现细节。公式统一采用代码默认的 **`[K, V]` 状态布局**；原文和部分论文使用 `[V, K]`，两种记法互为转置

核对日期为 **2026 年 9 月 26 日**。正式版以 PyPI 与 GitHub release 交叉确认，主分支固定到提交，避免后续代码变化影响阅读

| 项目 | 最新正式版 | 同时核对的主分支 |
| --- | --- | --- |
| Transformers | [v5.17.0](https://github.com/huggingface/transformers/releases/tag/v5.17.0) | [`27166ea`](https://github.com/huggingface/transformers/tree/27166ea03f12c940f23176a904ab1d2ff1a3dcbb) |
| Flash Linear Attention | [v0.5.2](https://github.com/fla-org/flash-linear-attention/releases/tag/v0.5.2) | [`954438d`](https://github.com/fla-org/flash-linear-attention/tree/954438d1fcb5e1bb05c22f9908de9c5c2df74ae5) |

正文以正式版的默认 Triton 路径为主，主分支差异单独说明。`Qwen3.5-9B` 用于展示具体尺寸；其他尺寸和 MoE 模型需要读取各自的 `text_config`

## 1. 从 Attention 到状态矩阵

### 1.1 线性复杂度来自哪里

设序列长度为 $`T`$，每个 head 的 key、value 维度分别为 $`d_k,d_v`$。标准因果 Attention 为

```math
O=\operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}+M_{\mathrm{causal}}\right)V
```

它需要计算 token 之间的两两关联。`FlashAttention` 通过分块、在线 softmax 和重计算减少 HBM 访问及中间结果存储，但保留了关于序列长度的二次计算量。实际何时受计算吞吐限制，取决于 GPU、head 维度、batch、dtype 和 kernel，不能用一个固定序列长度作为通用分界

先考虑不带归一化分母的点积型线性注意力。把每个 token 的 $`q_t,k_t,v_t`$ 都视为列向量，定义

```math
S_t=S_{t-1}+k_tv_t^{\top},
\qquad o_t=S_t^{\top}q_t,
\qquad S_t\in\mathbb R^{d_k\times d_v}
```

若 $`S_0=0`$，展开后得到

```math
o_t=\sum_{j\le t}(q_t^{\top}k_j)v_j
```

矩阵形式仍可写成 $`O=\operatorname{tril}(QK^{\top})V`$，但实现时不必物化完整的 $`T\times T`$ 矩阵，而是逐步维护 $`S_t`$。每步主要处理 $`d_k\times d_v`$ 个元素，总工作量随 $`T`$ 线性增长

这里的状态矩阵相当于一个线性关联存储器。向量 $`k_t`$ 指定读写方向，$`v_t`$ 是希望存入的信息。不同 key 不一定正交，因此向一个方向写入时，可能改变其他相关 key 的读出值；这种干扰没有一个等于 $`d_kd_v`$ 的硬性 token 数阈值

一般的核化线性注意力还可以引入特征映射和归一化状态。本文采用上面的简化形式，目的是推导 Delta Rule，并不把所有线性注意力都视为同一种 softmax 近似

### 1.2 遗忘与修正是两种操作

为旧状态增加一个标量衰减，得到

```math
S_t=\alpha_tS_{t-1}+k_tv_t^{\top},\qquad 0<\alpha_t<1
```

它可以清除陈旧信息，但所有方向都会同时衰减。这个式子便于理解 Mamba2 一类标量状态衰减；GLA 更一般地支持按通道的门控，不能将它的所有形式都缩成同一个标量 $`\alpha_t`$

`DeltaNet` 则读取当前 key 已经关联的 value，只写入误差

```math
\widehat v_t=S_{t-1}^{\top}k_t,
\qquad e_t=\beta_t(v_t-\widehat v_t),
\qquad S_t=S_{t-1}+k_te_t^{\top}
```

展开后是

```math
S_t=(I-\beta_tk_tk_t^{\top})S_{t-1}
    +\beta_tk_tv_t^{\top}
```

若 $`\lVert k_t\rVert_2=1`$ 且 $`\beta_t=1`$，更新后有 $`S_t^{\top}k_t=v_t`$，即当前 key 的读出被替换为新 value。对于另一个 key $`r`$，更新引起的变化为

```math
(S_t-S_{t-1})^{\top}r=e_t(k_t^{\top}r)
```

因此，**只有与当前 key 正交的方向才完全不受这次 delta 更新影响**。把它类比成修改字典中的一个条目很直观，但真实 key 是向量，不是彼此隔离的离散地址

## 2. Gated Delta Rule：先衰减，再写入误差

### 2.1 四行递推

`Gated DeltaNet` 把上述两种操作组合起来。用 $`\bar S_t`$ 表示衰减后的临时状态

```math
\begin{aligned}
\bar S_t&=\alpha_tS_{t-1},\\
\widehat v_t&=\bar S_t^{\top}k_t,\\
e_t&=\beta_t(v_t-\widehat v_t),\\
S_t&=\bar S_t+k_te_t^{\top},\qquad o_t=S_t^{\top}q_t.
\end{aligned}
```

等价地

```math
S_t=\alpha_t(I-\beta_tk_tk_t^{\top})S_{t-1}
    +\beta_tk_tv_t^{\top}
```

**读取旧 value 时，状态已经乘过 $`\alpha_t`$**。若从未衰减的状态读取，再在别处补一个衰减，会得到不同的更新规则。代码中的 `g` 是 $`\log\alpha_t`$，所以第一步使用 `exp(g)`，不是直接乘 `g`

以一个 $`2\times2`$ 状态为例，令

```math
S_{t-1}=I,\qquad
k_t=\begin{bmatrix}1\\0\end{bmatrix},\qquad
v_t=\begin{bmatrix}0.7\\0.3\end{bmatrix},\qquad
\alpha_t=0.9,\quad\beta_t=1
```

衰减后读出 $`\widehat v_t=[0.9,0]^{\top}`$，误差为 $`e_t=[-0.2,0.3]^{\top}`$，于是

```math
S_t=
\begin{bmatrix}0.9&0\\0&0.9\end{bmatrix}
+
\begin{bmatrix}-0.2&0.3\\0&0\end{bmatrix}
=
\begin{bmatrix}0.7&0.3\\0&0.9\end{bmatrix}
```

在本文的 `[K, V]` 布局下，第一个 key 基方向对应状态的第一行。若采用原文的 `[V, K]` 布局，将整个矩阵转置，更新就落在第一列；二者表示同一个算子

### 2.2 在线学习视角

把状态当作一个线性模型，令它用 key 预测 value

```math
\ell_t(S)=\frac12\lVert S^{\top}k_t-v_t\rVert_2^2,
\qquad
\nabla_S\ell_t=k_t(S^{\top}k_t-v_t)^{\top}
```

在 $`\bar S_t=\alpha_tS_{t-1}`$ 处做一步学习率为 $`\beta_t`$ 的梯度更新，就得到 Gated Delta Rule。这里被更新的是每条序列自己的状态，并不是推理时对模型投影权重调用 optimizer。这一视角解释了为什么写入误差比不断累加原始 value 更有针对性

推导与模型动机可参照 [Gated Delta Networks 论文](https://arxiv.org/abs/2412.06464)。下面把这个小型线性模型放回实际的 Qwen3.5 层

## 3. Qwen3.5：从 hidden states 到三个 gate

### 3.1 一层的数据流

`Qwen3.5-9B` 的配置共有 32 个文本层，其中 24 个使用 GDN，8 个使用 Full Attention，按三个 GDN 层接一个 Full Attention 层排列。Full Attention 的输出门控不会将它自动变成稀疏注意力；其可见范围与实现仍应按该层的 attention 路径理解。[模型配置](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/config.json)

下图只展开 GDN token mixer，外层残差连接与 FFN 略去

```text
hidden_states [B,T,D]
    │
    ├─ in_proj_qkv ─ depthwise causal Conv1d ─ SiLU ─ split ─ Q,K,V
    │                                                          │
    ├─ in_proj_b ─ sigmoid ─ beta ──────────────────────────────┤
    │                                                          │
    ├─ in_proj_a ─ softplus / A_log / dt_bias ─ g ──────────────┤
    │                                                          ▼
    │                                                Gated Delta Rule
    │                                                          │
    └─ in_proj_z ─ z ──────────────────────────────── RMSNorm + SiLU gate
                                                               │
                                                            out_proj
                                                               │
                                                        output [B,T,D]
```

这里有三个用途不同的控制量

| 变量 | 生成方式 | 控制对象 |
| --- | --- | --- |
| $`\alpha=\exp(g)`$ | `in_proj_a`、`A_log`、`dt_bias` | 旧状态的衰减 |
| $`\beta`$ | `sigmoid(in_proj_b(x))` | delta 误差的写入强度 |
| $`z`$ | `in_proj_z(x)` | 递推输出经过 RMSNorm 后的逐通道门控 |

`Q/K/V` 经过短卷积，`a/b/z` 由当前层输入直接投影。短卷积提供局部因果混合，每个通道拥有自己的卷积核；它本身不会把训练标签向后移动，也不等同于 next-token loss 的 shift。GDN 分支没有在这里对 Q/K 应用 RoPE；Full Attention 分支另有位置编码路径

### 3.2 张量尺寸与 GVA

以下尺寸取自 `Qwen3.5-9B` 配置，其中 `H_K` 表示 Q/K head 数，`H_V` 表示 value head 数

| 张量或参数 | 一般尺寸 | 9B 示例 |
| --- | --- | --- |
| `hidden_states` | `[B,T,D]` | `D=4096` |
| 卷积前后合并的 QKV | `[B,T,2 H_K d_k + H_V d_v]` | 最后一维 `8192` |
| `Q/K`，复制前 | `[B,T,H_K,d_k]` | `H_K=16, d_k=128` |
| `V` | `[B,T,H_V,d_v]` | `H_V=32, d_v=128` |
| `g/beta` | `[B,T,H_V]` | 每个 value head 一个标量 |
| `z` | `[B,T,H_V,d_v]` | 与递推输出逐通道对应 |
| 递推状态 | `[N,H_V,d_k,d_v]` | 等长 batch 中 `N=B` |

多个 value head 共享一个 Q/K head，称为 `Grouped Value Attention`。若 `H_K=2,H_V=4`，对应关系是 `0→0、1→0、2→1、3→1`

Transformers 5.17.0 的 Qwen3.5 实现仍会通过 `repeat_interleave` 显式复制 Q/K。FLA 0.5.2 的算子已经能直接接受不同的 head 数，在 kernel 内按下式映射

```python
qk_head = value_head // (num_value_heads // num_key_heads)
```

因此要区分两层：**FLA 算子具备原生 GVA 能力，当前 Transformers 模型入口仍先展开 Q/K**。不能只看到底层支持 GVA，就断言模型调用已经省掉了这次复制。[Transformers 模型入口](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L504-L663)、[FLA 算子接口](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gated_delta_rule/chunk.py#L395)

### 3.3 衰减、归一化与输出门控

状态衰减的参数化为

```math
g_{t,h}=-\exp(A_{\log,h})\,operatorname{softplus}(a_{t,h}+b_{dt,h}),
\qquad\alpha_{t,h}=\exp(g_{t,h})
```

对有限实数输入，$`g<0`$，从而 $`0<\alpha<1`$；浮点运算中极强衰减可能下溢为零。`g` 在模型入口以 FP32 计算，避免低精度参数变换带来额外溢出风险

当前 `A` 的初始化区间是 `(0.01,16)`，`A_log=log(A)`，下界避开零。实际模型还会执行 `PreTrainedModel` 的初始化逻辑；加载 checkpoint 时则使用训练好的权重。因此，仅把构造函数里的均值代入，不能得出已发布模型从近乎无记忆状态运行的结论

Q/K 的 L2 归一化与 query 缩放是两个步骤。将卷积后的向量记为 $`q_t^{\mathrm{raw}},k_t^{\mathrm{raw}}`$，递推真正使用

```math
k_t=\frac{k_t^{\mathrm{raw}}}
{\sqrt{\lVert k_t^{\mathrm{raw}}\rVert_2^2+\varepsilon}},
\qquad
q_t=\frac{1}{\sqrt{d_k}}
\frac{q_t^{\mathrm{raw}}}
{\sqrt{\lVert q_t^{\mathrm{raw}}\rVert_2^2+\varepsilon}}
```

后面的公式统一把 $`1/\sqrt{d_k}`$ 吸收到 $`q_t`$ 中，避免重复乘 `scale`。前文关于单位 key 的精确替换是理想数学条件；代码使用带 $`\varepsilon`$ 的归一化，范数与 1 可能有微小差别

得到递推输出 $`o_t`$ 后，模型对每个 value head 的最后一维做 RMSNorm，再乘输出门控

```math
y_t=
\left(w\odot\frac{o_t}{\sqrt{\operatorname{mean}(o_t^2)+\varepsilon}}\right)
\odot\operatorname{SiLU}(z_t)
```

这里是 **归一化 `o`，然后乘 `SiLU(z)`**，不是 `sigmoid(z)`，也不是 `o * RMSNorm(z)`。最后拼接各 head，经 `out_proj` 回到 hidden size。[门控归一化与参数初始化](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L216-L232)

## 4. Recurrent：一个 token 如何更新状态

### 4.1 kernel 中的对应关系

忽略地址计算、dtype 转换和边界掩码，默认 `[K,V]` 布局下的核心操作可写成

```python
state = state * exp(g_t)
prediction = sum(state * k_t[:, None], axis=0)
error = beta_t * (v_t - prediction)
state = state + k_t[:, None] * error[None, :]
out = sum(state * q_t[:, None], axis=0)
```

`sum(..., axis=0)` 沿 key 维归约，得到 value 向量；外积写回的是 `[K,V]` 矩阵。`out` 读取更新后的状态，因此当前 token 的写入也会参与当前输出

FLA 的 recurrent kernel 用 FP32 保存局部状态，沿 value 维切 tile。不同 tile 处理同一个 key 方向下的不同 value 通道，序列内的时间步仍按顺序执行。这条通用 Triton 路径使用逐元素乘加与归约，循环中没有 `tl.dot`；这描述的是该实现，并不意味着所有可能的 decode 优化都只能采用同样的硬件映射

FLA 0.5.2 的 launch grid 是 `(N_V, N*H_V)`；主分支将其压成一维，再从 `pid` 解码 value tile、序列和 head。调度形状改变了，状态递推公式没有改变。[正式版 recurrent](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gated_delta_rule/fused_recurrent.py)、[主分支对应提交](https://github.com/fla-org/flash-linear-attention/blob/954438d1fcb5e1bb05c22f9908de9c5c2df74ae5/fla/ops/gated_delta_rule/fused_recurrent.py)

### 4.2 模型什么时候使用 recurrent

当前 Transformers 入口的判断同时检查缓存与本次输入长度

| 条件 | GDN 计算路径 |
| --- | --- |
| 已有该层状态，且本次 `seq_len == 1` | recurrent |
| 首次 prefill、无已有状态 | chunk |
| 已有状态，但本次输入多个 token | chunk，以上次状态作为 `initial_state` |

所以单 token 输入并不必然走 recurrent：首次输入长度为 1 时，没有已有状态，仍可进入 chunk 路径。卷积更新还有 `record_past` 分支，不能仅根据 `model.training` 判断所有执行路径

FLA 当前这个 `fused_recurrent_gated_delta_rule` 的 backward 明确抛出 `NotImplementedError`。训练通常使用支持反向的 chunk 算子；这属于当前实现能力，不是递推公式在数学上不可微

函数名也不能单独证明实际运行了哪个 kernel。Transformers 5.17.0 的装饰器按配置优先选择 Hub kernel，再尝试本地 FLA，最后回退到 PyTorch 参考实现；导出时还有强制使用参考路径的处理。[内核选择逻辑](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/integrations/hub_kernels.py#L829-L896)

## 5. Chunkwise：把时间依赖移进下三角系统

### 5.1 三种衰减必须分清

现在只看一个长度为 $`C`$ 的 chunk，用 $`S_{\mathrm{in}}`$ 表示它开始前的状态。chunk 内位置从 1 开始，定义

```math
G_i=\sum_{r=1}^{i}g_r,
\qquad\gamma_i=\exp(G_i),
\qquad
D_{ij}=\begin{cases}
\exp(G_i-G_j),&j\le i,\\
0,&j>i.
\end{cases}
```

$`D_{ij}`$ 表示第 $`j`$ 个 token 写入后，到第 $`i`$ 个 token 为止经历的衰减，所以 $`D_{ii}=1`$。它与旧状态进入 chunk 时经历的 $`\gamma_i`$ 不同

| 用途 | 衰减系数 | 含义 |
| --- | --- | --- |
| 当前 query 读取块前状态 | $`\gamma_i`$ | 从 chunk 开始到位置 $`i`$ |
| 位置 $`i`$ 读取块内第 $`j`$ 次写入 | $`D_{ij}`$ | 写入之后从 $`j+1`$ 到 $`i`$ |
| 第 $`i`$ 次写入进入块末状态 | $`\gamma_C/\gamma_i`$ | 从 $`i+1`$ 到 chunk 末尾 |

若四个位置都取 $`\alpha=0.9`$，三组系数中的前后两组为

| 位置 | 读取块前状态：$`\gamma_i`$ | 写入到块末：$`\gamma_C/\gamma_i`$ |
| --- | ---: | ---: |
| 1 | 0.9 | 0.729 |
| 2 | 0.81 | 0.81 |
| 3 | 0.729 | 0.9 |
| 4 | 0.6561 | 1 |

原文中的左右箭头就是对这两种乘法的简写。下面直接使用 $`\gamma`$，并显式保留 $`D`$，避免把带衰减的因果权重误写成普通 0/1 mask

### 5.2 从误差递推得到线性系统

仍将每次实际写入的 value 向量记为 $`e_i`$。把前面所有写入展开

```math
S_i=\gamma_iS_{\mathrm{in}}
+\sum_{j\le i}D_{ij}k_je_j^{\top}
```

计算第 $`i`$ 次误差时，读取的是已经衰减、尚未写入本次误差的状态。因此

```math
e_i^{\top}
=\beta_i v_i^{\top}
-\beta_i\gamma_i k_i^{\top}S_{\mathrm{in}}
-\sum_{j<i}\beta_iD_{ij}(k_i^{\top}k_j)e_j^{\top}
```

把每个 token 作为矩阵的一行，令 $`B_\beta=\operatorname{diag}(\beta)`$，并定义严格下三角矩阵

```math
L=\operatorname{tril}\!\left(
B_\beta\bigl((KK^{\top})\odot D\bigr),-1
\right)
```

那么所有误差组成的矩阵 $`E\in\mathbb R^{C\times d_v}`$ 满足

```math
(I+L)E=B_\beta V-B_\beta\operatorname{diag}(\gamma)KS_{\mathrm{in}}
```

$`I+L`$ 的对角线全为 1，所以这是一个单位下三角系统。定义

```math
\begin{aligned}
A&=(I+L)^{-1},\\
U&=AB_\beta V,\\
W&=AB_\beta\operatorname{diag}(\gamma)K,\\
E&=U-WS_{\mathrm{in}}.
\end{aligned}
```

这就是代码中 `A / u / w / v_new` 的来源。**`A` 只表示逆矩阵，尚未包含右侧的 $`B_\beta`$**；某些文献把两者合并记成另一个变换矩阵，阅读时要先核对定义，不能再额外乘一次 beta

$`W`$ 与块前状态相乘，所以它需要额外携带 $`\gamma_i`$。$`U`$ 中也有衰减，但它通过 $`A`$ 内的 $`D_{ij}`$ 影响先前写入之间的相互作用；两者的衰减来源不同

### 5.3 输出与块末状态

一旦得到 $`E`$，chunk 输出为

```math
\boxed{
O=\operatorname{diag}(\gamma)QS_{\mathrm{in}}
+\bigl((QK^{\top})\odot D\bigr)E
}
```

第一项读取块前状态，第二项读取块内写入。第二项的 **$`D`$ 同时包含因果掩码与区间衰减**；只写 $`\operatorname{tril}(QK^{\top})E`$ 会漏掉旧写入到当前 query 之间的遗忘

块末状态则是

```math
\boxed{
S_{\mathrm{out}}=
\gamma_CS_{\mathrm{in}}
+\left(\operatorname{diag}\!\left(\frac{\gamma_C}{\gamma}\right)K\right)^{\top}E
}
```

两条公式可直接核对尺寸：`Q @ S_in` 是 `[C,V]`，`QKᵀ` 是 `[C,C]`，`Kᵀ @ E` 是 `[K,V]`

`U/W` 只依赖本 chunk 的输入，可以跨 chunk 并行预计算；`E` 还依赖 $`S_{\mathrm{in}}`$，这一步需要沿 chunk 顺序传播状态；各 chunk 的入口状态保存下来后，输出计算又可以跨 chunk 并行。**分块减少了时间方向的串行步数，并没有消除块间依赖**

```text
每个 chunk 独立：     K,V,g,beta ─── A ─── U,W
                                          │
沿 chunk 传状态：    S_in ─── E=U-W@S_in ── S_out ─── 下一个 chunk
                       │          │
每个 chunk 独立：     Q ────────── O
```

这些变换在实数算术下与逐 token 递推等价。浮点下，chunk size、归约顺序和中间 dtype 都可能改变末位结果，不能把数学等价写成逐位一致

## 6. 最新 FLA 如何执行这些公式

### 6.1 数学阶段与 kernel 边界

原文按 `cumsum → KKT → solve_tril → W/U → h → o` 六个阶段展开，有助于理解数学。FLA 0.5.2 的默认前向已将 KKT 和下三角求解放到同一个 kernel 中；`W/U` 仍由另一个 kernel 计算

```text
chunk_gated_delta_rule
  ├─ 可选 Q/K L2 normalization、beta activation
  └─ chunk_gated_delta_rule_fwd
       ├─ chunk_local_cumsum，或 gate activation + cumsum
       ├─ chunk_gated_delta_rule_fwd_intra
       │    ├─ KKT + solve，默认 C=64 的融合 kernel
       │    └─ recompute_w_u_fwd
       ├─ chunk_gated_delta_rule_fwd_h
       └─ chunk_fwd_o
```

上图描述默认 Triton 实现，省略 CP 前后处理。当前实现另有 backend dispatch，不能把图中步骤数等同于所有设备和配置上的固定 kernel launch 数。[前向组织](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gated_delta_rule/chunk.py)、[融合入口](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gated_delta_rule/chunk_fwd.py#L332-L427)

### 6.2 `cumsum`：从自然对数转换到 `exp2`

数学上使用 $`G_i=\sum_{j\le i}g_j`$。当前 FLA 的 chunk 路径在前缀和时乘 `RCP_LN2`，存入

```math
G_i^{(2)}=\frac{1}{\ln2}\sum_{j\le i}g_j,
\qquad
2^{G_i^{(2)}-G_j^{(2)}}=\exp(G_i-G_j)
```

后面的 kernel 使用 `exp2`，二者是一套配合。若照搬旧版伪代码，把预处理保留为自然对数累加、后面却改用 `exp2`，就改变了衰减强度

当前 API 还支持 `use_gate_in_kernel=True`：输入原始 `a`，由 kernel 结合 `A_log/dt_bias` 计算激活与前缀和。Qwen3.5 的上述 Transformers 入口已经在外面算好 `g`，不要再把它当作原始 `a` 激活一次

log-space 的好处是将连乘变成求和，并用差值计算局部区间衰减；这降低了直接计算两个极小乘积再相除的风险，但不能保证最终指数永不下溢

### 6.3 `KKT + solve`：严格下三角与块矩阵逆

默认 $`C=64`$ 时，融合 kernel 将 $`64\times64`$ 的系统拆成四个 $`16\times16`$ 对角块与六个非对角下三角块。它先计算 key 内积、衰减和 beta，再对角块求解，最后合并为整个逆矩阵

用一个二乘二块矩阵看合并关系最清楚

```math
M=\begin{bmatrix}M_{11}&0\\M_{21}&M_{22}\end{bmatrix},
\qquad
M^{-1}=\begin{bmatrix}
M_{11}^{-1}&0\\
-M_{22}^{-1}M_{21}M_{11}^{-1}&M_{22}^{-1}
\end{bmatrix}
```

左下角是整个逆矩阵的一个块，**不是 $`M_{21}^{-1}`$**。分块让求解中的工作能利用矩阵乘法，并减少中间严格下三角矩阵写回 HBM 的开销；它不等于从数学上消除了全部三次项

当前这条 intra-chunk 实现支持 `C=16/32/64`，默认取 64；16/32 使用分开的 KKT 与求解路径。原文中的通用 `solve_tril` 调优经验仍可用于理解取舍，但不能据此判断默认 64 路径的精度配置与性能，因为它已经走不同的融合 kernel

在 dtype 方面，融合 kernel 内有 FP32 累积，求解结果 `A` 的缓冲则按 key dtype 分配。因此 BF16 输入下，不应把整条下三角求解与后续矩阵运算描述为全程 FP32

### 6.4 `W/U`：两次矩阵乘法共享同一个逆

这一阶段对应

```math
U=A(\beta\odot V),
\qquad
W=A\bigl((\beta\odot\gamma)\odot K\bigr)
```

此处 $`\beta,\gamma`$ 按行广播。kernel 在 value 维分块计算 `U`，在 key 维分块计算 `W`；二者共享 `A`，但右侧向量不同。GVA 下，`K` 使用映射后的 Q/K head，`beta/g/U/W` 按 value head 定位。[WY 实现](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gated_delta_rule/wy_fast.py)

### 6.5 `forward_h`：保存的是每个 chunk 的入口状态

状态 kernel 按序列、value head 和 value tile 并行，在一个 program 内按顺序遍历该序列的 chunk。每一步的逻辑是

1. 保存当前 `S_in`，供输出 kernel 读取
2. 计算并保存 `E = U - W @ S_in`，代码中叫 `v_new`
3. 将 `S_in` 乘以整个 chunk 的衰减
4. 把每次写入按距 chunk 末尾的长度衰减后，累加到状态
5. 将更新后的状态传给下一个 chunk

要区分三个存储位置

| 对象 | 用途 | 当前默认实现中的 dtype |
| --- | --- | --- |
| program 内局部状态 | chunk 间递推累积 | FP32 |
| `h` | 保存各 chunk 的入口状态，供输出和反向重计算使用 | key dtype |
| `final_state` | 整条序列结束后的状态 | FP32 |

`h` 的尺寸包含 chunk 数，不能用推理缓存大小代替训练时的中间激活大小。将 key 维拆成多个 64 宽的局部块，也不意味着总寄存器需求凭空消失；实际压力取决于编译后的布局、同时存活的值以及 tile 大小。[状态前向实现](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/common/chunk_delta_h.py)

### 6.6 `chunk_fwd_o`：输出必须保留区间衰减

输出 kernel 在同一段计算中完成块间读取与块内读取

```text
inter = Q @ S_in
intra = Q @ K.T
inter *= gamma[:, None]
intra *= interval_decay
O = inter + intra @ E
```

这里 `interval_decay` 就是带因果约束的 $`D`$。当前实现中的 `exp2(g_i-g_j)` 与前面的 log2 前缀和配套；输出用的是更新后的状态，因此块内矩阵包含对角线，而构造下三角系统 $`L`$ 时使用严格下三角

这两处 mask 的差别来自语义：计算本次误差时尚未写入当前 token，所以排除对角线；计算输出时已经写入，所以包含对角线。`QKᵀ` 的局部块在 kernel 内参与计算，不需要物化完整序列的注意力矩阵。[输出实现](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/common/chunk_o.py)

## 7. Varlen 与缓存：两种不同的边界

### 7.1 token 偏移和 chunk 偏移

变长输入通常按 `[1,total_tokens,H,D]` 打平，由 `cu_seqlens` 恢复各样本的逻辑边界。例如

```text
cu_seqlens = [0, 16, 128]
C = 64

sequence 0: tokens [0,16)     → 1 chunk
sequence 1: tokens [16,128)   → 2 chunks

chunk_indices = [[0,0], [1,0], [1,1]]
chunk_offsets = [0,1,3]
```

`cu_seqlens` 指向 token 空间；`chunk_indices` 将一个并行任务映射到序列及其局部 chunk；`chunk_offsets` 指向 `h` 等 chunk 级缓冲。序列 1 的首个 chunk 从 token 16 开始，不是从全局 token 64 开始

没有块间依赖的阶段可以把所有 chunk 展开并行；状态递推需要在每条序列内部保持顺序，并为每条序列加载自己的初始状态。把两条序列直接拼起来却不传边界，会让后一条继承前一条的记忆

FLA 主分支新增了公开的 `chunk_indices` 参数，允许调用者传入预计算索引；0.5.2 的公开入口没有这个同名显式参数，主要在内部生成。若想减少热路径的元数据处理，应先核对安装版本的函数签名。[固定主分支接口](https://github.com/fla-org/flash-linear-attention/blob/954438d1fcb5e1bb05c22f9908de9c5c2df74ae5/fla/ops/gated_delta_rule/chunk.py#L397)

GDN 算子识别样本边界还不够：前面的 causal convolution 也必须识别同一组边界。并且，Transformers 的纯 PyTorch 参考函数虽接收 `**kwargs`，其函数体没有实现与 FLA 相同的 packed `cu_seqlens` 分段计算，不能据此假定所有 fallback 都支持相同的 packed 输入语义

### 7.2 推理需要两份 GDN 状态

增量生成除了保存 recurrent state，还要保存短卷积所需的历史投影值

| 缓存 | 常规增量推理中的形状 | 随上下文长度增长 |
| --- | --- | --- |
| GDN recurrent state | `[B,H_V,d_k,d_v]` | 否 |
| GDN convolution state | `[B,2 H_K d_k + H_V d_v,conv_width]` | 否 |
| Full Attention K/V | `[B,H_KV,T,head_dim]`，各一份 | 是 |

只复用 recurrent state 而丢掉卷积历史，下一步的 Q/K/V 就已改变。当前 Transformers 的 `LinearAttentionLayer` 管理两类状态，并通常原地复制以维持缓存地址；`record_past` 等回滚模式可能保存更多卷积历史，上表描述普通增量生成。[缓存实现](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/cache_utils.py#L1017-L1105)

以 9B、batch 1、recurrent state FP32、卷积及 K/V BF16 为例，忽略分配对齐和管理元数据

- 24 层 GDN recurrent state：`24 × 32 × 128 × 128 × 4 bytes = 48 MiB`
- 24 层短卷积状态：`24 × 8192 × 4 × 2 bytes = 1.5 MiB`
- 8 层 Full Attention，在 8192 token 时：`8 × 2 × 4 × 8192 × 256 × 2 bytes = 256 MiB`

这解释了混合架构的缓存优势，也说明 **Qwen3.5 整个模型的缓存仍随上下文增长**。固定大小的是 GDN 状态部分，不是所有层的缓存

## 8. 反向传播：重计算与逆序状态传递

chunk 前向中的 `U/W/h/v_new` 并非都长期保留。当前 FLA 的反向会利用保存的输入、求解结果 `A` 等重新构建部分中间值，用计算换取显存

整体依赖顺序是：重计算 `U/W` 和状态 → 计算输出对 `E` 的局部梯度 → 沿 chunk 逆序传递状态梯度 → 汇总 Q/K/W 与 gate 梯度 → 对 WY 变换求导 → 合并并传播到原始 gate 输入

两个位置值得单独核对

**前缀和的梯度是后缀和。** 若 $`G_i=\sum_{j\le i}g_j`$，则

```math
\frac{\partial\mathcal L}{\partial g_j}
=\sum_{i\ge j}\frac{\partial\mathcal L}{\partial G_i}
```

实现使用 reverse cumsum，并在具体计算链中处理 `exp2` 与对数底数转换。若修改 gate 激活或融合位置，需要核对完整链式法则，不能只改前向指数函数

**矩阵微分与反向梯度要区分。** 对 $`A=(I+L)^{-1}`$，微分满足 $`\mathrm dA=-A(\mathrm dL)A`$；在以元素内积定义梯度时，对应

```math
\nabla_L\mathcal L
=-\operatorname{tril}\!\left(
A^{\top}(\nabla_A\mathcal L)A^{\top},-1
\right)
```

两侧的转置来自向量—雅可比积。kernel 可能通过转置加载、交换操作数来实现它，所以阅读 `tl.dot` 时要连同实际内存视图一起看，不能直接把微分式抄成梯度式。[反向组织](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gated_delta_rule/chunk.py#L126-L250)、[WY 反向](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gated_delta_rule/wy_fast.py)

## 9. 数值核对与性能边界

### 9.1 先验证公式，再验证低精度 kernel

本文使用独立的逐 token 参考实现和下三角分块实现，对照输出、最终状态以及 Q/K/V/g/beta/初始状态的梯度。FP64 验证覆盖长度 `1/3/7/17/65`、chunk size `1/4/16`，key/value 维度故意取不同的 `5/3`，避免方阵掩盖转置错误

| 核对项 | 最大绝对误差 |
| --- | ---: |
| FP64 递推与分块：输出 | `2.22e-16` |
| FP64 递推与分块：最终状态 | `1.94e-16` |
| FP64 递推与分块：梯度 | `2.66e-15` |
| Transformers 5.17.0 实际参考函数：FP32 输出 | `4.47e-7` |
| Transformers 5.17.0 实际参考函数：FP32 最终状态 | `4.17e-7` |

后一组直接提取正式版源码中的 PyTorch chunk/recurrent 函数，移除选择 kernel 的装饰器后执行，覆盖长度 `1/7/65/137`、非零初始状态以及 Q/K L2 normalization。它验证参考函数的等价性，不代表完整模型端到端评测

另外在 `NVIDIA RTX A1000 Laptop GPU`、`PyTorch 2.12.1+cu130` 上执行了两个版本的真实 Triton 算子，关闭可选 backend dispatch，使对照确实落在本文分析的默认实现。随机种子为 `20260926`；输入尺寸为 `B=1,T=137,H_K=2,H_V=4,d_k=32,d_v=24`，Q/K/V 为 BF16，g/beta/初始状态为 FP32

下面报告相对于 FP32 逐 token 参考的 $`\lVert x-x_{\mathrm{ref}}\rVert_2/\lVert x_{\mathrm{ref}}\rVert_2`$，数值经过四舍五入

| GPU 核对项 | FLA 0.5.2 | 主分支 `954438d` |
| --- | ---: | ---: |
| Chunk 输出相对误差 | `4.16e-3` | `4.16e-3` |
| Chunk 最终状态相对误差 | `3.34e-3` | `3.34e-3` |
| Q/K/V/g/beta/初始状态六组梯度的最大相对误差 | `5.36e-3` | `5.36e-3` |
| Recurrent 输出相对误差 | `1.67e-3` | `1.67e-3` |
| Recurrent 最终状态相对误差 | `1.14e-7` | `1.14e-7` |

同一组数据还检查了两类接口语义：将长度 `3/65/69` 的序列打包后，与分别调用 chunk 算子的输出、最终状态完全一致；将初始状态转成 `[V,K]` 并启用 `state_v_first=True` 后，输出一致，转回来的最终状态最大绝对差约 `1.19e-7`

这些是小尺寸算子验证，覆盖了非整块尾部、GVA、非零初始状态、变长边界和状态转置。它们不等于完整 Qwen3.5 的模型精度评测，也没有覆盖 CP、多卡或其他硬件后端。BF16 chunk 与 FP32 reference 的误差进一步说明了数学等价和浮点逐位一致的区别

### 9.2 性能不能只由复杂度推断

recurrent 的每步状态工作量约为 $`O(d_kd_v)`$；chunk 通过 $`C\times C`$ 的局部交互与状态矩阵乘法提高并行度。主要矩阵工作包括 $`O(Td_kd_v)`$ 与 $`O(TC(d_k+d_v))`$，下三角求解还有自身开销。当 chunk size 固定时，总工作量对序列长度保持线性

这并不直接给出某张 GPU 上的加速倍数。prefill 需要考虑 chunk 数、矩阵乘法效率与中间缓冲；单 token decode 则常受状态读写、launch 开销、并发序列数和 value tile 划分影响。低 batch 下，增加并行 program 可能有益，也可能引入更多 Q/K 重复加载；融合能够减少访存，但可能增加寄存器压力

原文的 cumsum 加速、求解拆分以及特定 decode kernel 收益属于其测量配置。本文没有复跑相同 benchmark，不将这些数字当作最新版的性能结论，也不把数学等价视为更改累积精度的充分依据

## 10. 阅读源码时的对照表

| 原文中的讲解点 | 本文对照当前实现后的处理 |
| --- | --- |
| 论文状态常记为 `[V,K]` | 全文统一成默认代码布局 `[K,V]`，显式说明转置 |
| 用字典条目解释 delta 更新 | 补充单位 key 条件与非正交 key 的相互干扰 |
| 输出门控的注释 | 明确为 `RMSNorm(o) × SiLU(z)` |
| 分块输出中的普通 causal mask | 补全区间衰减 $`D_{ij}`$，保留对角线 |
| 六个数学阶段逐一对应 kernel | 更新成默认 64 路径的 `KKT + solve` 融合 |
| 自然对数前缀和与 `exp` | 补充当前 chunk 路径的 `RCP_LN2 + exp2` 配套 |
| GVA 需要在外部复制 Q/K | 区分 Transformers 仍复制与 FLA 已原生支持 |
| chunk size 不影响数值 | 限定为实数算术等价，浮点结果需要容差验证 |
| 初始衰减率的估算 | 核对 `(0.01,16)`、模型初始化与 checkpoint 加载 |
| recurrent 不支持 backward | 限定到当前 FLA 算子实现 |
| 固定大小的状态缓存 | 同时计入卷积状态，保留 Full Attention 的增长项 |

继续阅读时，可将本文的源码链与 [Token Mixer I](../token-mixer-i/) 中的 Gated DeltaNet、Kimi Linear 和 Gated DeltaNet-2 推导对照。原始算法出处是 [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)，本次重写的阅读起点为[知乎原文](https://zhuanlan.zhihu.com/p/2007937984738129405)

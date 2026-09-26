# Gated Delta Networks：原理推导与分块算法实现

Qwen3.5 系列模型采用了 GDN（Gated Delta Networks）结构。本文结合 flash-linear-attention 的实现，从数学原理、分块算法和 GPU kernel 三个层面分析 GDN。

前半部分介绍线性注意力、Delta Rule 与 Gated Delta Rule 的公式推导；后半部分解析模型调用、Chunk-wise 算法和 Recurrent 算法。分块算法的推导见第 3.3 节，实现见第 5 节。

本文的记号约定为：$`\mathbf S\in\mathbb R^{d_v\times d_k}`$；FLA 默认以转置布局 $`[K,V]`$ 存储状态，代码中的 `h` 对应 $`\mathbf S^\top`$。代码更新至 Transformers v5.17.0 与 FLA v0.5.2，函数体保留完整，装饰器及模块级配置从略。

论文：[Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)。参考实现：Flash Linear Attention。

## 1. 概述

### 从标准 Attention 到 Linear Attention

标准 Attention 的计算复杂度为 $`O(L^2d)`$。若显式存储注意力矩阵，需要 $`O(L^2)`$ 的空间；FlashAttention 通过分块和算子融合避免将完整注意力矩阵写入 HBM，但没有改变二次计算复杂度。其性能受序列长度、head 维度、数据类型和硬件共同影响，不能用固定的序列长度划分计算瓶颈与带宽瓶颈。

Linear Attention 以可分解的特征内积替代 softmax 权重。忽略因果掩码时，可以利用结合律先计算 $`\mathbf K^\top\mathbf V`$，再左乘 $`\mathbf Q`$；在自回归场景中，则转化为维护固定大小状态矩阵的递推。固定 head 维度后，计算量随序列长度线性增长。

逐 token 递推在时间维度上存在串行依赖，难以充分利用 Tensor Core。因此，训练采用 Chunk-wise 分块并行算法：块间递推状态，块内使用矩阵乘法。具体的性能交叉点取决于模型配置、算法开销和硬件；GDN 还包含 WY 表示与下三角求解，不能直接套用其他线性注意力算子的性能结果。

### GDN: Gated Delta Networks

线性注意力将历史压缩到固定大小的状态矩阵中，不同键值对可能相互干扰；没有衰减时，早期信息也会持续累积。针对这两个问题，可以分别引入遗忘与定向更新：

- **门控衰减**：以 Mamba2 的标量衰减为例，用 $`\alpha_t`$ 缩放历史状态。GLA 还可采用按通道的门控。
- **Delta Rule**：沿当前 key 的方向修正状态，写入当前 value 与已有预测之间的误差。

GDN 将二者结合，在递推中同时引入衰减系数 $`\alpha_t`$ 和更新强度 $`\beta_t`$。其分块形式涉及 WY 表示与下三角求解，下面先推导这些构件。

## 2. 前置知识

### 2.1 线性注意力与 Mamba2

给定长度为 $`L`$ 的序列，每个时间步通过投影产生列向量 $`\boldsymbol q_t,\boldsymbol k_t\in\mathbb R^{d_k}`$ 和 $`\boldsymbol v_t\in\mathbb R^{d_v}`$。以下均针对单个 head 推导；小写粗体表示列向量，大写粗体表示矩阵。

先考虑不带归一化分母的点积型线性注意力：

```math
\mathbf S_t=\mathbf S_{t-1}+\boldsymbol v_t\boldsymbol k_t^\top,
\qquad \boldsymbol o_t=\mathbf S_t\boldsymbol q_t,
\qquad \mathbf S_t\in\mathbb R^{d_v\times d_k}.
```

假设 $`\mathbf S_0=0`$，状态累积前 $`t`$ 步的键值对。这里需要区分两种乘法：

- 外积 $`\boldsymbol v_t\boldsymbol k_t^\top`$：列向量乘行向量，得到 $`d_v\times d_k`$ 矩阵。
- 内积 $`\boldsymbol k_j^\top\boldsymbol q_t`$：行向量乘列向量，得到标量 $`\sum_i k_{j,i}q_{t,i}`$。

上面的递推形式适合逐 token 推理，但训练时需要处理整个序列。为了理解 linear attention 与标准 attention 的关系，以及后续为什么需要 chunkwise 算法，先将递推公式展开为矩阵并行形式。将 $`\mathbf{S}_t = \sum_{j=1}^{t} \boldsymbol{v}_j \boldsymbol{k}_j^{\top}`$ 代入输出公式：

```math
\begin{aligned}
\boldsymbol o_t
&=\mathbf S_t\boldsymbol q_t
 =\left(\sum_{j=1}^{t}\boldsymbol v_j\boldsymbol k_j^\top\right)\boldsymbol q_t\\
&=\sum_{j=1}^{t}\boldsymbol v_j(\boldsymbol k_j^\top\boldsymbol q_t)
 =\sum_{j=1}^{t}(\boldsymbol q_t^\top\boldsymbol k_j)\boldsymbol v_j.
\end{aligned}
```

推导分为两步：

第一步利用矩阵乘法对加法的分配律，将 $`\boldsymbol q_t`$ 移入求和。第二步利用 $`\boldsymbol k_j^\top\boldsymbol q_t`$ 是标量，将其移到 value 向量前；实向量内积还满足 $`\boldsymbol k_j^\top\boldsymbol q_t=\boldsymbol q_t^\top\boldsymbol k_j`$。

最终形式 $`\sum (\boldsymbol{q}_t^{\top} \boldsymbol{k}_j) \boldsymbol{v}_j`$ 在结构上与标准 attention 的加权求和一致，由此可将所有时间步写成矩阵并行形式：

```math
\mathbf{O} = (\mathbf{Q} \mathbf{K}^{\top} \odot \mathbf{M}) \mathbf{V} \in \mathbb{R}^{L \times d_v}
```

其中 $`\mathbf M`$ 是因果掩码：$`\mathbf M_{ij}=1`$ 当 $`i\geq j`$，否则为 0。标准 Attention 对 $`\mathbf Q\mathbf K^\top/\sqrt{d_k}`$ 做 softmax；这里直接使用点积权重。更一般的线性注意力可以引入特征映射与归一化状态。

这个并行表达式仍需 $`O(L^2d)`$ 的计算，因为因果掩码作用于 token 两两关系，不能直接改写成 $`\mathbf Q(\mathbf K^\top\mathbf V)`$。递推形式为 $`O(Ld^2)`$，却受时间方向的串行依赖限制。Chunk-wise 算法正是为了兼顾线性复杂度与块内并行。

状态 $`\mathbf S_t\in\mathbb R^{d_v\times d_k}`$ 将历史压缩到固定大小的矩阵中，非正交 key 的写入可能相互干扰；这种干扰没有一个等于 $`d_kd_v`$ 的固定 token 阈值。为控制历史信息的保留时间，先引入 Mamba2 式的标量衰减：

```math
\mathbf{S}_t = \alpha_t \mathbf{S}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^{\top}, \quad \boldsymbol{o}_t = \mathbf{S}_t \boldsymbol{q}_t
```

其中 $`\alpha_t \in (0, 1)`$ 是数据相关的标量衰减。下面继续展开递推形式，先展开 $`\mathbf{S}_t`$，反复代入递推公式：

```math
\begin{aligned}
\mathbf S_t
&=\alpha_t\mathbf S_{t-1}+\boldsymbol v_t\boldsymbol k_t^\top\\
&=\alpha_t\alpha_{t-1}\mathbf S_{t-2}
 +\alpha_t\boldsymbol v_{t-1}\boldsymbol k_{t-1}^\top
 +\boldsymbol v_t\boldsymbol k_t^\top\\
&=\sum_{j=1}^{t}\left(\prod_{\tau=j+1}^{t}\alpha_\tau\right)
 \boldsymbol v_j\boldsymbol k_j^\top.
\end{aligned}
```

定义累积衰减乘积 $`\gamma_j = \prod_{i=1}^{j} \alpha_i`$，则从位置 j 到位置 t 的衰减系数为：

```math
\prod_{\tau=j+1}^{t} \alpha_\tau = \frac{\gamma_t}{\gamma_j}
```

将累计乘积在位置 j 处分开，有

```math
\begin{aligned}
\gamma_t
 &=\alpha_1\cdots\alpha_j\,\alpha_{j+1}\cdots\alpha_t\\
 &=\gamma_j\prod_{\tau=j+1}^{t}\alpha_\tau.
\end{aligned}
```

两边除以 $`\gamma_j`$ 后代入：

```math
\mathbf{S}_t = \sum_{j=1}^{t} \frac{\gamma_t}{\gamma_j} \boldsymbol{v}_j \boldsymbol{k}_j^{\top}
```

再代入 $`\boldsymbol{o}_t = \mathbf{S}_t \boldsymbol{q}_t`$ （与前文无衰减版本的推导相同，将 $`\boldsymbol{q}_t`$ 分配进求和并交换标量位置）：

```math
\boldsymbol{o}_t = \sum_{j=1}^{t} \frac{\gamma_t}{\gamma_j} (\boldsymbol{q}_t^{\top} \boldsymbol{k}_j) \boldsymbol{v}_j
```

写成矩阵形式， $`\mathbf{O}`$ 的第 t 行就是上式对所有 $`j \leq t`$ 的求和，即：

```math
\mathbf{O} = ((\mathbf{Q} \mathbf{K}^{\top}) \odot \Gamma) \mathbf{V}
```

其中 $`\Gamma_{ij} = \frac{\gamma_i}{\gamma_j}`$ （当 $`i \geq j`$，否则为 0）是衰减感知的因果掩码。与无衰减的 linear attention 对比，区别仅在于因果掩码从 $`\mathbf{M}`$ （0/1 掩码）变成了 $`\Gamma`$ （带衰减权重的掩码）。

#### 递推形式的局限

直接以逐 token 递推进行训练，主要面临以下限制：

- **状态存储**：直接使用 PyTorch autograd 展开递推，通常需要保存多个时间步的状态及中间量，开销随序列长度增长。手写反向可以通过重计算减少保存量。
- **计算形态**：外积 $`\boldsymbol v_t\boldsymbol k_t^\top`$ 是 $`(d_v,1)\times(1,d_k)`$ 的乘法，归约维度只有 1。当前 recurrent kernel 以逐元素运算和归约实现，不能像块矩阵乘法那样利用 Tensor Core 吞吐。
- **串行依赖**：$`\mathbf S_t`$ 依赖 $`\mathbf S_{t-1}`$，时间维度不能直接并行。

#### 分块并行训练

Chunk-wise 算法将输入分成大小为 $`C`$ 的块。每个块依赖前一个块的最终状态，并在块内并行处理 Q/K/V。在精确算术下，它与逐 token 递推等价；$`C=1`$ 退化为 recurrent 形式，$`C=L`$ 对应整个序列的并行表达。实际浮点计算中，不同分块会改变归约顺序，因而不保证逐位相同。

对本节的门控线性注意力，计算复杂度为 $`O(Ld^2+LCd)`$，固定 $`C`$ 后随 $`L`$ 线性增长。后续 GDN 的下三角求解还带来额外的块内计算。当前 FLA GDN 接口支持 `chunk_size=16/32/64`，默认使用 64。

下面推导分块公式。记号约定：下标 [t] 表示第 t 个 chunk，上标 r 表示 chunk 内的第 r 个位置（$`r = 1, \ldots, C`$）， $`\mathbf{S}_{[t]}`$ 是第 t 个 chunk 开始时的隐状态， $`\gamma^r = \prod_{i=1}^{r} \alpha^i`$ 是 chunk 内从位置 1 到位置 r 的累积衰减（为简洁，省略 chunk 下标 [t] 和全局偏移）。

隐状态 S 递推：从单步递推公式 $`\mathbf{S}_r = \alpha_r \mathbf{S}_{r-1} + \boldsymbol{v}_r \boldsymbol{k}_r^{\top}`$ 出发，将 chunk 内所有位置展开：

```math
\begin{aligned}
\mathbf S_{[t+1]}=\mathbf S_C
&=\alpha_C\mathbf S_{C-1}+\boldsymbol v_C\boldsymbol k_C^\top\\
&=\left(\prod_{i=1}^{C}\alpha_i\right)\mathbf S_{[t]}
 +\sum_{r=1}^{C}\left(\prod_{i=r+1}^{C}\alpha_i\right)
 \boldsymbol v_r\boldsymbol k_r^\top.
\end{aligned}
```

利用 $`\gamma^C = \prod_{i=1}^{C} \alpha_i`$ 和 $`\prod_{i=r+1}^{C} \alpha_i = \gamma^C / \gamma^r`$：

```math
\mathbf{S}_{[t+1]} = \gamma^C \mathbf{S}_{[t]} + \sum_{r=1}^{C} \frac{\gamma^C}{\gamma^r} \boldsymbol{v}_r \boldsymbol{k}_r^{\top}
```

将 $`\gamma^C / \gamma^r`$ 吸收到 $`\boldsymbol{k}_r`$ 中得到 $`\overrightarrow{\boldsymbol{k}_r} = (\gamma^C / \gamma^r) \boldsymbol{k}_r`$，写成矩阵形式：

```math
\mathbf{S}_{[t+1]} = \overrightarrow{\mathbf{S}_{[t]}} + \mathbf{V}_{[t]}^{\top} \overrightarrow{\mathbf{K}_{[t]}}
```

其中 $`\overrightarrow{\mathbf{S}_{[t]}} = \gamma^C \mathbf{S}_{[t]}`$。这里 $`\mathbf{V}_{[t]}^{\top} \overrightarrow{\mathbf{K}_{[t]}}`$ 就是 chunk 内所有位置的外积之和（$`\sum_r \boldsymbol{v}_r \overrightarrow{\boldsymbol{k}_r}^{\top}`$），其中 $`\mathbf{V}_{[t]} \in \mathbb{R}^{C \times d_v}`$， $`\overrightarrow{\mathbf{K}_{[t]}} \in \mathbb{R}^{C \times d_k}`$。

输出 O 计算：类似地，chunk 内位置 r 的输出 $`\boldsymbol{o}_r = \mathbf{S}_r \boldsymbol{q}_r`$，将 $`\mathbf{S}_r`$ 展开后拆为两部分：

```math
\boldsymbol{o}_r = \gamma^r \mathbf{S}_{[t]} \boldsymbol{q}_r + \sum_{j=1}^{r} \frac{\gamma^r}{\gamma^j} (\boldsymbol{q}_r^{\top} \boldsymbol{k}_j) \boldsymbol{v}_j
```

前半部分是 inter-chunk，后半部分是 intra-chunk。

inter-chunk：将 $`\gamma^r`$ 吸收到 $`\boldsymbol{q}_r`$ 中得到 $`\overleftarrow{\boldsymbol{q}_r} = \gamma^r \boldsymbol{q}_r`$，矩阵形式为 $`\overleftarrow{\mathbf{Q}_{[t]}} \mathbf{S}_{[t]}^{\top}`$

intra-chunk： $`\gamma^r / \gamma^j`$ 正好是 $`\Gamma_{[t]}`$ 矩阵的第 (r, j) 个元素，矩阵形式为 $`(\mathbf{Q}_{[t]} \mathbf{K}_{[t]}^{\top} \odot \Gamma_{[t]}) \mathbf{V}_{[t]}`$

合并：

```math
\mathbf{O}_{[t]} = \overleftarrow{\mathbf{Q}_{[t]}} \mathbf{S}_{[t]}^{\top} + (\mathbf{Q}_{[t]} \mathbf{K}_{[t]}^{\top} \odot \Gamma_{[t]}) \mathbf{V}_{[t]}
```

以上就是 chunkwise 算法的两个核心公式：隐状态 S 递推和输出 O 计算。整个算法按 chunk 顺序（$`t = 0, 1, 2, \ldots`$）执行，每个 chunk 做两件事：

隐状态 S 递推： $`\mathbf{S}_{[t+1]} = \overrightarrow{\mathbf{S}_{[t]}} + \mathbf{V}_{[t]}^{\top} \overrightarrow{\mathbf{K}_{[t]}}`$。chunk 间串行（$`\mathbf{S}_{[t+1]}`$ 依赖 $`\mathbf{S}_{[t]}`$），但串行步数只有 L/C 步而非 L 步。涉及 $`d_v \times d_k`$ 的矩阵运算，单步复杂度 O(Cd^2)。

输出 O 计算： $`\mathbf{O}_{[t]} = \overleftarrow{\mathbf{Q}_{[t]}} \mathbf{S}_{[t]}^{\top} + (\mathbf{Q}_{[t]} \mathbf{K}_{[t]}^{\top} \odot \Gamma_{[t]}) \mathbf{V}_{[t]}`$。chunk 内完全并行。inter-chunk 部分（$`\overleftarrow{\mathbf{Q}} \mathbf{S}^{\top}`$）是矩阵乘法 O(Cd^2)；intra-chunk 部分（$`(\mathbf{Q}\mathbf{K}^{\top} \odot \Gamma) \mathbf{V}`$）是 $`C \times C`$ 的注意力矩阵乘法 O(C^2 d)，可以充分利用 Tensor Core。

因此 chunkwise 算法在训练效率和线性复杂度之间取得了平衡。后文第 5 节的代码解析就是对这两个公式（以及 GDN 扩展后的版本）的逐步实现。

上面的推导中出现了箭头符号，这里展开解释。箭头表示将衰减系数“吸收”到向量中，方向代表衰减的参考点：

左箭头$`\overleftarrow{\cdot}`$（向 chunk 首位衰减）：位置 r 的向量乘以从 chunk 开头到位置 r 的累积衰减 $`\gamma_{[t]}^r`$。位置越靠后（r 越大）， $`\gamma_{[t]}^r`$ 越小（因为每多乘一个 $`\alpha < 1`$），即越远离 chunk 开头的位置衰减越多。一般用于 q，因为 q 需要与 chunk 之前的历史状态 $`\mathbf{S}_{[t]}`$ 交互，距离越远衰减越大，即 $`\overleftarrow{\boldsymbol{q}_{[t]}^r} = \gamma_{[t]}^r \boldsymbol{q}_{[t]}^r`$。

右箭头$`\overrightarrow{\cdot}`$（向 chunk 末位衰减）：位置 r 的向量乘以从位置 r 到 chunk 末尾的累积衰减 $`\gamma_{[t]}^C / \gamma_{[t]}^r`$。位置越靠前（r 越小），衰减越多，因为它距离 chunk 末尾越远。一般用于 k，因为 k 的贡献要传递到 chunk 末尾的隐状态 $`\mathbf{S}_{[t+1]}`$，越早的 k 经历越多衰减，即 $`\overrightarrow{\boldsymbol{k}_{[t]}^r} = \frac{\gamma_{[t]}^C}{\gamma_{[t]}^r} \boldsymbol{k}_{[t]}^r, \quad \overrightarrow{\mathbf{S}_{[t]}} = \gamma_{[t]}^C \mathbf{S}_{[t]}`$。其中$`\overrightarrow{\mathbf{S}_{[t]}}`$ 是将上一个 chunk 末尾的隐状态乘以当前 chunk 的整体衰减 $`\gamma_{[t]}^C = \prod_{i=1}^{C} \alpha_{[t]}^i`$，表示经过整个 chunk 后历史信息的衰减。

下面从直觉上来理解，假设 chunk 有 4 个位置，衰减率均为 $`\alpha = 0.9`$：

| 块内位置 | 累计衰减 $`\gamma^r`$ | Q 的衰减 $`\gamma^r`$ | K 的块末衰减 $`\gamma^C/\gamma^r`$ |
| --- | --- | --- | --- |
| 1 | 0.9 | 0.9 | 0.729 |
| 2 | 0.81 | 0.81 | 0.81 |
| 3 | 0.729 | 0.729 | 0.9 |
| 4 | 0.6561 | 0.6561 | 1.0 |

可以看到： $`\overleftarrow{q}`$ 的位置 4 衰减最多（距 chunk 开头最远）， $`\overrightarrow{k}`$ 的位置 1 衰减最多（距 chunk 末尾最远）。

### 2.2 DeltaNet: 带 Delta 规则的线性注意力

上一节用标量 $`\alpha_t`$ 均匀衰减整个状态。DeltaNet 则沿当前 key 的方向修改状态，并保留与该 key 正交的方向。非正交的其他 key 仍可能受到影响，因此需要区分方向性修正与互不干扰的字典条目替换。

其更新规则为：

```math
\mathbf{S}_t = \mathbf{S}_{t-1} (\mathbf{I} - \beta_t \boldsymbol{k}_t \boldsymbol{k}_t^{\top}) + \beta_t \boldsymbol{v}_t \boldsymbol{k}_t^{\top}
```

将状态视为线性关联存储器，$`\mathbf S_{t-1}\boldsymbol k_t`$ 是以当前 key 读出的旧 value。更新先减去 $`\beta_t(\mathbf S_{t-1}\boldsymbol k_t)\boldsymbol k_t^\top`$，再加上 $`\beta_t\boldsymbol v_t\boldsymbol k_t^\top`$。标量衰减会均匀缩放整个状态；Delta Rule 则沿 $`\boldsymbol k_t`$ 的方向修正预测，并保持其正交方向不变。

从优化的角度看，这等价于对损失 $`\mathcal{L}(\mathbf{S}) = \frac{1}{2} \|\mathbf{S} \boldsymbol{k}_t - \boldsymbol{v}_t\|^2`$ 做一步梯度下降（$`\beta_t`$ 就是学习率），具体推导见 3.2 节。

考虑一个理想化例子：假设“猫”“狗”“鸟”对应两两正交的单位 key，当前状态分别记录“在窗台上”“在院子里”“在树上”。新输入要求把“猫”对应的 value 更新为“坐在垫子上”。

**Gate：全局衰减，$`\alpha_t=0.9`$。**

| 条目 | 更新前 | 更新后 |
| --- | --- | --- |
| 猫 | 在窗台上 | 0.9 x 在窗台上 + 坐在垫子上 |
| 狗 | 在院子里 | 0.9 x 在院子里 |
| 鸟 | 在树上 | 0.9 x 在树上 |

所有条目都被乘以 0.9 均匀缩小，猫的旧值没有被清除，只是变小了，将新的值叠加上去。

**Delta Rule：定向更新，$`\beta_t=1`$。**

| 条目 | 更新前 | 更新后 |
| --- | --- | --- |
| 猫 | 在窗台上 | 坐在垫子上（旧值被擦除，写入新值） |
| 狗 | 在院子里 | 在院子里（不受影响） |
| 鸟 | 在树上 | 在树上（不受影响） |

在单位 key 且其他 key 与其正交的假设下，“猫”的旧值被替换，“狗”和“鸟”的读出保持不变。$`\beta_t`$ 控制更新幅度：$`\beta_t=1`$ 完全替换当前方向的读出，$`\beta_t=0.5`$ 则在新旧读出之间插值。

#### DeltaNet 的 WY 表示

DeltaNet 的分块算法需要计算块内转移矩阵的有序乘积 $`\prod_{i=1}^{r}(\mathbf I-\beta_i\boldsymbol k_i\boldsymbol k_i^\top)`$。当 key 为单位向量且 $`\beta_i=2`$ 时，单个因子是 Householder 反射；对一般 $`\beta_i`$，它是相同形式的秩一修正。

这类矩阵乘积可以写成紧凑的 WY 表示。下面给出归纳证明；经典来源为 Bischof 与 Van Loan 的 [The WY Representation for Products of Householder Matrices（1987）](https://epubs.siam.org/doi/abs/10.1137/0908009)。

定义 $`\mathbf{P}_n = \prod_{t=1}^n(\mathbf{I} - \beta_t \boldsymbol{k}_t \boldsymbol{k}_t^{\top})`$，要证明 $`\mathbf{P}_n = \mathbf{I} - \sum_{i=1}^n \boldsymbol{w}_i \boldsymbol{k}_i^{\top}`$。

基础情况 n=1： $`\mathbf{P}_1 = \mathbf{I} - \beta_1 \boldsymbol{k}_1 \boldsymbol{k}_1^{\top}`$，令 $`\boldsymbol{w}_1 = \beta_1 \boldsymbol{k}_1`$ 即满足。

归纳步骤：假设 n-1 时成立，则对 n：

```math
\begin{aligned}
\mathbf P_n
&=\mathbf P_{n-1}(\mathbf I-\beta_n\boldsymbol k_n\boldsymbol k_n^\top)\\
&=\left(\mathbf I-\sum_{t=1}^{n-1}\boldsymbol w_t\boldsymbol k_t^\top\right)
 (\mathbf I-\beta_n\boldsymbol k_n\boldsymbol k_n^\top)\\
&=\mathbf I-\sum_{t=1}^{n-1}\boldsymbol w_t\boldsymbol k_t^\top
 -\beta_n\boldsymbol k_n\boldsymbol k_n^\top
 +\beta_n\sum_{t=1}^{n-1}\boldsymbol w_t
 (\boldsymbol k_t^\top\boldsymbol k_n)\boldsymbol k_n^\top\\
&=\mathbf I-\sum_{t=1}^{n}\boldsymbol w_t\boldsymbol k_t^\top.
\end{aligned}
```

其中 $`\boldsymbol{w}_n = \beta_n \boldsymbol{k}_n - \beta_n \sum_{t=1}^{n-1} \boldsymbol{w}_t (\boldsymbol{k}_t^{\top} \boldsymbol{k}_n)`$。

从第三步到第四步的关键在于： $`\sum_{t=1}^{n-1} \boldsymbol{w}_t (\boldsymbol{k}_t^{\top} \boldsymbol{k}_n)`$ 是标量乘向量的求和，可以合并到 $`\boldsymbol{w}_n`$ 中。这个证明不仅确认了表示的正确性，还给出了 $`\boldsymbol{w}_n`$ 的递推计算公式。

当初始状态为零时，状态也可写成 $`\mathbf S_n=\sum_{t=1}^{n}\boldsymbol u_t\boldsymbol k_t^\top`$。$`\boldsymbol u`$ 的递推与 $`\boldsymbol w`$ 具有相同结构，只需将初始项 $`\beta_n\boldsymbol k_n`$ 换成 $`\beta_n\boldsymbol v_n`$：

```math
\boldsymbol{u}_n = \beta_n \left(\boldsymbol{v}_n - \sum_{t=1}^{n-1} \boldsymbol{u}_t (\boldsymbol{k}_t^{\top} \boldsymbol{k}_n)\right)
```

这两个递推的共同结构意味着它们可以共享同一个求解过程，即下文的 UT 变换。

写成矩阵形式：

```math
\mathbf{T}_{[t]} = \left[\mathbf{I} + \text{strictLower}\left(\text{diag}(\beta_{[t]}) \mathbf{K}_{[t]} \mathbf{K}_{[t]}^{\top}\right)\right]^{-1} \text{diag}(\beta_{[t]})
```

```math
\mathbf{W}_{[t]} = \mathbf{T}_{[t]} \mathbf{K}_{[t]}, \quad \mathbf{U}_{[t]} = \mathbf{T}_{[t]} \mathbf{V}_{[t]}
```

从图论角度看，$`\boldsymbol w`$ 和 $`\boldsymbol u`$ 的递推对应一个有向无环图：从位置 $`i`$ 到位置 $`r`$（$`i<r`$）的边权为 $`-\beta_r\boldsymbol k_i^\top\boldsymbol k_r`$。令 $`\mathbf L=\operatorname{strictLower}(\operatorname{diag}(\beta)\mathbf K\mathbf K^\top)`$，则边权矩阵是 $`-\mathbf L`$，路径权重之和为 $`(\mathbf I+\mathbf L)^{-1}`$。

由于 $`\mathbf L`$ 严格下三角，$`\mathbf L^C=0`$，逆矩阵也可表示为有限级数 $`\mathbf I-\mathbf L+\mathbf L^2-\cdots`$。实现中利用单位下三角结构做前代和块矩阵运算；显式构造完整逆矩阵仍包含三次阶的块内算术量，效率来自结构与硬件映射，而不是消除这部分计算。

最终 DeltaNet 的分块算法为：

隐状态 S 递推：

```math
\mathbf{S}_{[t+1]} = \mathbf{S}_{[t]} + (\mathbf{U}_{[t]} - \mathbf{W}_{[t]} \mathbf{S}_{[t]}^{\top})^{\top} \mathbf{K}_{[t]}
```

输出 O 计算：

```math
\mathbf{O}_{[t]} = \mathbf{Q}_{[t]} \mathbf{S}_{[t]}^{\top} + (\mathbf{Q}_{[t]} \mathbf{K}_{[t]}^{\top} \odot \mathbf{M})(\mathbf{U}_{[t]} - \mathbf{W}_{[t]} \mathbf{S}_{[t]}^{\top})
```

## 3. Gated Delta Rule

### 3.1 公式定义

Gated Delta Rule 将门控衰减和 delta 更新都放在一个公式里：

```math
\mathbf{S}_t = \mathbf{S}_{t-1} \left( \alpha_t (\mathbf{I} - \beta_t \boldsymbol{k}_t \boldsymbol{k}_t^{\top}) \right) + \beta_t \boldsymbol{v}_t \boldsymbol{k}_t^{\top}
```

展开后可以分解为三个操作：

```math
\mathbf{S}_t = \alpha_t \mathbf{S}_{t-1} - \alpha_t \beta_t (\mathbf{S}_{t-1} \boldsymbol{k}_t) \boldsymbol{k}_t^{\top} + \beta_t \boldsymbol{v}_t \boldsymbol{k}_t^{\top}
```

即：全局衰减、擦除旧值、写入新值。

取 $`d_v=d_k=2`$，通过一个数值例子分别观察衰减、擦除和写入的作用。令旧状态为：

```math
\mathbf{S}_{t-1} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix}
```

矩阵的列对应 key 空间，行对应 value 空间。因此，$`\mathbf S`$ 的第 j 列可理解为 key 空间第 j 个基方向所关联的 value 向量。

当前时间步的参数： $`\boldsymbol{k}_t = [1, 0]^{\top}`$， $`\boldsymbol{v}_t = [0.7, 0.3]^{\top}`$， $`\alpha_t = 0.9`$， $`\beta_t = 1.0`$ （完全替换）。其中 $`[1, 0]^{\top}`$ 是列向量的简写。

$`\boldsymbol k_t=[1,0]^\top`$ 选择 key 空间的第一个基方向。$`\mathbf S\boldsymbol k_t`$ 读出状态的第一列，外积 $`\boldsymbol v\boldsymbol k_t^\top`$ 则只修改第一列。三个操作分别为：

第一步：全局衰减 $`\alpha_t \mathbf{S}_{t-1}`$

```math
0.9 \times \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 0.9 & 0 \\ 0 & 0.9 \end{bmatrix}
```

第二步：擦除旧值 $`-\alpha_t \beta_t (\mathbf{S}_{t-1} \boldsymbol{k}_t) \boldsymbol{k}_t^{\top}`$

```math
-0.9 \times 1.0 \times \begin{bmatrix} 1 \\ 0 \end{bmatrix} [1 \; 0] = -\begin{bmatrix} 0.9 & 0 \\ 0 & 0 \end{bmatrix}
```

第三步：写入新值 $`\beta_t \boldsymbol{v}_t \boldsymbol{k}_t^{\top}`$

```math
1.0 \times \begin{bmatrix} 0.7 \\ 0.3 \end{bmatrix} [1 \; 0] = \begin{bmatrix} 0.7 & 0 \\ 0.3 & 0 \end{bmatrix}
```

最终结果：三者相加

```math
\mathbf{S}_t = \begin{bmatrix} 0.9 & 0 \\ 0 & 0.9 \end{bmatrix} - \begin{bmatrix} 0.9 & 0 \\ 0 & 0 \end{bmatrix} + \begin{bmatrix} 0.7 & 0 \\ 0.3 & 0 \end{bmatrix} = \begin{bmatrix} 0.7 & 0 \\ 0.3 & 0.9 \end{bmatrix}
```

第一列由 $`[1,0]^\top`$ 替换为 $`[0.7,0.3]^\top`$；第二列只做全局衰减，由 $`[0,1]^\top`$ 变为 $`[0,0.9]^\top`$。若 $`\alpha=1`$，第二列保持不变；若 $`\beta=0`$，则只执行全局衰减。

### 3.2 直觉理解

#### 在线学习视角

在线学习关注数据逐个到来时的增量更新。将隐状态 $`\mathbf S_t`$ 视为模型参数，每个 token 提供一对 key/value，可以将下列更新写成对应目标的闭式最小解。目标中的 $`\mathbf S_{t-1}`$、key、value 和 gate 均视为已知量。

**线性注意力**

目标：

```math
\|\mathbf{S}_t - \mathbf{S}_{t-1}\|_F^2 - 2\langle \mathbf{S}_t \boldsymbol{k}_t, \boldsymbol{v}_t \rangle
```

更新：

```math
\mathbf{S}_t = \mathbf{S}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^{\top}
```

**Mamba2**

目标：

```math
\|\mathbf{S}_t - \alpha_t \mathbf{S}_{t-1}\|_F^2 - 2\langle \mathbf{S}_t \boldsymbol{k}_t, \boldsymbol{v}_t \rangle
```

更新：

```math
\mathbf{S}_t = \alpha_t \mathbf{S}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^{\top}
```

**DeltaNet**

目标：

```math
\|\mathbf{S}_t - \mathbf{S}_{t-1}\|_F^2 - 2\langle \mathbf{S}_t \boldsymbol{k}_t, \beta_t(\boldsymbol{v}_t - \mathbf{S}_{t-1} \boldsymbol{k}_t) \rangle
```

更新：

```math
\mathbf{S}_t = \mathbf{S}_{t-1}(\mathbf{I} - \beta_t \boldsymbol{k}_t \boldsymbol{k}_t^{\top}) + \beta_t \boldsymbol{v}_t \boldsymbol{k}_t^{\top}
```

**Gated DeltaNet**

目标：

```math
\|\mathbf{S}_t - \alpha_t \mathbf{S}_{t-1}\|_F^2 - 2\langle \mathbf{S}_t \boldsymbol{k}_t, \beta_t(\boldsymbol{v}_t - \alpha_t \mathbf{S}_{t-1} \boldsymbol{k}_t) \rangle
```

更新：

```math
\mathbf{S}_t = \mathbf{S}_{t-1}(\alpha_t(\mathbf{I} - \beta_t \boldsymbol{k}_t \boldsymbol{k}_t^{\top})) + \beta_t \boldsymbol{v}_t \boldsymbol{k}_t^{\top}
```

目标函数的第一项 $`\|\mathbf{S}_t - \cdot\|_F^2`$ 是正则项，约束新状态不要偏离“参考点”太远（线性注意力的参考点是 $`\mathbf{S}_{t-1}`$，Mamba2/GDN 的参考点是 $`\alpha_t \mathbf{S}_{t-1}`$）。第二项是关联项，鼓励 $`\mathbf{S}_t`$ 在 $`\boldsymbol{k}_t`$ 方向上编码新信息 $`\boldsymbol{v}_t`$。

#### TTT (Test-Time Training) 视角

从 Test-Time Training（TTT）视角看，隐状态是一个随输入在线更新的线性模型。Delta Rule 的目标是使该模型以 key 预测 value，即 $`\mathbf S\boldsymbol k_t\approx\boldsymbol v_t`$。

对 $`\mathcal L(\mathbf S)=\frac12\|\mathbf S\boldsymbol k_t-\boldsymbol v_t\|^2`$ 求梯度，并在旧状态处执行一步梯度下降：

```math
\mathbf{S}_t = \mathbf{S}_{t-1} - \beta_t \nabla_{\mathbf{S}} \mathcal{L} = \mathbf{S}_{t-1} - \beta_t (\mathbf{S}_{t-1} \boldsymbol{k}_t - \boldsymbol{v}_t) \boldsymbol{k}_t^{\top}
```

这正是 DeltaNet 的更新，$`\beta_t`$ 对应步长。GDN 则先将参考状态衰减为 $`\alpha_t\mathbf S_{t-1}`$，再在该状态处计算梯度。

在线优化的闭式目标和 TTT 的梯度更新是两种描述方式：前者给出新状态最小化的目标，后者展示更新的计算过程。

### 3.3 硬件高效的分块并行训练算法

#### 展开递推

对 chunk [t] 内的递推进行部分展开：

```math
\mathbf{S}_{[t]}^r = \mathbf{S}_{[t]} \mathbf{F}_{[t]}^r + \mathbf{G}_{[t]}^r
```

其中，按时间顺序相乘的状态转移为：

```math
\begin{aligned}
\mathbf F_{[t]}^r
&=\prod_{i=1}^{r}\alpha_{[t]}^i
 (\mathbf I-\beta_{[t]}^i\boldsymbol k_{[t]}^i\boldsymbol k_{[t]}^{i\top})
 =\gamma_{[t]}^r\mathbf P_{[t]}^r,\\
\mathbf G_{[t]}^r
&=\sum_{i=1}^{r}\beta_{[t]}^i\boldsymbol v_{[t]}^i\boldsymbol k_{[t]}^{i\top}
 \prod_{j=i+1}^{r}\alpha_{[t]}^j
 (\mathbf I-\beta_{[t]}^j\boldsymbol k_{[t]}^j\boldsymbol k_{[t]}^{j\top}).
\end{aligned}
```

$`\mathbf F_{[t]}^r`$ 描述块入口状态如何传播，$`\mathbf G_{[t]}^r`$ 描述块内新写入的累积贡献。标量衰减可以从矩阵乘积中提出，因此 $`\mathbf F_{[t]}^r=\overleftarrow{\mathbf P_{[t]}^r}`$。

#### 扩展的 WY 表示

对 $`\mathbf{G}_{[t]}^r`$，通过引入衰减修改 DeltaNet 的 WY 表示（详细证明见论文附录 A）：

```math
\mathbf{G}_{[t]}^r = \sum_{i=1}^{r} \frac{\gamma_{[t]}^r}{\gamma_{[t]}^i} \tilde{\boldsymbol{u}}_{[t]}^i \boldsymbol{k}_{[t]}^{i\top} \in \mathbb{R}^{d_v \times d_k}
```

```math
\tilde{\boldsymbol{u}}_{[t]}^r = \beta_{[t]}^r \left(\boldsymbol{v}_{[t]}^r - \sum_{i=1}^{r-1} \left(\tilde{\boldsymbol{u}}_{[t]}^i (\frac{\gamma_{[t]}^r}{\gamma_{[t]}^i} \boldsymbol{k}_{[t]}^{i\top} \boldsymbol{k}_{[t]}^r)\right)\right) \in \mathbb{R}^{d_v}
```

通过 UT 变换写成矩阵形式：

```math
\begin{aligned}
\mathbf L_{[t]}^{(g)}
 &=\operatorname{strictLower}\!\left(
  \operatorname{diag}(\beta_{[t]})
  (\Gamma_{[t]}\odot\mathbf K_{[t]}\mathbf K_{[t]}^\top)\right),\\
\widetilde{\mathbf T}_{[t]}
 &=(\mathbf I+\mathbf L_{[t]}^{(g)})^{-1}\operatorname{diag}(\beta_{[t]}),\\
\widetilde{\mathbf U}_{[t]}
 &=\widetilde{\mathbf T}_{[t]}\mathbf V_{[t]}
 \in\mathbb R^{C\times d_v}.
\end{aligned}
```

相较于 DeltaNet，这里将 Gram 矩阵 $`\mathbf K\mathbf K^\top`$ 替换为 $`\Gamma\odot\mathbf K\mathbf K^\top`$，从而纳入块内写入之间的衰减。代码也先计算 KKT，再按行乘 beta、按元素乘衰减矩阵。

沿用第 2.2 节未加 gate 的 $`\mathbf W=\mathbf T\mathbf K`$，令 $`\mathbf D_\gamma=\operatorname{diag}(\gamma)`$，有

```math
\overleftarrow{\mathbf W}
=\mathbf D_\gamma\mathbf W
=\widetilde{\mathbf T}\mathbf D_\gamma\mathbf K.
```

这是因为 $`\mathbf L^{(g)}=\mathbf D_\gamma\mathbf L\mathbf D_\gamma^{-1}`$。它说明“对 W 按行乘累计衰减”与代码的“先衰减 K，再应用带 gate 的 UT 变换”是同一表达。

#### 最终的分块算法

类似于上面扩展线性注意力的方式，将 DeltaNet 的分块算法扩展为 Gated DeltaNet：

```math
\mathbf{S}_{[t+1]} = \overrightarrow{\mathbf{S}_{[t]}} + \left(\widetilde{\mathbf{U}}_{[t]} - \overleftarrow{\mathbf{W}_{[t]}} \mathbf{S}_{[t]}^{\top}\right)^{\top} \overrightarrow{\mathbf{K}_{[t]}} \in \mathbb{R}^{d_v \times d_k}
```

```math
\mathbf O_{[t]}
=\overleftarrow{\mathbf Q_{[t]}}\mathbf S_{[t]}^\top
 +(\mathbf Q_{[t]}\mathbf K_{[t]}^\top\odot\Gamma_{[t]})
 \left(\widetilde{\mathbf U}_{[t]}-
 \overleftarrow{\mathbf W_{[t]}}\mathbf S_{[t]}^\top\right)
 \in\mathbb R^{C\times d_v}.
```

输出的块内项使用带衰减的因果矩阵 $`\Gamma`$，其对角线为 1；构造 UT 系统时则取严格下三角。前者允许输出读取当前 token 的写入，后者表示当前写入误差只依赖更早的写入。

其中：

```math
\begin{aligned}
\overleftarrow{\boldsymbol q_{[t]}^r}&=\gamma_{[t]}^r\boldsymbol q_{[t]}^r,
&\quad\overleftarrow{\boldsymbol w_{[t]}^r}&=\gamma_{[t]}^r\boldsymbol w_{[t]}^r,\\
\overrightarrow{\boldsymbol k_{[t]}^r}&=\frac{\gamma_{[t]}^C}{\gamma_{[t]}^r}\boldsymbol k_{[t]}^r,
&\quad\overrightarrow{\mathbf S_{[t]}}&=\gamma_{[t]}^C\mathbf S_{[t]}.
\end{aligned}
```

分块形式将主要计算转化为矩阵乘法，适于使用 Tensor Core 执行。

## 4. 模型调用方式：以 Qwen3.5 GatedDeltaNet 为例

下面以 Transformers 的 `Qwen3_5MoeGatedDeltaNet` 为例，说明 GDN 在模型中的调用方式。对应文件为 `models/qwen3_5_moe/modeling_qwen3_5_moe.py`。

### 4.1 整体架构

Qwen3.5 采用混合架构，交替使用 Gated DeltaNet 与带输出门控的 full attention。GDN 通过固定大小状态处理历史，full attention 保留显式的 token 间注意力计算；下面展开 GDN 层。

### 4.2 初始化与参数定义

```python
class Qwen3_5MoeGatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3_5MoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        # Lower bound kept away from 0 so log(A) never becomes -inf
        A = torch.empty(self.num_v_heads).uniform_(0.01, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = Qwen3_5MoeRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.layer_type = config.layer_types[layer_idx]

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
```

### 4.3 前向传播流程

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    cache_params: Cache | None = None,
    attention_mask: torch.Tensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
):
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

    # Set up dimensions for reshapes later
    batch_size, seq_len, _ = hidden_states.shape
    use_precomputed_states = cache_params is not None and cache_params.has_previous_state(
        self.layer_idx, state_idx=0
    )

    mixed_qkv = self.in_proj_qkv(hidden_states)
    mixed_qkv = mixed_qkv.transpose(1, 2)

    z = self.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    if use_precomputed_states and seq_len == 1 and not cache_params.layers[self.layer_idx].record_past:
        conv_state = cache_params.layers[self.layer_idx].conv_states[0]
        # Single-token cached decode: the fused per-step kernel updates the conv state in-place.
        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_state,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.activation,
        )
    else:
        if cache_params is not None:
            mixed_qkv = cache_params.update_conv_state(
                mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size
            )

        mixed_qkv = causal_conv1d_fn(
            mixed_qkv,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            activation=self.activation,
            **kwargs,
        )

        # Drop the additional previous states
        if cache_params is not None:
            mixed_qkv = mixed_qkv[:, :, -seq_len:]

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

    # Update cache
    if cache_params is not None:
        cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

    # reshape input data into 2D tensor
    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    output = self.out_proj(core_attn_out)
    return output
```

**衰减率的参数化。** 代码先计算对数门控，再在 kernel 中恢复衰减系数：

```math
g_t=-\exp(A_{\log})\operatorname{softplus}(a(x_t)+\mathrm{dt\_bias}),
\qquad \alpha_t=\exp(g_t).
```

对数形式将连乘变成求和：$`\log\gamma_i=\sum_{j\leq i}g_j`$。区间衰减由两个累计量的差计算，避免直接对很小的累计乘积做除法。注意单步 gate $`g_i`$ 与其累计和不同；计算 $`\gamma_i/\gamma_j`$ 时，应使用累计和之差。

**输出门控。** `z` 经过独立投影，在 GDN 输出之后参与 $`\operatorname{RMSNorm}(o)\odot\operatorname{SiLU}(z)`$。RMSNorm 作用于 o，SiLU 作用于 z。Q/K 的 L2 归一化由 GDN 算子完成；默认 query 缩放为 $`d_k^{-1/2}`$。

当前初始化中，`A_log = log(uniform(0.01, 16))`，`dt_bias = 1`。作为数值示例，若取 $`A=8`$ 且 $`a(x)=0`$，则 $`g\approx-8\operatorname{softplus}(1)\approx-10.5`$，$`\alpha\approx2.7\times10^{-5}`$。这个例子仅说明参数如何控制衰减；不同 head 的 A 不同，$`a(x)`$ 取决于实际输入与权重，预训练 checkpoint 也使用已学习的参数。

**Short Conv。** `causal_conv1d_fn` 对投影后的 QKV 做逐通道因果卷积，再应用 SiLU；常见配置的卷积窗口为 4。它为 Q/K/V 引入局部上下文，使状态读写可以依赖短程 token 模式。从 TTT 视角看，这相当于丰富在线更新所使用的 key/value 特征，但不能仅由这段代码推出训练目标已从自预测变成下一 token 预测。

**GVA（Grouped Value Attention）。** 当 `num_k_heads < num_v_heads` 时，每组 V head 共享一组 Q/K，组大小为 `num_v_heads // num_k_heads`。例如 Q/K 有 2 个 head、V 有 4 个 head：

```text
V head：  0   1   2   3
Q/K head：0   0   1   1
```

Transformers 的入口使用 `repeat_interleave` 显式扩展 Q/K。FLA kernel 也支持直接通过 head 索引映射读取共享的 Q/K，省去这一步复制；当前代码以 H 表示 Q/K head 数、HV 表示 V head 数：

```python
# 当前 FLA 中，H 是 Q/K head 数，HV 是 V head 数。
B, T, H, K = q.shape
HV = v.shape[2]
V = v.shape[-1]

# kernel 内：按 V head 调度，Q/K 通过整除映射到共享 head。
i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
i_b, i_h = i_bh // HV, i_bh % HV
q += (bos * H + i_h // (HV // H)) * K
k += (bos * H + i_h // (HV // H)) * K
v += (bos * HV + i_h) * V
o += (bos * HV + i_h) * V
```

每 `HV // H` 个 V head 共享一组 Q/K。Q/K 沿时间的 stride 为 `H * K`，V/O 为 `HV * V`。后文的 KKT、W/U、状态更新和输出 kernel 都保留这一 head 映射。

## 5. Chunk-wise 算法代码解析

### 5.1 前向传播总流程

入口位于 `fla/ops/gated_delta_rule/chunk.py`。前向包含六个算法阶段：

1. 计算 chunk 内的累计 gate。
2. 构造带 beta 和衰减的 KKT 严格下三角矩阵。
3. 求解单位下三角系统。
4. 生成 W/U。
5. 沿 chunk 递推状态，并计算 `v_new`。
6. 计算输出。

当前版本将第 2—4 步封装为 `chunk_gated_delta_rule_fwd_intra()`；默认 `BT=64` 时，第 2、3 步进一步融合进 `chunk_gated_delta_rule_fwd_kkt_solve_kernel()`。下面先给出当前入口，再按上述六步解释其计算。

```python
def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    chunk_size: int = 64,
):
    g_input = g if use_gate_in_kernel else None
    if use_gate_in_kernel:
        g = gdn_gate_chunk_cumsum(
            g=g,
            A_log=A_log,
            chunk_size=chunk_size,
            scale=RCP_LN2,
            dt_bias=dt_bias,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
    else:
        g = chunk_local_cumsum(
            g,
            chunk_size=chunk_size,
            scale=RCP_LN2,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
    # obtain WY representation. u is actually the new v.
    # fused kkt + solve_tril + recompute_w_u
    w, u, A = chunk_gated_delta_rule_fwd_intra(
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )

    if cp_context is not None:
        initial_state = chunk_gated_delta_rule_fwd_h_pre_process(
            k=k,
            w=w,
            u=u,
            g=g,
            cu_seqlens=cu_seqlens,
            initial_state=initial_state,
            context=cp_context,
            state_v_first=state_v_first,
            chunk_size=chunk_size,
        )

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
        chunk_size=chunk_size,
    )

    if cp_context is not None:
        initial_state = compress_h0(initial_state, context=cp_context)

    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
        chunk_size=chunk_size,
    )
    return g, o, A, final_state, initial_state, g_input
```

这六个阶段之间的数据流关系如下（以 B=1, T=1024, H=32, K=128, V=128, BT=64 为例标注尺寸）：

```text
输入：q/k/v [1, 1024, 32, 128]；g/beta [1, 1024, 32]；BT=64
  |
  |-- (a) chunk 内 cumsum(g) / ln(2)             [1, 1024, 32]
  |
  |-- (b) A = strictLower(diag(beta) (Gamma ⊙ KK^T))
  |                                                [1, 1024, 32, 64]
  |-- (c) A_inv = (I + A)^(-1)                   [1, 1024, 32, 64]
  |           每个 head 有 16 个 64×64 的局部矩阵
  |
  |-- (d) w = A_inv @ diag(beta) @ diag(gamma) @ K
  |       u = A_inv @ diag(beta) @ V             各 [1, 1024, 32, 128]
  |
  |-- (e) v_new = u - w @ h                      [1, 1024, 32, 128]
  |       h_next = gamma_last * h + K_end^T @ v_new
  |       h 保存各 chunk 的入口状态                [1, 16, 32, 128, 128]
  |
  +-- (f) o = diag(gamma) @ Q @ h
              + ((QK^T) ⊙ Gamma) @ v_new         [1, 1024, 32, 128]
```

图中的 `h` 采用代码默认的 `[K, V]` 布局，即公式中的 $`\mathbf S^\top`$；`K_end` 是乘上块末衰减的 $`\overrightarrow{\mathbf K}`$。图中将 query 的缩放吸收进 Q，实际输出 kernel 在写回前乘 `scale`。

下面逐一解析每个 Kernel。

### 5.2 累积门控求和 chunk_local_cumsum

代码位于 `fla/ops/utils/cumsum.py`，GDN 使用的是标量版本（`chunk_local_cumsum_scalar_kernel`）：

```python
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    if HEAD_FIRST:
        p_s = s + bos*H + i_h*T + o_t
        p_o = o + bos*H + i_h*T + o_t
    else:
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

核心操作是 `tl.cumsum`。`REVERSE` 分支用于反向的后缀和：前缀和加上当前元素，再从总和中相减，即 `total_sum - cumsum + s`。

从数学上看，单步 gate 为 $`g_i=\ln\alpha_i`$，chunk 内的累计量为

```math
G_i=\sum_{j\leq i}g_j=\ln\gamma_i.
```

当前 FLA 在调用时设置 `scale=RCP_LN2`，实际存入张量的是 $`G_i/\ln2`$，后续使用 `exp2`：

```math
\operatorname{exp2}\!\left(\frac{G_i-G_j}{\ln2}\right)
=\exp(G_i-G_j)=\frac{\gamma_i}{\gamma_j}.
```

下面的数值例子仍以自然对数 $`G`$ 展示，便于直接核对累计乘积。

举例：假设 chunk 内有 4 个 token，各自的 $`\alpha`$ 为 [0.9, 0.8, 0.95, 0.85]，对应 $`g = \log\alpha`$ 为 [-0.105, -0.223, -0.051, -0.163]。

| token（从 0 编号） | $`g_i=\ln\alpha_i`$ | $`G_i=\sum_{j\leq i}g_j`$ | $`\gamma_i=e^{G_i}`$ |
| --- | --- | --- | --- |
| 0 | −0.10536 | −0.10536 | 0.9 |
| 1 | −0.22314 | −0.32850 | 0.72 |
| 2 | −0.05129 | −0.37980 | 0.684 |
| 3 | −0.16252 | −0.54232 | 0.5814 |

从 token 1 的写入传播到 token 3，衰减为 $`\gamma_3/\gamma_1=0.95\times0.85=0.8075`$，也等于 $`\exp(-0.54232+0.32850)`$。

**访存特征。** 在 `[B,T,H]` 布局下，同一 head 的相邻时间位置相隔 H 个元素。原文提出将一维 scan 改为同时覆盖多个 head 的二维 scan，并报告了特定长序列配置下约 5 倍的加速。带宽利用率与加速倍数依赖线程布局、向量化、dtype 和测试形状，不能作为当前版本的通用结论。

### 5.3 chunk_scaled_dot_kkt_fwd

对应公式：计算 $`\widetilde{\mathbf{U}}`$ 所需的 UT 变换矩阵中的严格下三角部分：

```math
\text{strictLower}\left(\text{diag}(\beta_{[t]}) (\Gamma_{[t]} \odot \mathbf{K}_{[t]} \mathbf{K}_{[t]}^{\top})\right)
```

该阶段对应 `fla/ops/common/chunk_scaled_dot_kkt.py` 中的独立实现。当前 GDN 在 `BT=16/32` 时使用此路径；`BT=64` 时在融合 kernel 中执行同一数学计算。为保持六步算法与代码逐项对应，下面给出独立 KKT kernel 的完整函数体。

```python
def chunk_scaled_dot_kkt_fwd_kernel(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
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

几个实现细节：

- **地址计算**：当前实现用显式 offset 和 `tl.load(..., mask=..., other=0)` 处理边界。beta/g 的形状是 `[B,T,HV]`，K 是 `[B,T,H,K]`；共享 head 通过 `i_h // (HV // H)` 定位。
- **beta 后乘**：`b_A *= b_b[:, None]` 对结果逐行缩放。它等价于先给 K 的各行乘 beta 再做 KKT，但放在 K 分片循环之后可以减少重复缩放。
- **K 维分块**：每次加载 `[BT,BK]`，计算局部 KKT 并累加，直到覆盖整个 head 维度。
- **变长序列**：`cu_seqlens` 标记各序列边界，每条序列独立分成 BT 大小的 chunk，尾块用 mask 处理。`chunk_indices` 负责将这些局部 chunk 映射到统一的调度空间。

`chunk_indices` 的 shape 为 [`NT_total`, 2]，其中 `NT_total` 是所有序列的 chunk 总数。第二维固定为 2，存储一对值 (序列ID, 序列内chunk ID)，其通过 `prepare_chunk_indices` 生成：

```python
def _segmented_arange(counts: torch.LongTensor) -> tuple[torch.LongTensor, torch.LongTensor]:
    """Expand per-segment counts into flat per-slot index tensors.

    Given segment sizes ``counts = [c0, c1, ...]``, return two 1-D tensors of
    length ``counts.sum()`` that together label every slot with its segment and
    its position within that segment.

    Example -- ``counts = [2, 3]`` (segment 0 spans 2 slots, segment 1 spans 3)::

        seg_id    = [0, 0, 1, 1, 1]   # which segment each slot belongs to
        intra_idx = [0, 1, 0, 1, 2]   # running index within that segment

    With CUDA ``counts``, ``repeat_interleave`` reads ``counts.sum()`` on the
    host (one device sync). Pass host-side counts to avoid it.
    """
    seg_id = torch.repeat_interleave(
        torch.arange(counts.numel(), device=counts.device, dtype=counts.dtype),
        counts,
    )
    seg_start = F.pad(counts.cumsum(0), (1, 0))[:-1]
    intra_idx = torch.arange(seg_id.shape[0], device=counts.device, dtype=counts.dtype) - seg_start[seg_id]
    return seg_id, intra_idx

def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    cu_seqlens_cpu: torch.LongTensor | None = None,
) -> torch.LongTensor:
    src = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    chunk_counts = (prepare_lens(src) + (chunk_size - 1)).div(chunk_size, rounding_mode='floor')
    seg_id, intra_chunk_idx = _segmented_arange(chunk_counts)
    return torch.stack([seg_id, intra_chunk_idx], 1).to(cu_seqlens)

# cu_seqlens = [0, 16, 128], BT = 64
# 序列 0：长度 16，1 个 chunk；序列 1：长度 112，2 个 chunk。
# chunk_counts = [1, 2]
# seg_id = [0, 1, 1]
# intra_chunk_idx = [0, 0, 1]
# chunk_indices = [[0, 0], [1, 0], [1, 1]]
```

所有序列的 chunk 被展平并行调度，KKT kernel 的 grid 为 `(NT_total, B * HV)`。当前 `prepare_chunk_indices()` 通过 `_segmented_arange()` 生成序列 ID 和序列内 chunk ID，并可接收 CPU 边界元数据。kernel 随后解码这些索引：

```python
i_t, i_bh = tl.program_id(0), tl.program_id(1)
i_b, i_h = i_bh // HV, i_bh % HV
if IS_VARLEN:
    i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
    i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
else:
    bos, eos = i_b * T, i_b * T + T
# 当前 chunk 的 token 起点为 bos + i_t * BT。
```

cumsum、`chunk_scaled_dot_kkt`、`solve_tril`、`recompute_w_u`、`chunk_fwd_o` 等 chunk 间无依赖的 kernel 都使用这种方案。而 `forward_h`（隐状态递推）由于 chunk 间有严格顺序依赖，需要再额外使用 `chunk_offsets`，详见 5.6 节。

输出 A 的布局为 `[B,T,HV,BT]`：每个 token 位置存储所在 chunk 中对应的一行。

| 代码 | 公式中的操作 |
| --- | --- |
| `tl.dot(b_k, tl.trans(b_k))` | $`\mathbf K\mathbf K^\top`$ |
| `exp2(b_g[:, None] - b_g[None, :])` | 带衰减的因果权重 $`\Gamma`$（随后施加掩码） |
| `b_A *= b_b[:, None]` | 左乘 $`\operatorname{diag}(\beta)`$ |
| `o_t[:, None] > o_t[None, :]` | 严格下三角掩码 |

### 5.4 solve_tril

上一步得到严格下三角矩阵 $`\mathbf L^{(g)}`$。本阶段计算 $`(\mathbf I+\mathbf L^{(g)})^{-1}`$，再由下一阶段乘 beta，形成第 3.3 节的 $`\widetilde{\mathbf T}`$：

```math
\widetilde{\mathbf T}_{[t]}
=(\mathbf I+\mathbf L_{[t]}^{(g)})^{-1}\operatorname{diag}(\beta_{[t]}).
```

`I + L` 的对角线全为 1，适合用前代求解。注意源码复用了变量名 A：求解前是严格下三角部分，求解后是加单位阵所得矩阵的逆。

`fla/ops/utils/solve_tril.py` 采用两级分块：先分别求四个 16×16 对角块的逆，再以块矩阵乘法求六个非对角块。下面给出 `merge_16x16_to_64x64_inverse_kernel` 的当前实现，包含普通加载与 TMA 两条路径。

默认 64-token GDN 路径已经把 KKT 与相同的求解步骤融合；独立 `solve_tril` 中的这个函数仍能清楚展示两级算法。`DOT_PRECISION` 控制点积精度，实际候选配置由环境变量和模块级配置决定。

```python
def merge_16x16_to_64x64_inverse_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    o_t = (i_t * BT + o_i).to(tl.int64)
    if not USE_TMA:
        p_A_11 = A + o_t[:, None] * (H*BT) + o_i[None, :]
        p_A_22 = A + (o_t[:, None] + 16) * (H*BT) + (o_i[None, :] + 16)
        p_A_33 = A + (o_t[:, None] + 32) * (H*BT) + (o_i[None, :] + 32)
        p_A_44 = A + (o_t[:, None] + 48) * (H*BT) + (o_i[None, :] + 48)
        b_Ai_11 = tl.load(p_A_11, mask=(o_t[:, None] < T), other=0.0).to(tl.float32)
        b_Ai_22 = tl.load(p_A_22, mask=(o_t[:, None] + 16 < T), other=0.0).to(tl.float32)
        b_Ai_33 = tl.load(p_A_33, mask=(o_t[:, None] + 32 < T), other=0.0).to(tl.float32)
        b_Ai_44 = tl.load(p_A_44, mask=(o_t[:, None] + 48 < T), other=0.0).to(tl.float32)
    else:
        desc = make_tensor_descriptor(A, [T, BT], [H*BT, 1], [16, 16])
        desc_o = make_tensor_descriptor(Ai, [T, BT], [H*BT, 1], [16, 16])
        b_Ai_11 = desc.load([i_t * BT + 0, 0]).to(tl.float32)
        b_Ai_22 = desc.load([i_t * BT + 16, 16]).to(tl.float32)
        b_Ai_33 = desc.load([i_t * BT + 32, 32]).to(tl.float32)
        b_Ai_44 = desc.load([i_t * BT + 48, 48]).to(tl.float32)

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

    if not USE_TMA:
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
    else:
        b_A_21 = desc.load([i_t * BT + 16, 0]).to(tl.float32)
        b_A_31 = desc.load([i_t * BT + 32, 0]).to(tl.float32)
        b_A_32 = desc.load([i_t * BT + 32, 16]).to(tl.float32)
        b_A_41 = desc.load([i_t * BT + 48, 0]).to(tl.float32)
        b_A_42 = desc.load([i_t * BT + 48, 16]).to(tl.float32)
        b_A_43 = desc.load([i_t * BT + 48, 32]).to(tl.float32)

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

    if not USE_TMA:
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
    else:
        desc_o.store([i_t * BT + 0, 0], b_Ai_11.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 16, 16], b_Ai_22.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 32, 32], b_Ai_33.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 48, 48], b_Ai_44.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 16, 0], b_Ai_21.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 32, 0], b_Ai_31.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 32, 16], b_Ai_32.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 48, 0], b_Ai_41.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 48, 16], b_Ai_42.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 48, 32], b_Ai_43.to(desc_o.dtype, fp_downcast_rounding="rtne"))
```

下面分步解释两级操作的原理。

#### 第一级：对角线上的 16x16 小块求逆

四个 16×16 对角块可以独立求逆。代码将严格下三角部分取负作为初值，从第 2 行开始执行前代：第 0 行没有待更新元素，第 1 行的严格下三角结果已经由初始化得到。

对后续各行，`tl.sum(b_a[:, None] * b_Ai, 0)` 累加已求出行的贡献，`tl.where((o_i == i)[:, None], ...)` 将结果写入当前行。循环结束后加上单位阵 `m_I`，得到该对角块的完整逆矩阵。

#### 第二级：合并 16x16 块为 64x64 逆矩阵

一个 64x64 的下三角矩阵可以分成 4x4 = 16 个 16x16 块：

```math
\mathbf M=\begin{bmatrix}
 A_{11}&0&0&0\\ A_{21}&A_{22}&0&0\\
 A_{31}&A_{32}&A_{33}&0\\ A_{41}&A_{42}&A_{43}&A_{44}
\end{bmatrix},\qquad
\mathbf X=\mathbf M^{-1}=\begin{bmatrix}
 X_{11}&0&0&0\\ X_{21}&X_{22}&0&0\\
 X_{31}&X_{32}&X_{33}&0\\ X_{41}&X_{42}&X_{43}&X_{44}
\end{bmatrix}.
```

其中 $`X_{ii}=A_{ii}^{-1}`$；非对角块 $`X_{ij}`$ 是整体逆矩阵中的一个块，并非对 $`A_{ij}`$ 单独求逆。

对角块 $`X_{ii}`$ 已在第一级求出，非对角块通过块前代计算，依赖关系如下：

```text
X11 = A11^(-1) --+
                 +--> X21 = -X22 @ A21 @ X11
X22 = A22^(-1) --+

X11 -------------+
X21 -------------+--> X31 = -X33 @ (A31 @ X11 + A32 @ X21)
X33 = A33^(-1) --+

其余非对角块按相同的块前代关系计算。
```

两级分块的好处：第一级的 4 个 16x16 块可以完全并行求逆；第二级利用已求出的对角块逆，通过矩阵乘法计算非对角块逆。如果直接对 64x64 矩阵做逐行前代，并行度低且访存分散。融合版本将两级操作放在同一个 thread block 内完成，局部性好且省去了中间结果的 HBM 读写。

求解后，A 矩阵就是 $`(\mathbf{I} + \text{strictLower}(...))^{-1}`$。这里还没有乘以 $`\text{diag}(\beta)`$，这步在下面的 `recompute_w_u_fwd` 中完成。

### 5.5 recompute_w_u_fwd

对应公式：

```math
\widetilde{\mathbf U}_{[t]}=\widetilde{\mathbf T}_{[t]}\mathbf V_{[t]},
\qquad
\overleftarrow{\mathbf W}_{[t]}=
\widetilde{\mathbf T}_{[t]}\operatorname{diag}(\gamma_{[t]})\mathbf K_{[t]}.
```

代码位于 `fla/ops/gated_delta_rule/wy_fast.py`，这个 kernel 把上一步得到的 $`(\mathbf{I} + A)^{-1}`$ （存在 `b_A` 中）与 $`\text{diag}(\beta)`$ 以及 K/V 相乘，一次性得到 W 和 U：

```python
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
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

U 使用 $`\mathbf V`$ 作为右端项，W 使用 $`\operatorname{diag}(\gamma)\mathbf K`$。所以代码先将 K 逐行乘 beta 和累计衰减，再左乘 `b_A`；U 则先将 V 乘 beta，再左乘同一个 `b_A`。

W 的额外衰减用于块入口状态的传播：$`\overleftarrow{\mathbf W}\mathbf S_{[t]}^\top`$ 要扣除旧状态在各位置的预测。U 描述本块输入 value 的贡献，其位置间衰减已经包含在带 gate 的 UT 矩阵中。结合第 3.3 节的相似变换关系，代码与 $`\overleftarrow{\boldsymbol w^r}=\gamma^r\boldsymbol w^r`$ 一致。

### 5.6 chunk_gated_delta_rule_fwd_h

对应公式：

```math
\mathbf{S}_{[t+1]} = \overrightarrow{\mathbf{S}_{[t]}} + \left(\widetilde{\mathbf{U}}_{[t]} - \overleftarrow{\mathbf{W}_{[t]}} \mathbf{S}_{[t]}^{\top}\right)^{\top} \overrightarrow{\mathbf{K}_{[t]}}
```

这是 Chunk-wise 算法的状态递推阶段，按顺序计算 $`\mathbf S_{[0]},\mathbf S_{[1]},\ldots`$。每个 chunk 的入口状态会保存到 HBM，供后续 `chunk_fwd_o` 读取；入口状态就绪后，各块的输出便可以并行计算。

代码位于 `fla/ops/common/chunk_delta_h.py`，kernel 名为 `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`。默认状态布局为 `[K,V]`，所以代码的 `h` 对应公式中的 $`\mathbf S^\top`$。

**变长输入与 chunk_offsets。** 状态更新的 chunk 间有顺序依赖，因此 grid 为 `(V_tiles, N * HV)`，每个 program 处理一条序列、一个 V head 和一片 V 通道，并在内部遍历该序列的所有 chunk。`chunk_offsets` 给出每条序列在状态缓冲区中的起始 chunk：

```python
# 例：cu_seqlens = [0, 128, 254, 1254], BT = 64
# 序列长度: [128, 126, 1000], chunk 数量: [2, 2, 16]
chunk_offsets = [0, 2, 4, 20]  # 累积和
# 序列 0 的 h 存在 h[0:2], 序列 1 在 h[2:4], 序列 2 在 h[4:20]
```

Python 调用侧逻辑（`chunk_gated_delta_rule_fwd_h` 函数）：

```python
def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V, HV = *k.shape, u.shape[-1], u.shape[2]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    if state_v_first:
        h = k.new_empty(B, NT, HV, V, K)
        final_state = k.new_zeros(N, HV, V, K, dtype=torch.float32) if output_final_state else None
    else:
        h = k.new_empty(B, NT, HV, K, V)
        final_state = k.new_zeros(N, HV, K, V, dtype=torch.float32) if output_final_state else None

    v_new = torch.empty_like(u) if save_new_value else None
    def grid(meta): return (triton.cdiv(V, meta['BV']), N*HV)
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        STATE_V_FIRST=state_v_first,
    )
    return h, v_new, final_state
```

关键点：

- `h` 的形状为 `[B,NT,HV,K,V]`，保存各个 chunk 的入口状态 $`\mathbf S_{[t]}^\top`$。变长模式下 B 为 1，NT 是所有序列的 chunk 总数。
- `v_new` 对应 $`\widetilde{\mathbf U}-\overleftarrow{\mathbf W}\mathbf S^\top`$。
- grid 按序列和 head 并行，chunk 维度在 program 内通过循环顺序遍历。
- `chunk_offsets` 仅用于变长模式；等长模式直接由序列编号计算状态偏移。
- K 维按 64 划分为 `b_h1`、`b_h2` 等子块，分别参与点积与更新。这种分块影响寄存器分配和编译器调度，但并不意味着总状态元素数量变少。
- 工作状态使用 FP32 累加，块入口快照 h 按 K 的 dtype 保存，返回的 `final_state` 为 FP32。

```python
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    if STATE_V_FIRST:
        b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([BV, 64], dtype=tl.float32)
    else:
        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += (boh * HV + i_h).to(tl.int64) * K*V
    v += (bos * HV + i_h).to(tl.int64) * V
    k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    if SAVE_NEW_VALUE:
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
        if STATE_V_FIRST:
            p_h0_1 = h0 + o_v[:, None] * K + o_k1[None, :]
            m_h0_1 = m_v[:, None] & m_k1[None, :]
        else:
            p_h0_1 = h0 + o_k1[:, None] * V + o_v[None, :]
            m_h0_1 = m_k1[:, None] & m_v[None, :]
        b_h1 += tl.load(p_h0_1, mask=m_h0_1, other=0.0).to(tl.float32)
        if K > 64:
            if STATE_V_FIRST:
                p_h0_2 = h0 + o_v[:, None] * K + o_k2[None, :]
                m_h0_2 = m_v[:, None] & m_k2[None, :]
            else:
                p_h0_2 = h0 + o_k2[:, None] * V + o_v[None, :]
                m_h0_2 = m_k2[:, None] & m_v[None, :]
            b_h2 += tl.load(p_h0_2, mask=m_h0_2, other=0.0).to(tl.float32)
        if K > 128:
            if STATE_V_FIRST:
                p_h0_3 = h0 + o_v[:, None] * K + o_k3[None, :]
                m_h0_3 = m_v[:, None] & m_k3[None, :]
            else:
                p_h0_3 = h0 + o_k3[:, None] * V + o_v[None, :]
                m_h0_3 = m_k3[:, None] & m_v[None, :]
            b_h3 += tl.load(p_h0_3, mask=m_h0_3, other=0.0).to(tl.float32)
        if K > 192:
            if STATE_V_FIRST:
                p_h0_4 = h0 + o_v[:, None] * K + o_k4[None, :]
                m_h0_4 = m_v[:, None] & m_k4[None, :]
            else:
                p_h0_4 = h0 + o_k4[:, None] * V + o_v[None, :]
                m_h0_4 = m_k4[:, None] & m_v[None, :]
            b_h4 += tl.load(p_h0_4, mask=m_h0_4, other=0.0).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        i_t_int64 = i_t.to(tl.int64)
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        if STATE_V_FIRST:
            p_h1 = h + i_t_int64 * HV*K*V + o_v[:, None] * K + o_k1[None, :]
            m_h1 = m_v[:, None] & m_k1[None, :]
        else:
            p_h1 = h + i_t_int64 * HV*K*V + o_k1[:, None] * V + o_v[None, :]
            m_h1 = m_k1[:, None] & m_v[None, :]
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), mask=m_h1)
        if K > 64:
            if STATE_V_FIRST:
                p_h2 = h + i_t_int64 * HV*K*V + o_v[:, None] * K + o_k2[None, :]
                m_h2 = m_v[:, None] & m_k2[None, :]
            else:
                p_h2 = h + i_t_int64 * HV*K*V + o_k2[:, None] * V + o_v[None, :]
                m_h2 = m_k2[:, None] & m_v[None, :]
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), mask=m_h2)
        if K > 128:
            if STATE_V_FIRST:
                p_h3 = h + i_t_int64 * HV*K*V + o_v[:, None] * K + o_k3[None, :]
                m_h3 = m_v[:, None] & m_k3[None, :]
            else:
                p_h3 = h + i_t_int64 * HV*K*V + o_k3[:, None] * V + o_v[None, :]
                m_h3 = m_k3[:, None] & m_v[None, :]
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), mask=m_h3)
        if K > 192:
            if STATE_V_FIRST:
                p_h4 = h + i_t_int64 * HV*K*V + o_v[:, None] * K + o_k4[None, :]
                m_h4 = m_v[:, None] & m_k4[None, :]
            else:
                p_h4 = h + i_t_int64 * HV*K*V + o_k4[:, None] * V + o_v[None, :]
                m_h4 = m_k4[:, None] & m_v[None, :]
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), mask=m_h4)

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
        if K > 128:
            p_w = w + o_t[:, None] * (HV*K) + o_k3[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k3[None, :], other=0.0)
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = w + o_t[:, None] * (HV*K) + o_k4[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k4[None, :], other=0.0)
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        p_v = v + o_t[:, None] * (HV*V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_t[:, None] & m_v[None, :], other=0.0) - b_v

        if SAVE_NEW_VALUE:
            p_v = v_new + o_t[:, None] * (HV*V) + o_v[None, :]
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])

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

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(gk + (bos + last_idx) * HV*K + i_h * K + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
            if STATE_V_FIRST:
                b_h1 *= exp2(b_gk_last1)[None, :]
            else:
                b_h1 *= exp2(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(gk + (bos + last_idx) * HV*K + i_h * K + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
                if STATE_V_FIRST:
                    b_h2 *= exp2(b_gk_last2)[None, :]
                else:
                    b_h2 *= exp2(b_gk_last2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(gk + (bos + last_idx) * HV*K + i_h * K + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
                if STATE_V_FIRST:
                    b_h3 *= exp2(b_gk_last3)[None, :]
                else:
                    b_h3 *= exp2(b_gk_last3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(gk + (bos + last_idx) * HV*K + i_h * K + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
                if STATE_V_FIRST:
                    b_h4 *= exp2(b_gk_last4)[None, :]
                else:
                    b_h4 *= exp2(b_gk_last4)[:, None]
        b_v = b_v.to(k.dtype.element_ty)

        p_k = k + o_k1[:, None] + o_t[None, :] * (H*K)
        b_k = tl.load(p_k, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        if STATE_V_FIRST:
            b_h1 += tl.trans(tl.dot(b_k, b_v))
        else:
            b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = k + o_k2[:, None] + o_t[None, :] * (H*K)
            b_k = tl.load(p_k, mask=m_k2[:, None] & m_t[None, :], other=0.0)
            if STATE_V_FIRST:
                b_h2 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            p_k = k + o_k3[:, None] + o_t[None, :] * (H*K)
            b_k = tl.load(p_k, mask=m_k3[:, None] & m_t[None, :], other=0.0)
            if STATE_V_FIRST:
                b_h3 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            p_k = k + o_k4[:, None] + o_t[None, :] * (H*K)
            b_k = tl.load(p_k, mask=m_k4[:, None] & m_t[None, :], other=0.0)
            if STATE_V_FIRST:
                b_h4 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h4 += tl.dot(b_k, b_v)

    if STORE_FINAL_STATE:
        if STATE_V_FIRST:
            p_ht = ht + o_v[:, None] * K + o_k1[None, :]
            m_ht = m_v[:, None] & m_k1[None, :]
        else:
            p_ht = ht + o_k1[:, None] * V + o_v[None, :]
            m_ht = m_k1[:, None] & m_v[None, :]
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), mask=m_ht)
        if K > 64:
            if STATE_V_FIRST:
                p_ht = ht + o_v[:, None] * K + o_k2[None, :]
                m_ht = m_v[:, None] & m_k2[None, :]
            else:
                p_ht = ht + o_k2[:, None] * V + o_v[None, :]
                m_ht = m_k2[:, None] & m_v[None, :]
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), mask=m_ht)
        if K > 128:
            if STATE_V_FIRST:
                p_ht = ht + o_v[:, None] * K + o_k3[None, :]
                m_ht = m_v[:, None] & m_k3[None, :]
            else:
                p_ht = ht + o_k3[:, None] * V + o_v[None, :]
                m_ht = m_k3[:, None] & m_v[None, :]
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), mask=m_ht)
        if K > 192:
            if STATE_V_FIRST:
                p_ht = ht + o_v[:, None] * K + o_k4[None, :]
                m_ht = m_v[:, None] & m_k4[None, :]
            else:
                p_ht = ht + o_k4[:, None] * V + o_v[None, :]
                m_ht = m_k4[:, None] & m_v[None, :]
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), mask=m_ht)
```

代码与公式的对应关系：

| 代码变量或操作 | 对应的公式 |
| --- | --- |
| 输入 `v` | $`\widetilde{\mathbf U}_{[t]}`$ |
| `w` | $`\overleftarrow{\mathbf W}_{[t]}`$ |
| 保存到 `v_new` 的 `b_v` | $`\widetilde{\mathbf U}_{[t]}-\overleftarrow{\mathbf W}_{[t]}\mathbf S_{[t]}^\top`$ |
| `b_h` | $`\mathbf S_{[t]}^\top`$ |
| `exp2(b_g_last)` | $`\gamma_{[t]}^C`$ |
| `exp2(b_g_last - b_g)` | 每次新写入传播到块末的衰减 $`\gamma^C/\gamma^r`$ |

`v_new` 在乘块末衰减之前保存，因为输出 kernel 还需要各个位置原始的写入误差。`last_idx` 使用当前 chunk 最后一个有效 token，尾块不足 BT 时也按真实序列终点更新状态。

#### Varlen 处理

其他 kernel 可以将各序列的 chunk 展平并行，而状态递推必须分别沿每条序列顺序执行。`prepare_chunk_offsets()` 先计算每条序列的 chunk 数，再做前缀和，定位该序列在 h 缓冲区中的区间：

```python
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    return F.pad(triton.cdiv(prepare_lens(cu_seqlens), chunk_size), (1, 0), value=0).cumsum(-1)

# 例：cu_seqlens = [0, 128, 254, 1254], BT = 64
# 序列长度: [128, 126, 1000], chunk 数量: [2, 2, 16]
# chunk_offsets = [0, 2, 4, 20]  （累积和，序列 0 在 h[0:2], 序列 1 在 h[2:4], 序列 2 在 h[4:20]）
```

调用侧通过 `chunk_offsets` 设置 grid 和传参，kernel 内部根据 `i_n`（序列 ID）从 `chunk_offsets` 读取偏移：

```python
if IS_VARLEN:
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    NT = tl.cdiv(T, BT)
    boh = tl.load(chunk_offsets + i_n).to(tl.int32)
else:
    bos, eos = i_n * T, i_n * T + T
    NT = tl.cdiv(T, BT)
    boh = i_n * NT

# token 偏移用于 K/V，head 映射支持 H != HV。
k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
v += (bos * HV + i_h).to(tl.int64) * V
# chunk 偏移用于块入口状态 h。
h += (boh * HV + i_h).to(tl.int64) * K * V
```

`bos` 是 token 空间的偏移，用于 K/V/W/g；`boh` 是 chunk 空间的偏移，用于状态快照 h。每条序列由独立的 program 维护工作状态，并从该序列的初始状态开始递推。

### 5.7 chunk_fwd_o

对应公式：

```math
\mathbf O_{[t]}
=\overleftarrow{\mathbf Q}_{[t]}\mathbf S_{[t]}^\top
+(\mathbf Q_{[t]}\mathbf K_{[t]}^\top\odot\Gamma_{[t]})
 \left(\widetilde{\mathbf U}_{[t]}-
 \overleftarrow{\mathbf W}_{[t]}\mathbf S_{[t]}^\top\right).
```

输出由两部分组成：inter-chunk（当前 query 与历史累积状态的交互）和 intra-chunk（当前 chunk 内部 query 和 key 的注意力交互）。

代码位于 `fla/ops/common/chunk_o.py`：

```python
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    g_gamma,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV

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
        if STATE_V_FIRST:
            p_h = h + o_v[:, None] * K + o_k[None, :]
            m_h = (o_v[:, None] < V) & m_k[None, :]
        else:
            p_h = h + o_k[:, None] * V + o_v[None, :]
            m_h = m_k[:, None] & (o_v[None, :] < V)
        # [BT, BK]
        b_q = tl.load(p_q, mask=m_t[:, None] & m_k[None, :], other=0.0)
        # [BK, BT]
        b_k = tl.load(p_k, mask=m_k[:, None] & m_t[None, :], other=0.0)
        b_h = tl.load(p_h, mask=m_h, other=0.0)

        # [BT, BK] @ [BK, BV] -> [BT, BV]
        if STATE_V_FIRST:
            b_o += tl.dot(b_q, tl.trans(b_h))
        else:
            b_o += tl.dot(b_q, b_h)
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        g += bos * HV + i_h
        p_g = g + o_t * HV
        b_g = tl.load(p_g, mask=m_t, other=0.0)
        b_o = b_o * exp2(b_g)[:, None]
        b_A = b_A * exp2(b_g[:, None] - b_g[None, :])
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
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

两部分的含义：

- **Inter-chunk**：$`\overleftarrow{\mathbf Q}\mathbf S^\top`$。当前 query 查询块入口之前的历史状态，并乘从块入口到当前位置的累计衰减。代码的 `b_h` 是状态快照 $`\mathbf S_{[t]}^\top`$。
- **Intra-chunk**：$`(\mathbf Q\mathbf K^\top\odot\Gamma)\,v_{\mathrm{new}}`$。当前 query 查询本块已经发生的写入；$`\Gamma`$ 同时包含因果约束和位置间衰减，对角线保留，因此可以读取当前 token 的新写入。

**IO 优化。** 两部分共享 Q 的加载，在同一个 kernel 内计算。局部矩阵 `b_A` 在片上形成，直接与 `b_v` 相乘，不需要将每个 chunk 的注意力矩阵写回 HBM。输出前，两项统一乘 `scale`；若公式中已经将 $`d_k^{-1/2}`$ 吸收到 Q，就不能再重复缩放。

衰减矩阵的作用：

```text
# 用自然对数累计量 G 表示：Gamma[i,j] = exp(G[i] - G[j])，j <= i。
# G = [-0.1, -0.3, -0.5, -0.7]
# 当前代码存储 g = G / ln(2)，相应使用 exp2(g[i] - g[j])。
#
#          j=0        j=1        j=2        j=3
# i=0   [exp(0)        0           0          0   ]
# i=1   [exp(-0.2)   exp(0)        0          0   ]
# i=2   [exp(-0.4)   exp(-0.2)   exp(0)       0   ]
# i=3   [exp(-0.6)   exp(-0.4)   exp(-0.2)  exp(0)]
```

### 5.8 反向传播

反向流程保留与前向相对应的阶段：重计算中间量，计算局部输出梯度，反向递推状态梯度，再经过 WY 表示传播回原始输入。当前实现如下：

```python
def chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_gate_in_kernel: bool = False,
    g_input: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    chunk_size: int = 64,
):
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    if cp_context is not None:
        initial_state = expand_h0(initial_state, context=cp_context)

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

    if cp_context is not None:
        # initial_state is None in the CP mode
        # We only need to compute dht of current rank and pass it to the backward kernel
        dht, initial_state = chunk_gated_delta_rule_bwd_dhu_pre_process(
            q=q,
            k=k,
            w=w,
            do=do,
            dv=dv,
            g=g,
            scale=scale,
            cu_seqlens=cu_seqlens,
            dht=dht,
            initial_state=initial_state,
            context=cp_context,
            state_v_first=state_v_first,
            chunk_size=chunk_size,
        )

    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=g,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
        chunk_size=chunk_size,
    )
    dq, dk, dw, dg = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,
        w=w,
        g=g,
        h=h,
        dv=dv,
        do=do,
        dh=dh,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
        chunk_size=chunk_size,
    )
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
    dA_log, ddt_bias = None, None
    if use_gate_in_kernel:
        dg, dA_log, ddt_bias = gdn_gate_bwd(g=g_input, A_log=A_log, dt_bias=dt_bias, dyg=dg)
    return dq, dk, dv, db, dg, dh0, dA_log, ddt_bias
```

主要步骤：

(a) 前向重计算：反向传播开始时，先用保存的 A 矩阵重算 w, u, h, `v_new`，而不是从前向传播中保存这些中间变量。这是经典的 activation checkpointing 策略，用计算换显存。

**(b) 局部梯度。** `chunk_bwd_dv_local` 计算当前块输出对 `v_new` 的贡献。前向矩阵转置后，因果方向相应反转；gate 也需要保留：

```math
\mathrm{d}v_{\mathrm{local}}
=\mathrm{scale}\cdot
 (\mathbf K\mathbf Q^\top\odot\Gamma^\top)\,\mathrm{d}o.
```

这里的 Q 尚未乘 `scale`，与 kernel 的调用约定一致。

**(c) 状态梯度。** `chunk_gated_delta_rule_bwd_dhu` 从后向前遍历 chunk，将未来状态对当前写入的梯度加到局部梯度上。其数学递推如下，对应 kernel 中按 K/V 分片累加的操作：

```python
# 单个 chunk 的数学伪代码：g 是 log2 累计衰减，h 的布局为 [K, V]。
# d_v_new 已包含当前块输出对 v_new 的局部梯度。
for i_t in range(NT - 1, -1, -1):
    save_dh(i_t, d_h)
    decay_to_end = exp2(g_last - g)
    d_v_new += (k @ d_h) * decay_to_end[:, None]
    d_h = (
        exp2(g_last) * d_h
        + (q * exp2(g)[:, None] * scale).T @ d_o
        - w.T @ d_v_new
    )
```

(d) `chunk_bwd_dqkwg`：在一个 kernel 中同时计算 dq, dk, dw, dg，合并多个梯度计算以减少 kernel launch 开销和重复的数据加载。

**(e) WY 表示的反向。** 求逆的微分满足 $`\mathrm d(\mathbf A^{-1})=-\mathbf A^{-1}(\mathrm d\mathbf A)\mathbf A^{-1}`$。若写成标量损失的梯度，则为 $`\nabla_{\mathbf A}\mathcal L=-\mathbf A^{-\top}(\nabla_{\mathbf A^{-1}}\mathcal L)\mathbf A^{-\top}`$。当前 kernel 以转置布局加载逆矩阵，因此代码中的 `b_A` 已对应转置后的因子：

```python
# prepare_wy_repr_bwd_kernel 中，b_A 已按转置布局加载。
m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
b_dA = tl.where(m_A, b_dA, 0)
b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
if USE_G:
    b_dA *= exp2(b_g[:, None] - b_g[None, :])
b_dA = tl.where(m_A, -b_dA, 0).to(k.dtype.element_ty)
```

**(f) gate 的反向累计。** 对 $`y_i=\sum_{j\leq i}x_j`$，有 $`\partial\mathcal L/\partial x_j=\sum_{i\geq j}\partial\mathcal L/\partial y_i`$，因此使用 reverse cumsum。前向的 `1 / ln(2)` 与指数换底的链式因子抵消；若 gate 在 kernel 内激活，还会继续反传到 `A_log` 和 `dt_bias`。Q/K 的归一化梯度由外层 autograd 包装处理。

## 6. Recurrent 算法代码解析

### 6.1 Fused Recurrent 说明

Recurrent 模式是直接按时间步逐个计算递推过程，适用于推理解码场景（逐 token 生成）。调用侧代码位于 `fla/ops/gated_delta_rule/fused_recurrent.py`：

```python
def fused_recurrent_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    BV = min(8, triton.next_power_of_2(V)) if gv is None else triton.next_power_of_2(V)
    NV = triton.cdiv(V, BV)

    o = torch.empty_like(v)
    if output_final_state:
        if state_v_first:
            final_state = q.new_empty(N, HV, V, K, dtype=torch.float32)
        else:
            final_state = q.new_empty(N, HV, K, V, dtype=torch.float32)
    else:
        final_state = None

    grid = (NV, N * HV)
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        gk=gk,
        gv=gv,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        IS_BETA_HEADWISE=beta.ndim != v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        APPLY_BETA_SIGMOID=use_beta_sigmoid_in_kernel,
        ALLOW_NEG_EIGVAL=allow_neg_eigval,
        STATE_V_FIRST=state_v_first,
        num_warps=1,
        num_stages=3,
    )
    return o, final_state
```

`final_state` 的形状为 `[N,HV,K,V]`，类型为 FP32。对 GDN 层而言，递归状态的元素数与序列长度无关，为 $`N\times H_V\times d_k\times d_v`$；完整模型还需要乘以对应的 GDN 层数，并计入短卷积状态。

标准注意力的 K/V cache 则随序列长度增长，元素数约为 $`2\times N\times n_{\mathrm{layers}}\times L\times H_{\mathrm{KV}}\times d`$。混合架构需要同时保留 GDN 状态和 full-attention 层的 K/V cache。

同时调用侧进行了Grid 的划分：

- `program_id(0)`：V 维分片索引，`NV = cdiv(V, BV)`。
- `program_id(1)`：序列与 V head 的组合索引，范围为 `N * HV`。
- 每个 program 维护 `[BK,BV]` 的 FP32 状态块。
- GVA 通过 `i_h = i_hv // (HV // H)` 将 V head 映射到 Q/K head。

当前 FLA 的 fused recurrent 包装没有实现 backward，训练使用 chunk 路径。这是当前实现的接口限制；递推形式本身可以通过保存状态或重计算实现反向传播。

### 6.2 Kernel 实现

下面给出 recurrent kernel 的完整实现，用于与第 3.1 节的递推公式直接对照。当前版本同时包含共享 head、可选的 gate 激活和两种状态布局。

```python
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    gk,
    gv,
    beta,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GATE_IN_KERNEL: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    APPLY_BETA_SIGMOID: tl.constexpr,
    ALLOW_NEG_EIGVAL: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if USE_G:
        p_g = g + bos * HV + i_hv
    if USE_GK:
        p_gk = gk + (bos * HV + i_hv) * K + o_k
    if USE_GV:
        p_gv = gv + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + bos * HV + i_hv
    else:
        p_beta = beta + (bos * HV + i_hv) * V + o_v

    p_o = o + (bos * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    if STATE_V_FIRST:
        mask_h = mask_v[:, None] & mask_k[None, :]
    else:
        mask_h = mask_k[:, None] & mask_v[None, :]

    if STATE_V_FIRST:
        b_h = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if STATE_V_FIRST:
            p_h0 = h0 + i_nh * K*V + o_v[:, None] * K + o_k[None, :]
        else:
            p_h0 = h0 + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in tl.range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta).to(tl.float32)
        else:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        if APPLY_BETA_SIGMOID:
            b_beta = tl.sigmoid(b_beta)
            if ALLOW_NEG_EIGVAL:
                b_beta = b_beta * 2

        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            if USE_GATE_IN_KERNEL:
                b_A = tl.load(A_log + i_hv).to(tl.float32)
                if HAS_DT_BIAS:
                    b_g = b_g + tl.load(dt_bias + i_hv).to(tl.float32)
                b_g = -exp(b_A) * softplus(b_g)
            b_h *= exp(b_g)

        if USE_GK:
            b_gk = tl.load(p_gk).to(tl.float32)
            if STATE_V_FIRST:
                b_h *= exp(b_gk[None, :])
            else:
                b_h *= exp(b_gk[:, None])

        if USE_GV:
            b_gv = tl.load(p_gv).to(tl.float32)
            if STATE_V_FIRST:
                b_h *= exp(b_gv[:, None])
            else:
                b_h *= exp(b_gv[None, :])

        if STATE_V_FIRST:
            b_v = b_beta * (b_v - tl.sum(b_h * b_k[None, :], 1))
            b_h += b_v[:, None] * b_k[None, :]
            b_o = tl.sum(b_h * b_q[None, :], 1)
        else:
            b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
            b_h += b_k[:, None] * b_v
            b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H*K
        p_k += H*K
        p_v += HV*V
        if USE_G:
            p_g += HV
        if USE_GK:
            p_gk += HV*K
        if USE_GV:
            p_gv += HV*V
        p_beta += HV * (1 if IS_BETA_HEADWISE else V)
        p_o += HV*V

    if STORE_FINAL_STATE:
        if STATE_V_FIRST:
            p_ht = ht + i_nh * K*V + o_v[:, None] * K + o_k[None, :]
        else:
            p_ht = ht + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
```

然后列出代码与 GDN 递推公式的对应关系：

```math
\begin{aligned}
\mathbf S_t
&=\mathbf S_{t-1}\alpha_t(\mathbf I-\beta_t\boldsymbol k_t\boldsymbol k_t^\top)
 +\beta_t\boldsymbol v_t\boldsymbol k_t^\top\\
&=\alpha_t\mathbf S_{t-1}
 +\beta_t(\boldsymbol v_t-\alpha_t\mathbf S_{t-1}\boldsymbol k_t)\boldsymbol k_t^\top.
\end{aligned}
```

代码中的 `b_h` 存储上述状态的转置：

```python
b_h *= exp(b_g)                                      # h = alpha * h
b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))  # 写入误差
b_h += b_k[:, None] * b_v                            # h += k * error^T
```

门控衰减在 delta 更新之前执行，因此计算旧 value 时，状态已经衰减为 $`\alpha_t\mathbf S_{t-1}`$。代码采用转置布局，`tl.sum(b_h * b_k[:, None], 0)` 正好对应公式中的 $`\alpha_t\mathbf S_{t-1}\boldsymbol k_t`$。

Q/K 沿时间前进 `H * K` 个元素，V/O 前进 `HV * V`；标量 gate 的 stride 为 HV，beta 的 stride 取决于 headwise 或逐通道模式。

这个 recurrent kernel 的循环由逐元素运算、外积和归约组成，没有 `tl.dot`。它适合单 token 解码：只需读取和更新一次状态；长序列训练则通过前文的 Chunk-wise 算法，把块内的大部分工作改写成矩阵乘法。原文提到的 CuTe DSL decoder 优化属于进一步的推理实现优化，不影响上述递推与分块公式的等价性。

---

整理自[原文](https://zhuanlan.zhihu.com/p/2007937984738129405)，保留其章节、推导与代码讲解顺序。代码按 Transformers v5.17.0、FLA v0.5.2 更新（2026-09-26）。源码分别遵循 [Transformers Apache-2.0](https://github.com/huggingface/transformers/blob/v5.17.0/LICENSE) 与 [FLA MIT](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/LICENSE) 许可；版权归 The Qwen Team、The HuggingFace Inc. team，以及 Songlin Yang、Yu Zhang、Zhiyuan Li 等贡献者。

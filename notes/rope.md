# `RoFormer`: `Enhanced Transformer with Rotary Position Embedding`

先从一维位置推导旋转，再沿 `Qwen3.5` 的实际输入链路展开：`processor` 生成媒体网格，`get_rope_index()` 为每个 `token` 分配三轴位置，`rotary_emb` 将位置变成旋转系数，`attention` 再把系数作用到 `Q/K`。最后补充 `TM-RoPE` 如何让音频和视频共用时间坐标。

下文沿用 [`Qwen3.5-27B` 的配置](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/config.json)，接口核对到 **2026-09-19 的 `Transformers` 主干提交 [`c587bc8`](https://github.com/huggingface/transformers/tree/c587bc884db2c2e31fc2b8102314656b17aa07b1)**。配置、张量形状和调用位置均按这个版本说明。

## 一维位置与 `partial RoPE`

`RoPE`（`Rotary Position Embedding`）通过旋转 `Q/K` 特征，使 `attention` 内积包含相对位置信息。设旋转维度为偶数 $`d_r`$，频率底数为 $`\beta`$，位置为 $`p`$；第 $`i`$ 个二维子空间的角频率与旋转角为

```math
\omega_i=\beta^{-2i/d_r},\qquad
\phi_{p,i}=p\omega_i,\qquad
i=0,\ldots,\frac{d_r}{2}-1.
```

- $`\beta`$ 对应 `rope_theta`，源码变量名为 `base`，是生成频率序列的底数
- 当 $`\beta>1`$ 时，频率以公比 $`\beta^{-2/d_r}`$ 递减：$`\omega_0=1`$，后续频率逐渐降低。增大 `base` 会拉长 $`i>0`$ 各分量的周期 $`2\pi/\omega_i`$

- `inv_freq[i]` 表示第 $`i`$ 对旋转特征每单位位置的相位增量，实际旋转角为 `position_id * inv_freq[i]`

```json
{
  "text_config": {
    "hidden_size": 5120,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "max_position_embeddings": 262144,
    "rope_parameters": {
      "rope_type": "default",
      "rope_theta": 10000000,
      "partial_rotary_factor": 0.25,
      "mrope_interleaved": true,
      "mrope_section": [11, 11, 10]
    }
  }
}
```

源码采用 `[B,H,T,d_h]` 布局：`B` 为 `batch size`，`T` 为当前输入长度，`Q/K` 的 `head` 数分别为 `24/4`；`head_dim` 为 $`d_h=256`$，旋转比例 $`r=0.25`$，故 $`d_r=\lfloor d_h r\rfloor=64`$

```text
Q / K            [B, 24, T, 256] / [B, 4, T, 256]
position_ids     [B, T]
inv_freq         [32]
相位 Φ           [B, T, 32]
cos / sin        [B, T, 64] → unsqueeze(1) → [B, 1, T, 64]
```

先由 `Qwen3_5TextRotaryEmbedding.compute_default_rope_parameters()` 计算 32 个频率：

```python
base = config.rope_parameters["rope_theta"]
partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
dim = int(head_dim * partial_rotary_factor)

attention_factor = 1.0
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
return inv_freq.to(device), attention_factor
```

- `arange(0,64,2)/64` 生成指数 $`[0,1/32,\ldots,31/32]`$，得到 `inv_freq` 为 $`[1,10^{-7/32},\ldots,10^{-217/32}]`$
- 频率按旋转维度 `64` 直接计算，而非从全维 `RoPE` 中截取

`RoFormer` §3.2.1 式（13）先给出二维形式。只写 `Q` 分支，以 $`\mathbf q_m`$ 表示旋转前的 `Q`、$`\widetilde{\mathbf q}_m`$ 表示旋转后的 `Q`：

```math
\begin{aligned}
\widetilde{\mathbf q}_m&=f_q(\mathbf x_m,m)\\
&=
\begin{bmatrix}
\cos(m\omega)&-\sin(m\omega)\\
\sin(m\omega)&\cos(m\omega)
\end{bmatrix}
\begin{bmatrix}
W_q^{(11)}&W_q^{(12)}\\
W_q^{(21)}&W_q^{(22)}
\end{bmatrix}
\begin{bmatrix}x_m^{(1)}\\x_m^{(2)}\end{bmatrix}\\
&=R(m\omega)\mathbf q_m,
\qquad \mathbf q_m=W_q\mathbf x_m.
\end{aligned}
```

- 从右向左计算：输入 $`\mathbf x_m`$ 先经 $`W_q`$ 投影得到 $`\mathbf q_m`$，再乘旋转矩阵得到 $`\widetilde{\mathbf q}_m`$。角频率统一记为 $`\omega`$，对应前文某一通道对的 $`\omega_i`$；$`m\omega`$ 是位置乘频率

- 对应到 `Qwen3.5`，取一个 `token`、一个 `head` 传入 `RoPE` 的 `Q` 向量 `q`（已完成投影和 `Q norm`）。设位置 $`p=1`$，`q[0]=1`、`q[32]=2`；这对通道的频率 $`\omega_0=1`$，因此：

```math
\begin{bmatrix}\widetilde q_0\\\widetilde q_{32}\end{bmatrix}
=
\begin{bmatrix}\cos 1&-\sin 1\\\sin 1&\cos 1\end{bmatrix}
\begin{bmatrix}1\\2\end{bmatrix}
\approx
\begin{bmatrix}-1.142640\\1.922076\end{bmatrix}.
```

- 结果已经是旋转后 `Q` 的第 0、32 维。其余 31 对同样计算，再接回未旋转的后 192 维，得到完整的 $`\widetilde{\mathbf q}_p=\widehat R_p\mathbf q_p\in\mathbb R^{256}`$；$`\widehat R_p`$ 的完整矩阵见下文。`K` 同理，随后用旋转后的 `Q/K` 计算 `attention` 内积。同一 `token` 的各 `head` 复用 `cos/sin`，分别旋转各自的 `Q/K` 数值

`RoFormer` 式（15）对全部 $`d`$ 维旋转。令 $`F=d/2`$、$`\phi_i=p\beta^{-2i/d}`$，以从 0 开始的索引写为：

```math
R_p^{(d)}=
\begin{pmatrix}
\cos\phi_0&-\sin\phi_0&0&0&\cdots&0&0\\
\sin\phi_0& \cos\phi_0&0&0&\cdots&0&0\\
0&0&\cos\phi_1&-\sin\phi_1&\cdots&0&0\\
0&0&\sin\phi_1& \cos\phi_1&\cdots&0&0\\
\vdots&\vdots&\vdots&\vdots&\ddots&\vdots&\vdots\\
0&0&0&0&\cdots&\cos\phi_{F-1}&-\sin\phi_{F-1}\\
0&0&0&0&\cdots&\sin\phi_{F-1}& \cos\phi_{F-1}
\end{pmatrix}
\in\mathbb R^{d\times d}.
```

- 每个 $`2\times2`$ 块作用于 `Q` 的相邻分量 $`(q_{2i},q_{2i+1})`$，`K` 同理；整体为 $`\operatorname{diag}(R(\phi_0),\ldots,R(\phi_{F-1}))`$。式（14）定义 $`\widetilde{\mathbf q}_m=R_m^{(d)}W_q\mathbf x_m`$、$`\widetilde{\mathbf k}_n=R_n^{(d)}W_k\mathbf x_n`$；记投影结果为 $`\mathbf q_m,\mathbf k_n`$，由 $`R_m^\top R_n=R_{n-m}`$ 得到相对位置内积：

```math
\begin{aligned}
\widetilde{\mathbf q}_m^\top\widetilde{\mathbf k}_n
&=(R_m\mathbf q_m)^\top(R_n\mathbf k_n)\\
&=\mathbf q_m^\top R_m^\top R_n\mathbf k_n\\
&=\mathbf q_m^\top R_{n-m}\mathbf k_n.
\end{aligned}
```

`Qwen3.5` 的 `partial RoPE` 是分块正交变换：前 64 维参与旋转，后 192 维由单位映射保留

- 令 $`D_c(p)=\operatorname{diag}(\cos\phi_{p,0},\ldots,\cos\phi_{p,31})`$、$`D_s(p)=\operatorname{diag}(\sin\phi_{p,0},\ldots,\sin\phi_{p,31})`$，其中 $`\phi_{p,i}=p\beta^{-2i/64}`$，则

```math
\begin{gathered}
\widehat R_p=
\left(
\begin{array}{cc|c}
D_c(p)&-D_s(p)&0\\
D_s(p)& D_c(p)&0\\\hline
0&0&I_{192}
\end{array}
\right),\\[6pt]
\widetilde{\mathbf q}_p=\widehat R_p\mathbf q_p,
\qquad
\widetilde{\mathbf k}_p=\widehat R_p\mathbf k_p.
\end{gathered}
```

- 左上角的 $`64\times64`$ 旋转块记为 $`\mathcal R_p`$，通过固定的通道置换与论文的块对角矩阵对应。旋转在 `attention` 内积之前分别作用于 `Q` 和 `K`，只在各自的成对通道内作线性组合，保持向量范数，不含平移项；旋转通道与保留通道之间没有交叉混合。

- 完整 `head` 在添加 `mask` 前的 `attention score` 因而分解为：

```math
s_{mn}=\frac{
(\mathbf q_m^{\mathrm{rot}})^\top \mathcal R_{n-m}\mathbf k_n^{\mathrm{rot}}
+(\mathbf q_m^{\mathrm{pass}})^\top\mathbf k_n^{\mathrm{pass}}
}{\sqrt{d_h}}.
```

- 第一项通过 $`\mathcal R_{n-m}`$ 引入相对位置，第二项是保留通道的普通内积；两项共同构成 `attention score`，缩放分母仍为 $`\sqrt{d_h}=\sqrt{256}`$。完整的 256 维 `K` 均写入 `cache`

对普通 `RoPE`，位置张量为 $`P\in\mathbb Z^{B\times T}`$，其中 $`P_{b,t}`$ 保存第 $`b`$ 个样本中第 $`t`$ 个 `token` 的位置编号。每个位置与 32 个频率分别相乘，得到该 `token` 的 32 个旋转角：

```math
\begin{gathered}
\Phi_{b,t,i}=P_{b,t}\omega_i,\qquad
\Phi\in\mathbb R^{B\times T\times(d_r/2)},\\[6pt]
C=[\cos\Phi,\cos\Phi],\quad S=[\sin\Phi,\sin\Phi].
\end{gathered}
```

- $`\Phi`$ 保存旋转角，本例形状为 `[B,T,32]`；$`C`$、$`S`$ 保存对应的 `cos/sin` 系数。方括号表示沿最后一维拼接，使 $`C/S`$ 的形状成为 `[B,T,64]`：第 $`i`$、$`i+32`$ 个通道属于同一旋转对，因而需要相同的系数。`unsqueeze(1)` 再将系数变为 `[B,1,T,64]`，供同一 `token` 的所有 `Q/K head` 复用。

- 固定一个 `token` 和一个 `head`，记 $`p=P_{b,t}`$。源码将二维旋转矩阵的乘法展开为逐元素乘加，第 $`i`$、$`i+32`$ 个通道的输出为：

```math
\begin{aligned}
\widetilde q_i&=q_i\cos(p\omega_i)-q_{i+32}\sin(p\omega_i),\\
\widetilde q_{i+32}&=q_i\sin(p\omega_i)+q_{i+32}\cos(p\omega_i),
\qquad i=0,\ldots,31.
\end{aligned}
```

- `rotate_half(q_rot)` 在这两个位置分别提供 $`-q_{i+32}`$ 和 $`q_i`$；乘以 `sin` 后，再加上 `q_rot * cos`，就得到上式。这里代码中的 `cos`、`sin` 对应广播后的 $`C`$、$`S`$，`K` 的计算相同

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed
```

## `Qwen3.5`：从一维编号到三轴位置

前面每对通道都使用同一个位置 $`p`$。图像进入语言模型后，虽然已经展平成一串 `token`，仍需要保留它在图像中的行、列关系。`MRoPE` 为每个 `token` 保存 $`(p^t,p^h,p^w)`$ 三个位置，再为每对旋转通道选择其中一个轴。二维旋转公式保持不变，只把其中的 $`p`$ 换成选中的坐标。

下文用 $`L`$ 表示语言序列长度，$`T_g,H_g,W_g`$ 表示视觉网格大小，避免把序列长度与时间轴混在一起：

```text
input_ids        [B, L]
position_ids     [3, B, L]       第 0 / 1 / 2 行为 temporal / height / width
image_grid_thw   [N_image, 3]   每张图片的 (T_g, H_g, W_g)
video_grid_thw   [N_video, 3]   每段视频的 (T_g, H_g, W_g)
```

普通文本在三轴上使用同一个编号：$`(p,p,p)`$，因而无论某对通道选哪一轴，旋转角都是 $`p\omega_i`$，严格退化为前面的一维 `RoPE`。图片内部则沿行、列编号；单张图片的 $`T_g=1`$，时间坐标在整张图内保持不变。

### `processor` 提供的是哪一种网格

`Qwen3.5` 使用 `Qwen3VLProcessor`。`image_grid_thw` 和 `video_grid_thw` 描述 **`patch embedding` 之后、空间合并之前**的网格；它们既不是原始像素尺寸，也不是已经合并好的语言模型网格。设 `spatial_merge_size = s`，则一份视觉输入对应的语言模型网格和 `token` 数为：

```math
T_{\mathrm{llm}}=T_g,\qquad
H_{\mathrm{llm}}=H_g/s,\qquad
W_{\mathrm{llm}}=W_g/s,\qquad
N_{\mathrm{vision}}=T_gH_gW_g/s^2.
```

本例 $`s=2`$，一个合并后的视觉 `token` 对应一个 $`2\times2`$ 空间 `patch` 块。`T_g` 已经包含视觉处理器的时间切块结果，`get_rope_index()` 不再把它除以 `temporal_patch_size`。`processor` 也按上述数量扩展 `<|image_pad|>` 或 `<|video_pad|>`，使占位槽位与视觉编码器输出一一对应。模型随后用视觉特征替换这些槽位的 `embedding`。[处理器实现](https://github.com/huggingface/transformers/blob/c587bc884db2c2e31fc2b8102314656b17aa07b1/src/transformers/models/qwen3_vl/processing_qwen3_vl.py#L76)

`processor` 同时返回 `[B,L]` 的 `mm_token_type_ids`：文本为 `0`，图像占位为 `1`，视频占位为 `2`。`<|vision_start|>`、`<|vision_end|>` 和视频时间戳走文本编号。这里的类型只负责识别序列区段；实际像素放在 `pixel_values`、`pixel_values_videos` 中。

### `get_rope_index()` 怎样连接文本与视觉区段

在 `Qwen3_5ForConditionalGeneration` 实例 `model` 上，方法位于 **`model.model.get_rope_index()`**，所属类是 `Qwen3_5Model`。当前接口如下，省略了类型注解：

```python
def get_rope_index(
    self,
    input_ids,
    mm_token_type_ids,
    image_grid_thw=None,
    video_grid_thw=None,
    attention_mask=None,
    **kwargs,
):
    ...  # 返回 position_ids, mrope_position_deltas
```

它根据 `attention_mask` 暂时去掉每个样本的 `padding`，然后按 `mm_token_type_ids` 的连续区段遍历序列。设当前可以分配的位置为 $`c`$：

- 长度为 $`n`$ 的文本区段，三轴都填 $`c,c+1,\ldots,c+n-1`$，然后令 $`c\leftarrow c+n`$。
- 遇到图像或视频区段，从对应的网格列表取下一行，由 `get_vision_position_ids()` 生成坐标。空间展开顺序为宽度最快、高度次之、时间最慢。
- 对 `Qwen3.5` 的单图或拆开的视频时间片，时间轴长度为 `1`。视觉区段使用 $`(c,c+h,c+w)`$，然后令 $`c\leftarrow c+\max(H_{\mathrm{llm}},W_{\mathrm{llm}})`$。

因此，**视觉区段消耗的序列槽位数是网格面积，后续位置编号推进的是网格最大边长**。它们从这里开始不再相等。[位置构造实现](https://github.com/huggingface/transformers/blob/c587bc884db2c2e31fc2b8102314656b17aa07b1/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1326)

沿用 $`s=2`$，取 `image_grid_thw = [[1,4,6]]`，得到 $`1\times2\times3`$ 的语言模型网格，共 6 个图像 `token`。图片前有两个普通文本 `token` 和一个 `<|vision_start|>`，于是视觉起点为 $`c=3`$。用 `I_hw` 标识合并后第 $`h`$ 行、第 $`w`$ 列：

| 序列下标 | 内容 | `mm_token_type_ids` | $`p^t`$ | $`p^h`$ | $`p^w`$ |
| --- | --- | --- | --- | --- | --- |
| 0 | `text_0` | 0 | 0 | 0 | 0 |
| 1 | `text_1` | 0 | 1 | 1 | 1 |
| 2 | `<\|vision_start\|>` | 0 | 2 | 2 | 2 |
| 3 | `I_00` | 1 | 3 | 3 | 3 |
| 4 | `I_01` | 1 | 3 | 3 | 4 |
| 5 | `I_02` | 1 | 3 | 3 | 5 |
| 6 | `I_10` | 1 | 3 | 4 | 3 |
| 7 | `I_11` | 1 | 3 | 4 | 4 |
| 8 | `I_12` | 1 | 3 | 4 | 5 |
| 9 | `<\|vision_end\|>` | 0 | 6 | 6 | 6 |
| 10 | `text_2` | 0 | 7 | 7 | 7 |

6 个图像 `token` 占据序列下标 `3…8`，但它们的坐标最大值只有 `5`。所以 `<|vision_end|>` 使用位置 `6`，后面的文本使用 `7`。跨行时宽度坐标会从 `5` 回到 `3`，这是二维网格展开后的正常结果，不能据此判断序列重新开始。

构造完有效位置后，函数将它们填回 `[3,B,L]`，当前实现的 `padding` 位置保留初始化值 `0`；同时返回 `[B,1]` 的 `mrope_position_deltas`。虽然函数文档把后者简写成 `[B]`，实际代码通过 `unsqueeze(1)` 增加了最后一维。这个差值怎样用于生成，后面继续用同一例子说明。

### 视频时间戳怎样进入这条链路

`Qwen3VLProcessor` 将视频展开成以下形式，每个视觉时间片前都有文本时间戳：

```text
<0.5 seconds><|vision_start|>本时间片的 video token<|vision_end|>
<1.5 seconds><|vision_start|>下一时间片的 video token<|vision_end|>
```

这些时间戳根据抽帧索引、原视频 `fps` 以及时间切块大小计算；一个时间块使用首尾帧时间的平均值，再格式化成保留一位小数的文本。`<0.5 seconds>` 会经过 `tokenizer`，不能假定它只占一个 `token`。[时间戳构造](https://github.com/huggingface/transformers/blob/c587bc884db2c2e31fc2b8102314656b17aa07b1/src/transformers/models/qwen3_vl/processing_qwen3_vl.py#L81)

为匹配这种序列，`get_rope_index()` 先按 `T_g` 重复 `video_grid_thw` 的每一行，再把每行的时间长度改成 `1`。例如 `[2,4,6]` 会变成两个 `[1,4,6]`，分别与两个视频区段对应。每个时间片内的 `p^t` 保持为该区段的起点；下一个起点由前面的文本和视觉编号继续推进。**这里的 `p^t` 不直接等于视频秒数**，实际秒数已经写在文本时间戳中。

因此，`Qwen3.5.get_rope_index()` 的有效输入是类型标记和视觉网格，它没有消费 `second_per_grids` 这样的音视频时间间隔参数。后面的 `TM-RoPE` 才会明确把秒数换成旋转所用的时间坐标。

## 三轴位置怎样变成旋转后的 `Q/K`

现在 `get_rope_index()` 已经给出了 $`P\in\mathbb Z^{3\times B\times L}`$。`Qwen3_5TextRotaryEmbedding` 仍使用前面那 32 个频率，只是先为三个轴分别计算相位：

```math
\Phi_{a,b,\ell,i}=P_{a,b,\ell}\omega_i,
\qquad a\in\{t,h,w\},\qquad
\Phi\in\mathbb R^{3\times B\times L\times32}.
```

同一个 `inv_freq[i]` 会与三个位置分别相乘。随后 `mrope_section = [11,11,10]` 决定最终 32 对通道中，11 对取时间坐标、11 对取高度坐标、10 对取宽度坐标。**这是 32 个频率的轴分配，计数单位是旋转通道对**；三轴共用原有频率序列，不是分别重新生成长度为 `11/11/10` 的频率。

当前 `Qwen3.5` 使用交错布局，其 `recomposition_frequencies()` 在已经算好的 `cos/sin` 上选择轴，再复制到两半通道：

```python
def recomposition_frequencies(self, freq):
    # freq: [3, B, L, 32]，为三个轴各自的 cos 或 sin
    freqs_thw = freq[0]
    for dim, offset in enumerate((1, 2), start=1):
        length = self.mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_thw[..., idx] = freq[dim, ..., idx]
    return torch.cat((freqs_thw, freqs_thw), dim=-1)
```

对于本例，选择结果如下：[旋转系数实现](https://github.com/huggingface/transformers/blob/c587bc884db2c2e31fc2b8102314656b17aa07b1/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L143)

| 坐标轴 | 频率下标 $`i`$ | 旋转通道对 |
| --- | --- | --- |
| `temporal` | `0,3,6,…,30` | $`(i,i+32)`$，共 11 对 |
| `height` | `1,4,7,…,31` | $`(i,i+32)`$，共 11 对 |
| `width` | `2,5,8,…,29` | $`(i,i+32)`$，共 10 对 |

这里有两个独立的布局：`T/H/W` 交错决定每个频率读取哪一个位置；前后半区配对决定旋转作用于哪两个 `Q/K` 通道。`interleaved` 不会把前面的 `rotate_half()` 改成相邻通道配对，也不是把三个角度依次旋转到同一对通道上。

取表中的 `I_12`，它的位置为 $`(3,4,5)`$。前 3 对通道实际使用的角度为：

```math
\phi_0=3\omega_0=3,\qquad
\phi_1=4\omega_1,\qquad
\phi_2=5\omega_2.
```

于是 `(q_0,q_32)` 旋转 `3` 弧度，`(q_1,q_33)` 旋转 $`4\omega_1`$，`(q_2,q_34)` 旋转 $`5\omega_2`$。余下通道对按相同规则处理。完整张量链路为：

```text
position_ids                          [3, B, L]
三轴相位 Φ                             [3, B, L, 32]
三轴 cos / sin                        [3, B, L, 32]
按频率下标选择 T / H / W                [B, L, 32]
沿最后一维复制                         [B, L, 64]
unsqueeze(1)                          [B, 1, L, 64]
旋转 Q / K 前 64 维，接回后 192 维       [B, 24 / 4, L, 256]
```

设第 $`i`$ 对通道选择轴 $`a(i)`$，两个 `token` $`m,n`$ 在该通道对中的相对角度就是：

```math
\Delta\phi_i=
\bigl(P_{a(i),n}-P_{a(i),m}\bigr)\omega_i.
```

例如 `I_00` 与 `I_01` 在时间、高度轴上的差为 `0`，宽度轴上的差为 `1`，因此只有分配给宽度的通道对出现相对位置旋转；二者内容向量仍然不同。视觉 token 的二维相邻关系由这些坐标差表达，不依赖展平后的序列下标差。

这套旋转用于 `Qwen3.5` 文本骨干中的 **`full_attention` 层**。`Qwen3_5Attention.forward()` 先完成投影和 `Q/K norm`，再调用前文的 `apply_rotary_pos_emb()`；旋转后的 `K` 与未旋转的 `V` 更新 `cache`，随后进入具体的 `attention` 实现。混合架构中的 `GatedDeltaNet` 层走自己的递推计算，不能把这里的旋转套到每一层。视觉编码器内部也有自己的二维 `RoPE`，它与此处为语言模型构造的三轴位置是两条计算路径。[层调用实现](https://github.com/huggingface/transformers/blob/c587bc884db2c2e31fc2b8102314656b17aa07b1/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L749)

## `Transformers`：从输入接口到生成续接

上面的网格、位置和旋转在一次正常调用中连成如下路径：

```text
AutoProcessor → Qwen3VLProcessor
  └─ input_ids / attention_mask / mm_token_type_ids
     pixel_values / image_grid_thw，或视频对应字段

Qwen3_5ForConditionalGeneration.forward()
  └─ model: Qwen3_5Model.forward()
      ├─ visual → 用视觉特征替换媒体占位 embedding
      ├─ compute_3d_position_ids() → get_rope_index()
      │    └─ position_ids，并保存 self.rope_deltas
      └─ language_model: Qwen3_5TextModel.forward()
           ├─ rotary_emb(hidden_states, position_ids) → cos / sin
           └─ full_attention 层 → apply_rotary_pos_emb(Q, K, cos, sin)
  └─ lm_head → logits
```

`rotary_emb` 在文本模型入口计算一次当前输入的 `cos/sin`，各个 `full_attention` 层复用这些系数，但旋转各层自己投影得到的 `Q/K`。`get_rope_index()` 只构造位置，既不运行视觉编码器，也不直接计算 `cos/sin`。[模型入口](https://github.com/huggingface/transformers/blob/c587bc884db2c2e31fc2b8102314656b17aa07b1/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1544)

### 自动构造与显式传入

以下是普通图文生成接口。图片路径替换为实际文件；`processor` 负责返回配套字段，无需手工数图像 `token`：

```python
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

model_id = "Qwen/Qwen3.5-27B"
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    model_id, dtype="auto", device_map="auto"
)

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "/path/to/image.jpg"},
        {"type": "text", "text": "描述这张图片。"},
    ],
}]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=128)
```

直接调用 `model(**inputs)` 且未传 `position_ids` 时，`Qwen3_5Model.forward()` 会自动构造三轴位置。当前版本若收到视觉网格和 `input_ids`，却缺失 `mm_token_type_ids`，会报错；不能在整理处理器输出时把它丢掉。

需要检查具体编号时，可以只调用位置函数：

```python
position_ids, rope_deltas = model.model.get_rope_index(
    input_ids=inputs["input_ids"],
    mm_token_type_ids=inputs["mm_token_type_ids"],
    image_grid_thw=inputs.get("image_grid_thw"),
    video_grid_thw=inputs.get("video_grid_thw"),
    attention_mask=inputs.get("attention_mask"),
)
# position_ids: [3, B, L]
# rope_deltas:  [B, 1]
```

`forward()` 也接受显式的 `position_ids`。一旦传入，就跳过自动位置构造；调用 `get_rope_index()` 本身不会替调用者保存 `rope_deltas`。因此，手写缓存解码时，位置张量、差值和对应的 `past_key_values` 必须作为同一条序列的状态一起维护。标准 `generate()` 会处理这些步骤。

### 为什么同时会见到 `[B,L]`、`[3,B,L]` 和 `[4,B,L]`

`Qwen3_5TextModel.forward()` 对位置张量的处理如下：

| 传入形状 | 含义与处理 |
| --- | --- |
| `[B,L]` | 一维位置；复制成四行，供普通文本使用 |
| `[3,B,L]` | 已构造好的 `T/H/W`；三行全部用于旋转 |
| `[4,B,L]` | 第 0 行是线性 `text_position_ids`，后三行是 `T/H/W` |

对于四行形式，文本模型先取出 `position_ids[0]` 供遮罩构造使用，再把 `position_ids[1:]` 送入 `rotary_emb`。**第四行结构没有引入第四个旋转轴**。它让序列顺序与三轴空间位置分别参与各自的计算。[文本模型实现](https://github.com/huggingface/transformers/blob/c587bc884db2c2e31fc2b8102314656b17aa07b1/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1240)

三行形式不会从 `T/H/W` 中另取一行当作线性位置；正常的因果遮罩仍依据输入序列、缓存和 `attention_mask` 构造。前面图片的宽度编号可以回退，注意力的先后顺序仍由展平后的序列决定。

`generate()` 在首次准备输入时还有一条入口：`Qwen3_5ForConditionalGeneration._prepare_position_ids_for_generation()`。它先生成一维文本位置，再调用 `model.get_rope_index()` 得到三轴位置，拼成 `[4,B,L]`，并保存 `model.rope_deltas`。这时后续 `forward()` 收到的已经是完整位置，无需重新构造。[生成入口](https://github.com/huggingface/transformers/blob/c587bc884db2c2e31fc2b8102314656b17aa07b1/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1885)

### `rope_deltas` 如何接上生成位置

设一个样本去除 `padding` 后有 $`N`$ 个输入 `token`，三轴已分配位置的最大值为 $`P_{\max}`$。当前位置函数返回：

```math
\delta=P_{\max}+1-N.
```

继续生成普通文本时，第 $`j`$ 个新增位置从 $`j=0`$ 开始，三轴均使用：

```math
p^t_j=p^h_j=p^w_j=N+j+\delta=P_{\max}+1+j.
```

回到前面的图片例子：$`N=11`$，$`P_{\max}=7`$，所以 $`\delta=-3`$。下一个进入模型的文本 `token` 在序列中的下标是 `11`，但它的旋转位置为 $`11-3=8`$，之后为 `9,10,…`。缓存仍保存前面全部 11 个槽位，负的差值不会压缩或删除 `KV cache`。

批处理中的 $`N`$ 是有效长度，不是补齐后的共同宽度；有 `padding` 时，一维有效位置由 `attention_mask.cumsum(-1)-1` 得到。标准生成使用左侧补齐，既保留正确的有效位置，也让各样本最后一列是生成所接续的有效输入。`rope_deltas` 是每个样本各自的 `[B,1]` 偏移。

在同一次 `generate()` 中，通用生成循环会把已有 `position_ids` 的末项逐次加一；缓存中的差值用于需要按一维位置恢复多模态续接位置的入口。不能把这两步同时理解成每生成一个 `token` 都重新累加一次 `delta`。[生成循环中的位置更新](https://github.com/huggingface/transformers/blob/c587bc884db2c2e31fc2b8102314656b17aa07b1/src/transformers/generation/utils.py#L1106)

## `TM-RoPE`：把音频与视频放到同一时间轴

到这里，一条图文序列已经完成了从网格到 `Q/K` 的旋转。加入音频后，还需要解决另一个问题：音频 `token` 的序号与视频时间片的序号具有不同的时间尺度。`TM-RoPE`（`Time-aligned Multimodal RoPE`）把它们换算到共同的时间坐标，然后仍通过三轴位置、频率选择和二维旋转进入 `attention`。

以下采用当前 `Transformers` 已公开实现的 **`Qwen3-Omni-30B-A3B-Instruct` 的 `Thinker` 输入链路**。该检查点的 `position_id_per_seconds = 13`、`audio_config.n_window = 50`，均从实际配置读取；不使用配置类中未加载检查点时的默认时间粒度。[检查点配置](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/blob/main/config.json)

更新的 `Qwen3.5-Omni` 公开论文描述了显式音视频时间戳和 6.25 Hz 音频编码器，但本次核对的 `Transformers` 主干尚无对应模型实现。因此，本节的类和参数明确属于 `Qwen3-Omni`；6.25 Hz 的编码器输出率不能直接替换成这个检查点的 `position_id_per_seconds`。[`Qwen3.5-Omni` 论文 §2.2–2.3](https://arxiv.org/html/2604.15804v1#S2.SS2)

### 时间坐标与音频长度

设音视频有效内容的共同起点为 $`c`$，时间换算率为 $`\rho=13`$ 个位置单位每秒，视频相邻时间网格的间隔为 $`\Delta\tau`$ 秒。第 $`k`$ 个视频时间片、第 $`h`$ 行、第 $`w`$ 列的位置为：

```math
\mathbf p^{\mathrm{video}}_{k,h,w}
=\bigl(c+k\Delta\tau\rho,\ c+h,\ c+w\bigr).
```

相应的第 $`j`$ 个音频 `token` 在三个轴上使用相同编号：

```math
\mathbf p^{\mathrm{audio}}_j=(c+j,\ c+j,\ c+j).
```

视频时间轴因此与音频编号使用同一尺度；音频自身的三个分量相同，仍按一维方式旋转。对任意一对通道，旋转角始终等于“所选坐标 × `inv_freq[i]`”，不会因为坐标来自秒数而换一套旋转公式。

这里的 `audio_seqlens` 必须是 **有效声学特征帧数**，通常来自 `feature_attention_mask.sum(-1)`。它不是波形采样点数，也不是已经进入语言模型的音频 `token` 数。`Qwen3-Omni` 在模型和处理器中都使用 `_get_feat_extract_output_lengths()` 换算长度。

以检查点的 `n_window=50` 为例，每个完整块包含 100 个声学特征帧，产生 13 个音频 `token`。设有效帧数为 $`F`$，剩余帧数为 $`r=F\bmod100`$，源码中的整数运算可写成：

```math
N_{\mathrm{audio}}
=13\left\lfloor\frac{F}{100}\right\rfloor
+\left\lceil\frac{r}{8}\right\rceil.
```

所以 `F=100` 得到 13 个 `token`，`F=101` 得到 14 个，`F=200` 得到 26 个。长度必须按编码器的分块和下采样规则计算，不能直接用 `F/8` 或音频秒数取整代替。[长度函数与位置实现](https://github.com/huggingface/transformers/blob/c587bc884db2c2e31fc2b8102314656b17aa07b1/src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py#L152)

处理器输出的 `video_second_per_grid` 是视频相邻**时间网格**的秒数间隔。其当前构造为 `temporal_patch_size / fps`，这里的 `fps` 必须与送入处理器的视频采样一致；时间切块大小已经计入这个间隔。模型收到它后，传给位置函数的参数名变为 `second_per_grids`。

当前 `Qwen3-Omni` 的多模态 `position_ids` 使用浮点数，视频时间坐标也保留小数。例如 $`\Delta\tau=0.5`$ 时，相邻时间片相差 $`0.5\times13=6.5`$ 个位置单位。即使部分类型注解仍写着 `LongTensor`，也不能据此把位置强制转换为整数。[浮点位置构造](https://github.com/huggingface/transformers/blob/c587bc884db2c2e31fc2b8102314656b17aa07b1/src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py#L297)

### 同一段音视频怎样交错进入序列

时间对齐还必须体现在占位槽位的顺序上。启用 `use_audio_in_video=True` 后，当前处理器先分别生成视频时间坐标与音频编号，再按时间归并；相同时间下，视频 `token` 排在音频 `token` 前面。`get_rope_index()` 使用相同规则归并位置，确保第 $`\ell`$ 个位置对应第 $`\ell`$ 个媒体槽位。[处理器的归并实现](https://github.com/huggingface/transformers/blob/c587bc884db2c2e31fc2b8102314656b17aa07b1/src/transformers/models/qwen3_omni_moe/processing_qwen3_omni_moe.py#L247)

继续使用 $`2\times3`$ 的空间网格，取两个视频时间片 `video_grid_thw=[[2,4,6]]`，间隔为 `0.5` 秒；配套音频有 `100` 个有效特征帧，产生 `13` 个音频 `token`。内容前放一个文本 `token`、`<|vision_start|>` 和 `<|audio_start|>`，三者位置依次为 `0,1,2`，共同内容起点即 $`c=3`$：

| 内容 | 数量 | 时间坐标 $`p^t`$ | 高度、宽度坐标 |
| --- | --- | --- | --- |
| 第 0 个视频时间片 | 6 | `3` | 高度 `3,4`；宽度 `3,4,5` |
| 音频 `A_0…A_6` | 7 | `3…9` | 与各自时间坐标相同 |
| 第 1 个视频时间片 | 6 | `9.5` | 高度 `3,4`；宽度 `3,4,5` |
| 音频 `A_7…A_12` | 6 | `10…15` | 与各自时间坐标相同 |
| `<\|audio_end\|>`、`<\|vision_end\|>` | 2 | `16,17` | 三轴相同 |
| 后续文本 | 1 | `18` | 三轴相同 |

最终序列顺序为：

```text
text → vision_start → audio_start
     → V_0 的 6 个 token → A_0…A_6
     → V_1 的 6 个 token → A_7…A_12
     → audio_end → vision_end → text
```

`V_0` 与 `A_0` 都从时间位置 `3` 开始；第二个视频时间片的 `9.5` 正好位于 `A_6` 的 `9` 与 `A_7` 的 `10` 之间。序列共有 31 个 `token`，最大三轴位置为 `18`，因此返回的 `rope_delta=19-31=-12`，下一个普通文本位置为 `19`。与图片例子相同，序列槽位数和位置坐标范围分别计数。

这个归并描述的是当前 `Qwen3-Omni` 的实际路径。虽然处理器的参数定义中仍保留 `seconds_per_chunk`，上述归并循环按每个 `token` 的时间比较，不能由参数名称推断成每两秒把整段视频与整段音频拼一次。

### `Thinker` 接口如何接收这些字段

对于 `Qwen3OmniMoeForConditionalGeneration` 实例 `omni`，输入理解由 `omni.thinker` 完成。位置函数位于 **`omni.thinker.get_rope_index()`**，其参数与 `Qwen3.5` 不同：

```python
def get_rope_index(
    self,
    input_ids=None,
    image_grid_thw=None,
    video_grid_thw=None,
    attention_mask=None,
    use_audio_in_video=False,
    audio_seqlens=None,
    second_per_grids=None,
):
    ...  # 返回 [3, B, L] 的位置，以及 [B, 1] 的差值
```

它根据 `input_ids` 中的媒体起止标记和占位 `token` 识别区段，不接收 `Qwen3.5` 使用的 `mm_token_type_ids`。与位置有关的处理器字段、`Thinker.forward()` 参数和位置函数参数对应如下：

| 处理器输出 / 调用设置 | `Thinker.forward()` | `get_rope_index()` 中的用途 |
| --- | --- | --- |
| `input_ids`、`attention_mask` | 同名 | 确定有效序列和媒体区段 |
| `image_grid_thw`、`video_grid_thw` | 同名 | 生成空间坐标及视频时间片 |
| `feature_attention_mask` | 同名；求和得到 `audio_feature_lengths` | 作为 `audio_seqlens` 传入，再换算音频 `token` 数 |
| `video_second_per_grid` | 同名 | 作为 `second_per_grids` 传入，换算时间坐标 |
| `use_audio_in_video=True` | 同名 | 按同一段音视频的归并规则构造位置 |
| `input_features`、`pixel_values_videos` | 同名 | 进入各自编码器，特征填入媒体槽位；不直接参与位置函数 |

处理器也应从同一检查点加载。设 `chat_text` 是 `processor.apply_chat_template(..., tokenize=False)` 渲染的视频消息，`videos` 和 `audios` 是按消息顺序读取的视频及对应的 16 kHz 音轨，`sample_fps` 与视频采样一致，则接口可以连接为：

```python
from transformers import Qwen3OmniMoeProcessor

processor = Qwen3OmniMoeProcessor.from_pretrained(
    "Qwen/Qwen3-Omni-30B-A3B-Instruct"
)
# omni 已从同一检查点加载，为 Qwen3OmniMoeForConditionalGeneration
batch = processor(
    text=chat_text,
    videos=videos,
    audio=audios,
    fps=sample_fps,
    use_audio_in_video=True,
    position_id_per_seconds=omni.thinker.config.position_id_per_seconds,
    n_window=omni.thinker.config.audio_config.n_window,
    padding=True,
    return_tensors="pt",
).to(omni.device)
```

得到 `batch` 后，可单独查看它对应的位置：

```python
thinker = omni.thinker
audio_feature_lengths = batch["feature_attention_mask"].sum(-1)

position_ids, rope_deltas = thinker.get_rope_index(
    input_ids=batch["input_ids"],
    image_grid_thw=batch.get("image_grid_thw"),
    video_grid_thw=batch.get("video_grid_thw"),
    attention_mask=batch["attention_mask"],
    use_audio_in_video=True,
    audio_seqlens=audio_feature_lengths,
    second_per_grids=batch.get("video_second_per_grid"),
)
```

正常推理无需手工调用这个函数。`thinker(**batch, use_audio_in_video=True)` 会在 `position_ids` 缺省时自动构造；整体模型的文本生成入口则是：

```python
generated_ids = omni.generate(
    **batch,
    use_audio_in_video=True,
    return_audio=False,
    thinker_max_new_tokens=128,
)
```

处理器和模型两端的 `use_audio_in_video` 必须一致，前者决定槽位如何交错，后者决定位置如何交错。该选项不会自动把一份没有音频特征的视频输入变成带音轨输入；处理器必须同时收到配套音频。时间换算率也应与 `omni.thinker.config.position_id_per_seconds` 一致。[`Thinker` 的自动构造入口](https://github.com/huggingface/transformers/blob/c587bc884db2c2e31fc2b8102314656b17aa07b1/src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py#L1997)

位置进入 `Qwen3OmniMoeThinkerTextRotaryEmbedding` 后，再乘频率并选择三轴。该检查点 `head_dim=128`、`rope_theta=1000000`、`mrope_section=[24,20,20]`，64 对通道覆盖整个 `head`；高度取频率下标 `1,4,…,58`，宽度取 `2,5,…,59`，其余 24 对取时间轴。最后复制为 `[B,L,128]` 的 `cos/sin`，以 `(i,i+64)` 配对旋转 `Q/K`。这一步与前面的三轴旋转遵循同一公式，新增的时间对齐已经体现在 `position_ids` 和序列排列中。[`Thinker` 旋转实现](https://github.com/huggingface/transformers/blob/c587bc884db2c2e31fc2b8102314656b17aa07b1/src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py#L1242)

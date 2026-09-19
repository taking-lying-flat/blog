# Interleaved Rotary Position Embedding

## `RoPE`：旋转位置编码

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
head_dim = (
    getattr(config, "head_dim", None)
    or config.hidden_size // config.num_attention_heads
)
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

- 从右向左计算：输入 $`\mathbf x_m`$ 先经 $`W_q`$ 投影得到 $`\mathbf q_m`$，再乘旋转矩阵得到 $`\widetilde{\mathbf q}_m`$。角频率统一记为 $`\omega`$，对应某一通道对的 $`\omega_i`$；$`m\omega`$ 是位置乘频率

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

- 第一项通过 $`\mathcal R_{n-m}`$ 引入相对位置，第二项是保留通道的普通内积；两项共同构成 `attention score`。完整的 256 维 `K` 均写入 `cache`

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

- `rotate_half(q_rot)` 在这两个位置分别提供 $`-q_{i+32}`$ 和 $`q_i`$；乘以 `sin` 后，再加上 `q_rot * cos`，就得到上式。`cos`、`sin` 对应广播后的 $`C`$、$`S`$，`K` 的计算相同

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

## `Qwen3.5`：文本、图片与视频的 `MRoPE` 流程

取 `Qwen3.5-27B`、`bs=1`，一条已分词的输入包含两个普通文本 token、一张图片及其起止标记。设 `image_grid_thw=[[1,4,6]]`、`spatial_merge_size=2`，空间合并后是 `1×2×3` 网格，对应六个图片槽位，整条序列长度 `L=10`，没有批处理补齐。

1. **对齐内容与位置：写出三轴位置表**。用 `t0/t1` 表示两个文本 token，`S/E` 表示 `<|vision_start|>/<|vision_end|>`，`Ihw` 表示图片第 `h` 行、第 `w` 列的槽位。六个 `Ihw` 的词表编号都是同一个 `<|image_pad|>`，填入的视觉特征和三轴坐标各不相同。

   | 序列槽位 `l` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
   | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
   | token | `t0` | `S` | `I00` | `I01` | `I02` | `I10` | `I11` | `I12` | `E` | `t1` |
   | `mm_token_type_ids[0,l]` | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
   | `T = position_ids[0,0,l]` | 0 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 5 | 6 |
   | `H = position_ids[1,0,l]` | 0 | 1 | 2 | 2 | 2 | 3 | 3 | 3 | 5 | 6 |
   | `W = position_ids[2,0,l]` | 0 | 1 | 2 | 3 | 4 | 2 | 3 | 4 | 5 | 6 |

   `get_rope_index()` 先给 `t0/S` 分配 `0/1`，图片区段从 `2` 开始。图片只有一个时间片，`T` 全为 `2`；两行的 `H` 为 `2/3`，三列的 `W` 为 `2/3/4`，按宽度最快的顺序展平。图片区段占用的旋转位置跨度为 `max(2,3)=3`，因此 `E/t1` 从 `5/6` 继续。

   `inputs_embeds` 为 `[1,10,5120]`，`position_ids` 为 `[3,1,10]`。例如 `inputs_embeds[0,7,:]` 是 `I12` 的内容向量，`position_ids[:,0,7]=[2,3,4]` 是它的旋转坐标。**位置表提供旋转角公式中的位置因子**；三行同一列对应同一个 token。固定批次下标 `0`，将表记为 $`P_{a,l}=\text{position\_ids}[a,0,l]`$，轴 $`a=0,1,2`$ 分别表示 `T/H/W`。

2. **位置乘频率：为每个 token 计算三组候选系数**。旋转维度为 `64`，共 `32` 对通道；第 `i` 对通道使用 `inv_freq[i]`。同一个频率分别乘该 token 的三个坐标：

   ```math
   \begin{aligned}
   \omega_i&=\text{inv\_freq}[i]=10^{-7i/32},\qquad i=0,\ldots,31,\\
   \Phi_{a,l,i}&=P_{a,l}\omega_i,\\
   C^{\mathrm{axis}}_{a,l,i}&=\cos(P_{a,l}\omega_i),\\
   S^{\mathrm{axis}}_{a,l,i}&=\sin(P_{a,l}\omega_i).
   \end{aligned}
   ```

   `Qwen3_5TextRotaryEmbedding.forward()` 将位置扩展为 `[3,1,1,10]`，将频率扩展为 `[3,1,32,1]`，矩阵乘法再转置得到 `freqs: [3,1,10,32]`。对每个轴、每个 token，都计算全部 `32` 个频率的角度；默认配置的 `attention_scaling=1`，求 `cos/sin` 后形状不变。取 `I12`，其坐标 `[2,3,4]` 产生：

   | 候选角度 | `i=0` | `i=1` | `i=2` | … | `i=31` |
   | --- | --- | --- | --- | --- | --- |
   | `T` | $`2\omega_0`$ | $`2\omega_1`$ | $`2\omega_2`$ | … | $`2\omega_{31}`$ |
   | `H` | $`3\omega_0`$ | $`3\omega_1`$ | $`3\omega_2`$ | … | $`3\omega_{31}`$ |
   | `W` | $`4\omega_0`$ | $`4\omega_1`$ | $`4\omega_2`$ | … | $`4\omega_{31}`$ |

3. **按通道对交错选轴：每一列只取一个候选**。`mrope_section=[11,11,10]` 将 `32` 对通道分给 `T/H/W`，读取顺序为 `T,H,W,T,H,W,…,T,H`。对这组配置，轴下标为 $`a(i)=i\bmod3`$；第 `l` 个 token、第 `i` 对通道实际采用的角度是：

   ```math
   \varphi_{l,i}=P_{a(i),l}\omega_i.
   ```

   `recomposition_frequencies()` 实际选择的是已经算好的 `cos/sin`：频率下标 `0` 取 `T` 行，下标 `1` 取 `H` 行，下标 `2` 取 `W` 行，依此类推。对 `I12`，被选中的角度依次为：

   ```math
   \varphi_{7,:}=
   [2\omega_0,\ 3\omega_1,\ 4\omega_2,\ 2\omega_3,\ 3\omega_4,\ 4\omega_5,
   \ldots,\ 2\omega_{30},\ 3\omega_{31}].
   ```

   **交错发生在频率／通道对这一维**：同一个 token 的不同通道对分别读取 `T/H/W`；每对通道只采用一个坐标。所有 token 使用相同的选轴规则，所读的坐标数值由各自在位置表中的列决定。

4. **复制系数并旋转 Q/K：一对通道共用一个角度**。选轴后，`cos/sin` 各为 `[1,10,32]`；`torch.cat((freqs_thw, freqs_thw), dim=-1)` 复制成 `[1,10,64]`。记复制后的系数为 $`C/S`$，固定批次下标后：

   ```math
   \begin{aligned}
   C_{l,i}=C_{l,i+32}&=\cos\varphi_{l,i},\\
   S_{l,i}=S_{l,i+32}&=\sin\varphi_{l,i},
   \qquad i=0,\ldots,31.
   \end{aligned}
   ```

   `Qwen3_5Attention.forward()` 在投影和归一化后得到 `Q: [1,24,10,256]`、`K: [1,4,10,256]`，再调用 `apply_rotary_pos_emb()`。系数经 `unsqueeze(1)` 变为 `[1,1,10,64]`，广播到所有头；`Q/K` 的前 `64` 维按 `(0,32)、(1,33)、…、(31,63)` 配对旋转。对一个 token、一个头，`rotate_half()` 在通道 `i/i+32` 上分别提供 `-q[i+32]/q[i]`，因而逐元素乘加等价于：

   ```math
   \begin{gathered}
   R(\varphi)=
   \begin{bmatrix}
   \cos\varphi&-\sin\varphi\\
   \sin\varphi&\cos\varphi
   \end{bmatrix},\\[6pt]
   \begin{bmatrix}\widetilde q_{l,i}\\\widetilde q_{l,i+32}\end{bmatrix}
   =R(\varphi_{l,i})
   \begin{bmatrix}q_{l,i}\\q_{l,i+32}\end{bmatrix},\qquad
   \begin{bmatrix}\widetilde k_{l,i}\\\widetilde k_{l,i+32}\end{bmatrix}
   =R(\varphi_{l,i})
   \begin{bmatrix}k_{l,i}\\k_{l,i+32}\end{bmatrix}.
   \end{gathered}
   ```

   具体取 `I12` 的通道对 `(2,34)`：`i=2` 读取 `W` 坐标 `4`，所以角度为 $`\varphi_{7,2}=4\omega_2`$，其中 $`\omega_2=10^{-7/16}`$。省略固定的 token 和头下标，这一对 `Q/K` 的输出就是：

   ```math
   \begin{aligned}
   \widetilde q_2&=q_2\cos(4\omega_2)-q_{34}\sin(4\omega_2),\\
   \widetilde q_{34}&=q_2\sin(4\omega_2)+q_{34}\cos(4\omega_2),\\
   \widetilde k_2&=k_2\cos(4\omega_2)-k_{34}\sin(4\omega_2),\\
   \widetilde k_{34}&=k_2\sin(4\omega_2)+k_{34}\cos(4\omega_2).
   \end{aligned}
   ```

   其余通道对分别代入各自选中的坐标和频率。后 `192` 维原样接回，`V` 不旋转；注意力使用旋转后的 `Q/K`。文本 token 的三轴坐标相同，例如 `t1` 为 `[6,6,6]`，无论选哪个轴，角度都是 $`6\omega_i`$，得到普通一维 RoPE。

### 混合序列与媒体字段

- **混合序列与 `mm tokens`**：文本、图片和视频共用一条序列。图片对应连续的 `<|image_pad|>`；视频按时间片展开，每片带文本时间戳和连续的 `<|video_pad|>`。`mm tokens` 指这些媒体占位 token，`Qwen3.5` 没有独立的 `mm_tokens` 输入参数：占位的词表编号保存在 `input_ids`，每个槽位的媒体类型保存在 `mm_token_type_ids`，媒体内容保存在 `pixel_values` 或 `pixel_values_videos`。

  ```text
  文本 → <|vision_start|> image_pad ... image_pad <|vision_end|> → 文本
       → <|vision_start|>
         <时间戳> <|vision_start|> video_pad ... video_pad <|vision_end|>
         <时间戳> <|vision_start|> video_pad ... video_pad <|vision_end|>
       → <|vision_end|> → 文本
  ```

- **字段与形状**：`B/L/D` 分别表示批大小、补齐后的序列长度和文本隐藏维度；`N_image/N_video` 是整个批次中的图片、视频数量。各字段分别保存：

  | 字段 | 结构 | 内容与用途 |
  | --- | --- | --- |
  | `input_ids` | `[B,L]` | 文本、媒体起止标记和媒体占位 token 的词表编号 |
  | `mm_token_type_ids` | `[B,L]` | 与 `input_ids` 逐槽位对应：文本 `0`、图片 `1`、视频 `2` |
  | `attention_mask` | `[B,L]` | 有效槽位为 `1`，批处理补齐位置为 `0` |
  | `image_grid_thw` | `[N_image,3]` | 每张图片的 `(T_g,H_g,W_g)` 视觉网格 |
  | `video_grid_thw` | `[N_video,3]` | 每段视频的 `(T_g,H_g,W_g)` 视觉网格 |
  | `pixel_values` | `[ΣP_image,patch_dim]` | 图片的展平 patch 数据 |
  | `pixel_values_videos` | `[ΣP_video,patch_dim]` | 视频的展平时空 patch 数据 |
  | `inputs_embeds` | `[B,L,D]` | 文本嵌入与视觉特征合并后的语言模型输入 |
  | `position_ids` | `[3,B,L]` | `get_rope_index()` 为各槽位生成的 `T/H/W` 旋转坐标 |

- **媒体槽位怎样展开**：`processor.apply_chat_template(..., tokenize=True, return_dict=True)` 先按消息中的 `type` 写入媒体标记，再交给 `Qwen3VLProcessor` 处理。`replace_image_token()` 按 `T_g×H_g×W_g / spatial_merge_size²` 扩展图片占位；`replace_video_token()` 为每个时间片写入时间戳和 `H_g×W_g / spatial_merge_size²` 个视频占位。网格是空间合并前的大小；`ΣP` 是展平 patch 的总数，`patch_dim` 是一个 patch 展平后的向量长度。

  `Qwen3VLProcessor.replace_image_token()` 的展开语句为：

  ```python
  merge_length = self.image_processor.merge_size**2
  num_image_tokens = image_inputs["image_grid_thw"][image_idx].prod() // merge_length
  return self.image_token * num_image_tokens
  ```

- **视频时间片怎样展开**：`replace_video_token()` 用时间网格长度作为 `num_frames`；一个时间片可以包含多个原始帧。时间戳由原始帧编号、帧率和 `temporal_patch_size` 计算，作为普通文本分词；每个时间片的媒体占位前后另加起止标记：

  ```python
  merge_length = self.video_processor.merge_size**2
  num_frames = video_inputs["video_grid_thw"][video_idx][0]
  frame_seqlen = video_inputs["video_grid_thw"][video_idx][1:].prod() // merge_length
  metadata = video_inputs["video_metadata"][video_idx]
  curr_timestamp = self._calculate_timestamps(
      metadata.frames_indices,
      metadata.fps,
      self.video_processor.temporal_patch_size,
  )
  video_placeholder = ""
  for frame_idx in range(num_frames):
      curr_time = curr_timestamp[frame_idx]
      video_placeholder += f"<{curr_time:.1f} seconds>"
      video_placeholder += (
          self.vision_start_token
          + self.video_token * frame_seqlen
          + self.vision_end_token
      )
  ```

- **媒体类型怎样记录**：分词后，`create_mm_token_type_ids()` 按词表编号标记媒体槽位。所有图片占位共享同一个 `image_token_id`，所有视频占位共享同一个 `video_token_id`；槽位的空间差异由三维位置表记录：

  ```python
  tokenizer_input = np.array(tokenizer_input)
  mm_token_types = np.zeros_like(tokenizer_input)
  mm_token_types[np.isin(tokenizer_input, self.image_token_ids)] = 1
  mm_token_types[np.isin(tokenizer_input, self.video_token_ids)] = 2
  ```

- **区分媒体占位和补齐**：`<|image_pad|>`、`<|video_pad|>` 是有效媒体槽位，不是 padding 或 EOS token，`attention_mask` 为 `1`。`Qwen3.5-27B` 的 padding token 是 `<|endoftext|>`，tokenizer 的 EOS 是 `<|im_end|>`；生成配置把两者都列为停止 token。时间戳、消息标记和 `<|vision_start|>/<|vision_end|>` 的媒体类型均为 `0`，按文本位置规则处理。

- **条件生成模型的入口**：`Qwen3_5ForConditionalGeneration.forward()` 将混合输入和位置参数传给 `self.model`，即 `Qwen3_5Model`。位置表沿 `position_ids` 参数传递，视觉数据和类型标记各有独立参数；该层的实际调用为：

  ```python
  outputs = self.model(
      input_ids=input_ids,
      pixel_values=pixel_values,
      pixel_values_videos=pixel_values_videos,
      image_grid_thw=image_grid_thw,
      video_grid_thw=video_grid_thw,
      position_ids=position_ids,
      attention_mask=attention_mask,
      past_key_values=past_key_values,
      inputs_embeds=inputs_embeds,
      mm_token_type_ids=mm_token_type_ids,
      **kwargs,
  )
  ```

- **`special_*_mask` 只定位视觉特征的填充位置**：`Qwen3_5Model.get_placeholder_mask()` 在 `input_ids` 路径中比较特殊 token 编号，先得到 `[B,L]` 布尔表，再扩展为 `[B,L,1]`。最后一维广播到隐藏维度 `D`，使一个媒体槽位的整条向量被替换。返回的两个掩码分别由调用方接收为 `image_mask` 和 `video_mask`，定位与扩展的关键语句为：

  ```python
  special_image_mask = input_ids == self.config.image_token_id
  special_video_mask = input_ids == self.config.video_token_id
  special_image_mask = special_image_mask.unsqueeze(-1).to(inputs_embeds.device)
  special_video_mask = special_video_mask.unsqueeze(-1).to(inputs_embeds.device)
  return special_image_mask, special_video_mask
  ```

- **填入内容，保留槽位顺序**：`Qwen3_5Model.forward()` 先计算 `inputs_embeds`，再处理视觉分支。图片分支如下，视频分支以同样方式填充 `video_mask` 指定的槽位。`masked_scatter()` 只替换内容，`L` 和 token 顺序不变；随后生成的位置表仍与这些槽位逐列对应：

  ```python
  if inputs_embeds is None:
      inputs_embeds = self.get_input_embeddings()(input_ids)

  if pixel_values is not None:
      image_outputs = self.get_image_features(
          pixel_values, image_grid_thw, return_dict=True, **kwargs
      )
      image_embeds = image_outputs.pooler_output
      image_embeds = torch.cat(image_embeds, dim=0).to(
          inputs_embeds.device, inputs_embeds.dtype
      )
      image_mask, _ = self.get_placeholder_mask(
          input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
      )
      inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
  ```

### `get_rope_index()`：生成三维位置表

- **接口与输出**：对 `Qwen3_5ForConditionalGeneration` 实例 `model`，调用 `model.model.get_rope_index()`；返回 `position_ids: [3,B,L]` 和 `rope_deltas: [B,1]`。`position_ids[:,b,l]` 是一个槽位的 `T/H/W` 三个坐标；函数只读取类型和网格，不读取像素：

  ```python
  position_ids, rope_deltas = model.model.get_rope_index(
      input_ids=input_ids, mm_token_type_ids=mm_token_type_ids,
      image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
      attention_mask=attention_mask,
  )
  ```

- **准备网格与输出**：视频已被时间戳分成多个区段，`repeat_interleave()` 按每段视频的时间网格长度复制网格，再把每行的时间长度设为 `1`。图片、视频各有一个由整个批次共用的网格迭代器，按样本及媒体出现顺序消费；输出位置表先初始化为 `[3,B,L]`：

  ```python
  if video_grid_thw is not None:
      video_grid_thw = torch.repeat_interleave(
          video_grid_thw, video_grid_thw[:, 0], dim=0
      )
      video_grid_thw[:, 0] = 1
  spatial_merge_size = self.config.vision_config.spatial_merge_size
  mrope_position_deltas = []
  position_ids = torch.zeros(
      3, input_ids.shape[0], input_ids.shape[1],
      dtype=input_ids.dtype, device=input_ids.device,
  )
  grid_iters = {
      1: iter(image_grid_thw) if image_grid_thw is not None else None,
      2: iter(video_grid_thw) if video_grid_thw is not None else None,
  }
  ```

- **过滤补齐并划分区段**：对每个样本，用同一份 `attention_mask` 同时过滤 `input_ids` 和 `mm_token_type_ids`，保持逐位置对应。`groupby()` 只合并连续相同的类型；两个图片区段即使都是 `1`，中间的起止标记为 `0`，仍会分开。`start_index/end_index` 是过滤后序列的左闭右开区间：

  ```python
  for batch_idx, current_input_ids in enumerate(input_ids):
      input_token_type = mm_token_type_ids[batch_idx]
      if attention_mask is not None:
          current_input_ids = current_input_ids[
              attention_mask[batch_idx].bool()
          ]
          input_token_type = input_token_type[
              attention_mask[batch_idx].bool()
          ]

      input_type_group = []
      for key, group in itertools.groupby(
          enumerate(input_token_type.tolist()), lambda x: x[1]
      ):
          group = list(group)
          start_index = group[0][0]
          end_index = group[-1][0] + 1
          input_type_group.append((key, start_index, end_index))
  ```

- **推进位置**：在上述批次循环内，为每个样本从 `current_pos=0` 开始分配位置。`start_idx/end_idx` 表示序列区间，`current_pos` 表示旋转位置起点。类型 `0` 将连续编号复制到三行；类型 `1/2` 分别取下一个图片网格或视频时间片网格，生成 `[3,区段长度]` 坐标。视觉区段结束后，按合并网格的最大边长推进，不能用视觉 `token` 总数代替：

  ```python
  current_pos = 0
  llm_pos_ids_list = []
  for modality_type, start_idx, end_idx in input_type_group:
      if modality_type == 0:
          text_len = end_idx - start_idx
          llm_pos_ids_list.append(
              torch.arange(text_len, device=input_ids.device)
              .view(1, -1)
              .expand(3, -1)
              + current_pos
          )
          current_pos += text_len
      else:
          grid_thw = next(grid_iters[modality_type])
          vision_position_ids = self.get_vision_position_ids(
              current_pos, grid_thw, 1, spatial_merge_size, device=input_ids.device
          )
          llm_pos_ids_list.append(vision_position_ids)
          current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size
  ```

- **展平视觉网格**：`grid_thw` 保存 `patch embedding` 后、空间合并前的网格，`H/W` 先除以 `spatial_merge_size`，与语言模型中的视觉槽位数量对齐。`get_vision_position_ids()` 用 `meshgrid()` 为每个槽位分配 `(t,h,w)`，按宽度最快、高度次之、时间最慢的顺序展平；三行的同一列始终对应同一个视觉槽位。`start_position` 将三个轴平移到当前区段起点：

  ```python
  llm_grid_t, llm_grid_h, llm_grid_w = (
      grid_thw[0].item() // temp_merge_size,
      grid_thw[1].item() // spatial_merge_size,
      grid_thw[2].item() // spatial_merge_size,
  )
  position_temporal = (torch.arange(llm_grid_t, device=device) * time_interval).long()
  position_height = torch.arange(llm_grid_h, device=device) + start_position
  position_width = torch.arange(llm_grid_w, device=device) + start_position
  T_grid, H_grid, W_grid = torch.meshgrid(
      position_temporal, position_height, position_width, indexing="ij"
  )
  vision_position_ids = torch.stack([T_grid, H_grid, W_grid], dim=0).reshape(3, -1)
  vision_position_ids[0] += start_position
  return vision_position_ids
  ```

- **视频时间片的位置**：上述调用的 `temp_merge_size=1`、`time_interval=1.0`；单张图片和每个视频时间片的时间轴长度均为 `1`，所以同一区段的 `T` 坐标相同。时间戳和起止标记按文本规则推进 `current_pos`，下一个时间片使用新的起点；实际视频时间由时间戳文本表达。

- **拼回整条序列**：各段沿 `dim=1` 拼接，得到当前样本去除补齐后的 `[3,有效长度]` 位置表；再用原始 `attention_mask` 将各列写回 `[3,B,L]` 的有效槽位。补齐位置保留初始化的 `0`，由掩码排除。`current_input_ids` 已经过滤，因此差值减去的是有效长度，不是补齐后的 `L`：

  ```python
  llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
  if attention_mask is not None:
      position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = (
          llm_positions.to(position_ids.device)
      )
  else:
      position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
  mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
  ```

- **返回位置与差值**：`position_ids: [3,B,L]` 提供每个 token 的三轴旋转坐标；`rope_deltas: [B,1]` 保存“下一个旋转位置 − 有效序列长度”，用于后续生成时将序列编号对齐到 RoPE 位置，不是 `KV cache` 的存储索引。每条样本各保存一个差值：

  ```python
  mrope_position_deltas = torch.tensor(
      mrope_position_deltas, device=input_ids.device
  ).unsqueeze(1)
  return position_ids, mrope_position_deltas
  ```

### 交错 `MRoPE`：按通道对选择 `T/H/W` 坐标

- **位置表提供旋转坐标**：`position_ids` 是普通 RoPE 中“位置 × 频率”的位置输入；`rope_theta` 决定频率底数。`Qwen3.5` 为每个 token 保存 `T/H/W` 三个坐标，用同一组 `inv_freq` 分别计算三个轴的角度。`Qwen3_5TextRotaryEmbedding.forward()` 中，相位与三角函数的核心语句为：

  ```python
  inv_freq_expanded = (
      self.inv_freq[None, None, :, None]
      .float()
      .expand(3, position_ids.shape[1], -1, 1)
  )
  position_ids_expanded = position_ids[:, :, None, :].float()
  freqs = (
      inv_freq_expanded.float() @ position_ids_expanded.float()
  ).transpose(2, 3)
  cos = freqs.cos() * self.attention_scaling
  sin = freqs.sin() * self.attention_scaling
  sin = self.recomposition_frequencies(sin)
  cos = self.recomposition_frequencies(cos)
  ```

- **每对通道从三个轴中选一个**：选轴前的 `cos/sin` 均为 `[3,B,L,32]`，每个 token 的每对旋转通道都有三个候选系数。`recomposition_frequencies()` 先取时间轴，再以步长 `3` 替换高度、宽度轴的系数，形成 `T,H,W,T,H,W,…` 的交错分配。`mrope_section=[11,11,10]` 表示三个轴分别负责 `11/11/10` 对通道：

  ```python
  def recomposition_frequencies(self, freq):
      freqs_thw = freq[0]
      for dim, offset in enumerate((1, 2), start=1):
          length = self.mrope_section[dim] * 3
          idx = slice(offset, length, 3)
          freqs_thw[..., idx] = freq[dim, ..., idx]
      return torch.cat((freqs_thw, freqs_thw), dim=-1)
  ```

  | 读取的坐标轴 | 频率下标 `i` | 对应的旋转通道对 `(i,i+32)` |
  | --- | --- | --- |
  | `T` | `0,3,6,…,30` | `(0,32)、(3,35)、…、(30,62)` |
  | `H` | `1,4,7,…,31` | `(1,33)、(4,36)、…、(31,63)` |
  | `W` | `2,5,8,…,29` | `(2,34)、(5,37)、…、(29,61)` |

- **交错的是坐标轴分配，配对方式仍是前后半区**：选轴后剩下 `[B,L,32]`；`cat()` 将这组系数复制成 `[B,L,64]`，让通道 `i` 和 `i+32` 使用同一个角度，通过 `rotate_half()` 完成二维旋转。每个 token 都使用表中的分配规则。文本的 `T/H/W` 坐标相同，得到普通一维 RoPE；图片和视频的坐标不同，各通道对分别编码时间、行、列位置。

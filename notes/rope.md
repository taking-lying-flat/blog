# `RoFormer`: `Enhanced Transformer with Rotary Position Embedding`

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

**1. 混合序列与媒体字段**：文本、图片和视频进入语言模型时共用一条序列。图片对应一段连续的 `<|image_pad|>`；视频按时间片展开，每个时间片前有文本时间戳，内部是一段连续的 `<|video_pad|>`：

```text
文本
<|vision_start|> image_pad ... image_pad <|vision_end|>
文本
<时间戳> <|vision_start|> video_pad ... video_pad <|vision_end|>
<时间戳> <|vision_start|> video_pad ... video_pad <|vision_end|>
文本
```

`mm tokens` 指媒体占位 `token`。模型输入中没有独立的 `mm_tokens` 参数：占位 `token` 保存在 `input_ids`，每个位置的媒体类型保存在 `mm_token_type_ids`，媒体内容则保存在像素张量中。设批大小为 `B`，补齐后的序列长度为 `L`，文本隐藏维度为 `D`：

| 字段 | 结构 | 用途 |
| --- | --- | --- |
| `input_ids` | `[B,L]` | 文本、媒体起止标记、媒体占位 `token` 的词表编号 |
| `mm_token_type_ids` | `[B,L]` | 与 `input_ids` 逐位置对应；`0` 为文本，`1` 为图片，`2` 为视频 |
| `attention_mask` | `[B,L]` | `1` 为有效槽位，`0` 为批处理补齐的槽位 |
| `image_grid_thw` | `[N_image,3]` | 每张图片的 `(T_g,H_g,W_g)` 视觉网格 |
| `video_grid_thw` | `[N_video,3]` | 每段视频的 `(T_g,H_g,W_g)` 视觉网格 |
| `pixel_values` | `[ΣP_image,patch_dim]` | 图片的展平 `patch` 数据 |
| `pixel_values_videos` | `[ΣP_video,patch_dim]` | 视频的展平时空 `patch` 数据 |
| `inputs_embeds` | `[B,L,D]` | 文本 `embedding` 与视觉编码器输出合并后的语言模型输入 |

`N_image/N_video` 是整个批次中的媒体数量；网格行的顺序与各样本中媒体出现的顺序一致。`T_g/H_g/W_g` 是 `patch embedding` 后、空间合并前的网格大小。语言模型使用的高度、宽度还要分别除以 `spatial_merge_size`；时间维已经反映 `temporal_patch_size` 的切块结果。`Qwen3VLProcessor.replace_image_token()` 按合并后的视觉特征数量扩展图片占位：

```python
merge_length = self.image_processor.merge_size**2
num_image_tokens = image_inputs["image_grid_thw"][image_idx].prod() // merge_length
return self.image_token * num_image_tokens
```

视频的占位按每个时间片分别扩展。`replace_video_token()` 先确定时间片数量与每片的空间槽位数，再将时间戳和起止标记写入文本：

```python
merge_length = self.video_processor.merge_size**2
num_frames = video_inputs["video_grid_thw"][video_idx][0]
frame_seqlen = video_inputs["video_grid_thw"][video_idx][1:].prod() // merge_length
metadata = video_inputs["video_metadata"][video_idx]
video_placeholder = ""

curr_timestamp = self._calculate_timestamps(metadata.frames_indices, metadata.fps, self.video_processor.temporal_patch_size)

for frame_idx in range(num_frames):
    curr_time = curr_timestamp[frame_idx]
    video_placeholder += f"<{curr_time:.1f} seconds>"
    video_placeholder += self.vision_start_token + self.video_token * frame_seqlen + self.vision_end_token
```

`num_frames` 对应视觉时间网格的长度，一个时间片可以包含多个原始视频帧。时间戳作为普通字符串交给 `tokenizer`，占多少文本 `token` 由实际编码决定。`mm_token_type_ids` 在文本编码后生成。`create_mm_token_type_ids()` 的主体按词表编号识别媒体占位：

```python
tokenizer_input = np.array(tokenizer_input)
mm_token_types = np.zeros_like(tokenizer_input)
mm_token_types[np.isin(tokenizer_input, self.image_token_ids)] = 1
mm_token_types[np.isin(tokenizer_input, self.video_token_ids)] = 2
mm_token_types[np.isin(tokenizer_input, self.audio_token_ids)] = 3
mm_token_type_ids.append(mm_token_types.tolist())
```

这是处理器的通用逻辑；`Qwen3.5` 的文本、图片、视频输入使用 `0/1/2`。`<|vision_start|>`、`<|vision_end|>` 和时间戳都保留类型 `0`。因此，连续的图片或视频槽位被文本类型的边界隔开，位置函数能够逐段取出对应网格。

媒体占位虽然叫 `pad`，其 `attention_mask` 仍为 `1`。真正的批处理补齐位置才为 `0`。二维 `attention_mask` 标识有效槽位，因果可见性由后面的遮罩构造另行加入。

`Qwen3_5Model.forward()` 先取得 `[B,L,D]` 的文本嵌入，再把图片、视频编码器输出填回各自的占位位置。图片分支为：

```python
inputs_embeds = self.get_input_embeddings()(input_ids)

image_outputs = self.get_image_features(pixel_values, image_grid_thw, return_dict=True, **kwargs)
image_embeds = image_outputs.pooler_output
image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
```

`get_placeholder_mask()` 用 `input_ids == image_token_id` 或 `video_token_id` 找槽位，并检查槽位数量是否匹配视觉特征数量。视频分支执行相同的填充过程。填充只替换 `embedding` 内容，序列长度及各槽位顺序保持不变；随后构造的每一列位置都与这条混合序列的一列对应。

### 2. `get_rope_index()`：从区段生成三维位置表

对 `Qwen3_5ForConditionalGeneration` 实例 `model`，位置函数位于 `model.model`：

```python
position_ids, rope_deltas = model.model.get_rope_index(
    input_ids=input_ids,
    mm_token_type_ids=mm_token_type_ids,
    image_grid_thw=image_grid_thw,
    video_grid_thw=video_grid_thw,
    attention_mask=attention_mask,
)
```

两个输出分别是 `[3,B,L]` 的三轴位置表与 `[B,1]` 的位置差值。`get_rope_index()` 使用类型和网格构造位置，不读取像素，也不计算旋转系数。

由于视频在序列中已被时间戳拆成多个区段，函数首先在内部将每段视频的网格拆成多个时间长度为 `1` 的网格，再为图片和视频各建立一个迭代器：

```python
if video_grid_thw is not None:
    video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
    video_grid_thw[:, 0] = 1
spatial_merge_size = self.config.vision_config.spatial_merge_size

mrope_position_deltas = []
position_ids = torch.zeros(3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device)
grid_iters = {
    1: iter(image_grid_thw) if image_grid_thw is not None else None,
    2: iter(video_grid_thw) if video_grid_thw is not None else None,
}
```

网格迭代器建立在批次循环外：先遍历样本，再按样本内部的媒体顺序消费网格。它们需要与处理器打包像素和网格的顺序一致。

对每个样本，代码先用 `attention_mask` 去掉补齐位置，再将连续相同类型的槽位组成 `(modality_type,start_idx,end_idx)`：

```python
for batch_idx, current_input_ids in enumerate(input_ids):
    input_token_type = mm_token_type_ids[batch_idx]
    if attention_mask is not None:
        current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
        input_token_type = input_token_type[attention_mask[batch_idx].bool()]

    input_type_group = []
    for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
        group = list(group)
        start_index = group[0][0]
        end_index = group[-1][0] + 1
        input_type_group.append((key, start_index, end_index))
```

其中 `start_idx/end_idx` 是去除补齐后的**序列区间**。接下来另用 `current_pos` 保存可分配的**旋转位置起点**。这两类坐标在文本区段中同步递增，在视觉区段中按不同规则推进：

```python
current_pos = 0
llm_pos_ids_list = []
for modality_type, start_idx, end_idx in input_type_group:
    if modality_type == 0:
        text_len = end_idx - start_idx
        llm_pos_ids_list.append(torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos)
        current_pos += text_len
    else:
        grid_thw = next(grid_iters[modality_type])
        vision_position_ids = self.get_vision_position_ids(
            current_pos, grid_thw, 1, spatial_merge_size, device=input_ids.device
        )
        llm_pos_ids_list.append(vision_position_ids)
        current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size
```

文本区段将一维编号复制到三行，同一个文本 `token` 的 `T/H/W` 位置相同。视觉区段调用 `get_vision_position_ids()`，用网格坐标替代连续的一维编号；区段结束后，`current_pos` 按合并后网格的最大边长推进，而不是按视觉 `token` 总数推进。

`get_vision_position_ids()` 的坐标构造主体为：

```python
llm_grid_t, llm_grid_h, llm_grid_w = (
    grid_thw[0].item() // temp_merge_size,
    grid_thw[1].item() // spatial_merge_size,
    grid_thw[2].item() // spatial_merge_size,
)

position_temporal = (torch.arange(llm_grid_t, device=device) * time_interval).long()
position_height = torch.arange(llm_grid_h, device=device) + start_position
position_width = torch.arange(llm_grid_w, device=device) + start_position

T_grid, H_grid, W_grid = torch.meshgrid(position_temporal, position_height, position_width, indexing="ij")
vision_position_ids = torch.stack([T_grid, H_grid, W_grid], dim=0).reshape(3, -1)
vision_position_ids[0] += start_position
return vision_position_ids
```

- `temp_merge_size` 在上述调用中为 `1`；`time_interval` 未传入，使用默认值 `1.0`。
- `meshgrid()` 让每个视觉槽位取得一组 `(t,h,w)`，`stack()` 将三个坐标分开存为三张网格。
- `reshape(3,-1)` 以宽度最快、高度次之、时间最慢的顺序展平；同一列始终对应同一个视觉槽位。
- `start_position` 将局部网格平移到整条混合序列当前的位置范围。

对于单张图片或拆开的视频时间片，时间网格长度为 `1`。设该区段起点为 `c`，其三行结构为：

```text
视觉槽位       第 0 行各列                    第 1 行各列                 ...
temporal      c,   c,   ..., c              c,   c,   ..., c            ...
height        c,   c,   ..., c              c+1, c+1, ..., c+1          ...
width         c, c+1, ..., c+W_llm-1        c, c+1, ..., c+W_llm-1       ...
```

视频的每个时间片分别取得自己的 `c`。两片之间的文本时间戳和起止标记也会推进 `current_pos`，因此 `temporal` 行保存的是这条编号规则产生的位置；视频实际时间则由前面的时间戳文本表达。各区段的 `[3,区段长度]` 张量沿序列维连接，再填回原批次的有效槽位：

```python
llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
if attention_mask is not None:
    position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(position_ids.device)
else:
    position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
```

批次循环结束后，差值列表转成 `[B,1]`，与位置表一起返回：

```python
mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
return position_ids, mrope_position_deltas
```

最终 `position_ids[axis,b,l]` 表示第 `b` 个样本、第 `l` 个槽位在指定轴上的位置。它的三行分别是 `temporal/height/width`，并非三个 `token`；补齐槽位保留初始化值，由 `attention_mask` 排除。函数内部的 `mrope_position_deltas` 在调用处解包为 `rope_deltas`。正常 `forward()` 未显式传入 `position_ids` 时，`Qwen3_5Model` 会调用 `compute_3d_position_ids()`。它在具有 `input_ids`、`mm_token_type_ids` 和视觉网格，且需要重新构造位置时，执行下面的分支：

```python
if can_compute_mrope and (self.rope_deltas is None or past_key_values_length == 0):
    position_ids, rope_deltas = self.get_rope_index(
        input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
        mm_token_type_ids=mm_token_type_ids,
    )
    self.rope_deltas = rope_deltas
```

`position_ids` 随合并后的 `inputs_embeds` 传入文本模型，差值留在 `Qwen3_5Model.rope_deltas` 中供后续接续位置使用：

```python
outputs = self.language_model(
    input_ids=None,
    position_ids=position_ids,
    attention_mask=attention_mask,
    past_key_values=past_key_values,
    inputs_embeds=inputs_embeds,
    **kwargs,
)
```

### 3. 三维位置表怎样进入旋转系数

`Qwen3_5TextModel.forward()` 将三轴位置传入 `Qwen3_5TextRotaryEmbedding`：

```python
hidden_states = inputs_embeds
position_embeddings = self.rotary_emb(hidden_states, position_ids)
```

`hidden_states` 提供设备和输出精度，旋转角由 `position_ids` 与 `inv_freq` 决定。沿用前文的配置，旋转区有 32 对通道，`inv_freq` 保存 32 个频率。`Qwen3_5TextRotaryEmbedding.forward()` 的计算为：

```python
inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
position_ids_expanded = position_ids[:, :, None, :].float()

device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
with maybe_autocast(device_type=device_type, enabled=False):
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
    cos = freqs.cos() * self.attention_scaling
    sin = freqs.sin() * self.attention_scaling

sin = self.recomposition_frequencies(sin)
cos = self.recomposition_frequencies(cos)
return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
```

矩阵乘法的两个收缩维度均为 `1`，作用是将每个位置与每个频率相乘：

```text
inv_freq                              [32]
inv_freq_expanded                     [3, B, 32, 1]
position_ids                          [3, B, L]
position_ids_expanded                 [3, B, 1, L]
矩阵乘法结果                           [3, B, 32, L]
transpose(2, 3) 后的 freqs             [3, B, L, 32]
求 cos / sin                          [3, B, L, 32]
```

固定一个槽位 `l` 和一个频率下标 `i`，`freqs` 中同时存在三个候选角度：

```text
freqs[0, b, l, i] = position_ids[0, b, l] * inv_freq[i]   temporal
freqs[1, b, l, i] = position_ids[1, b, l] * inv_freq[i]   height
freqs[2, b, l, i] = position_ids[2, b, l] * inv_freq[i]   width
```

三个轴使用同一条 `inv_freq`，只改变相乘的位置。当前 `rope_type="default"` 下 `attention_scaling=1.0`。源码在 `float32` 中计算相位与三角函数，最后再转换成 `hidden_states` 的精度。

### 4. `mrope_section` 怎样选择三轴系数

一对 `Q/K` 通道只需要一个旋转角，因此还要从上述三个候选中选出一个。`recomposition_frequencies()` 接收已经求好的三轴 `cos` 或 `sin`，按频率下标选择对应轴：

```python
def recomposition_frequencies(self, freq):
    freqs_thw = freq[0]
    for dim, offset in enumerate((1, 2), start=1):
        length = self.mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_thw[..., idx] = freq[dim, ..., idx]
    return torch.cat((freqs_thw, freqs_thw), dim=-1)
```

`freq[0]` 先取出全部时间轴系数，形状从 `[3,B,L,32]` 变成 `[B,L,32]`。循环再分两次替换最后一维的部分位置：`dim=1` 从高度轴取系数，`dim=2` 从宽度轴取系数；未替换的位置继续使用时间轴。

前文配置中的 `mrope_section=[11,11,10]` 表示分配给 `T/H/W` 的**频率数量，也就是旋转通道对数量**。对应切片为：

| 轴 | 选择方式 | 频率下标 |
| --- | --- | --- |
| `temporal` | 保留未被替换的位置 | `0,3,6,…,30` |
| `height` | `slice(1,33,3)` | `1,4,7,…,31` |
| `width` | `slice(2,30,3)` | `2,5,8,…,29` |

因此，每个槽位最终使用的系数按 `T,H,W,T,H,W,…` 交错排列。这个选择沿**频率维**进行，不沿 `token` 维进行；每个 `token` 都使用同样的轴分配规则。对于文本，三轴位置相同，选择任何一轴都得到相同角度。对于图片和视频，不同通道对分别携带时间、行、列位置。

最后的 `torch.cat((freqs_thw,freqs_thw),dim=-1)` 将 32 个系数复制为 64 个。前文的 `rotate_half()` 将旋转区的前后两半配成通道对，因此同一对的两个通道必须取得相同的 `cos/sin`：

```text
选轴后          [cos(φ_0), ..., cos(φ_31)]
复制后          [cos(φ_0), ..., cos(φ_31), cos(φ_0), ..., cos(φ_31)]
通道配对         (0,32), (1,33), ..., (31,63)
```

`T/H/W` 交错决定每一对读取哪个坐标；前后半区配对决定这一个角度作用于哪两个通道。源码不会对同一对通道连续施加三次旋转，也不会将三个轴的角度相加。

### 5. `cos/sin` 怎样作用到 `Q/K`

文本模型在层循环外生成 `position_embeddings=(cos,sin)`，随后传给各层：

```python
position_embeddings = self.rotary_emb(hidden_states, position_ids)

for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
    hidden_states = decoder_layer(
        hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=causal_mask_mapping[self.config.layer_types[i]],
        position_ids=text_position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        **kwargs,
    )
```

各层复用当前输入的旋转系数，但每层的 `Q/K` 来自各自的投影。在 `full_attention` 分支中，`Qwen3_5Attention.forward()` 先分离查询与输出门控，再完成 `Q/K norm` 和维度转置：

```python
input_shape = hidden_states.shape[:-1]
hidden_shape = (*input_shape, -1, self.head_dim)

query_states, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
gate = gate.reshape(*input_shape, -1)

query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

cos, sin = position_embeddings
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

此时 `Q/K` 分别为 `[B,H_q,L,d_h]` 和 `[B,H_kv,L,d_h]`；`cos/sin` 为 `[B,L,d_r]`。`apply_rotary_pos_emb()` 先在 `head` 位置增加长度为 `1` 的维度，让所有头广播使用同一组位置系数，再拆出旋转区：

```python
cos = cos.unsqueeze(unsqueeze_dim)
sin = sin.unsqueeze(unsqueeze_dim)

rotary_dim = cos.shape[-1]
q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

q_embed = torch.cat([q_embed, q_pass], dim=-1)
k_embed = torch.cat([k_embed, k_pass], dim=-1)
```

`rotate_half()` 将旋转区记作前半 `x1`、后半 `x2`，返回 `[-x2,x1]`。因此，上述逐元素运算在每对通道内完成以下两项：

```text
前半输出 = 前半输入 × cos - 后半输入 × sin
后半输出 = 后半输入 × cos + 前半输入 × sin
```

这就是前文的二维旋转。三维位置表没有直接加到 `embedding` 上，而是通过“位置乘频率 → 三角函数 → 按频率选轴 → 广播到各头”生成旋转系数。旋转区以外的 `q_pass/k_pass` 原样接回；`V` 不经过这一步。

旋转后的键才进入缓存，然后与查询一起交给注意力实现：

```python
if past_key_values is not None:
    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)

attn_output, attn_weights = attention_interface(
    self,
    query_states,
    key_states,
    value_states,
    attention_mask,
    dropout=0.0 if not self.training else self.attention_dropout,
    scaling=self.scaling,
    **kwargs,
)
```

缓存中的历史 `K` 已在写入时完成旋转，后续解码只旋转当前新增的 `Q/K`。这组旋转用于 `Qwen3.5` 文本骨干的 `full_attention` 层；`linear_attention` 层由 `GatedDeltaNet` 处理，不执行这一组 `Q/K` 旋转。

### 6. `text_position_ids` 与 `rope_deltas` 分别负责什么

三轴位置描述文本和视觉网格的位置关系，遮罩还需要处理整条序列的先后关系。`text_position_ids` 是 `[B,L]` 的一维序列位置，**包含文本、图片占位、视频占位和起止标记的全部槽位**；名字中的 `text` 不表示只给普通文字编号。`generate()` 首先调用 `_prepare_position_ids_for_generation()`。普通位置由 `GenerationMixin` 根据 `attention_mask` 生成：

```python
position_ids = attention_mask.long().cumsum(-1) - 1
position_ids = position_ids.masked_fill(attention_mask == 0, 0)
```

`Qwen3_5ForConditionalGeneration` 将这份一维结果保存为 `text_positions`，再取得三轴位置。多模态分支中：

```python
vision_positions, rope_deltas = self.model.get_rope_index(inputs_tensor, **model_kwargs)
self.model.rope_deltas = rope_deltas
```

随后把一维序列位置和三轴位置拼在同一个参数里传下去：

```python
text_positions = text_positions[None, ...]
position_ids = torch.cat([text_positions, vision_positions], dim=0)
```

`[4,B,L]` 是一个传参结构：第 0 行用于序列位置，后 3 行用于旋转。`Qwen3_5TextModel.forward()` 在入口将它拆开：

```python
if position_ids is None:
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
    position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
elif position_ids.ndim == 2:
    position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

if position_ids.ndim == 3 and position_ids.shape[0] == 4:
    text_position_ids = position_ids[0]
    position_ids = position_ids[1:]
else:
    text_position_ids = None
```

拆出的 `text_position_ids` 进入遮罩构造，后三行进入前面已经展开的 `rotary_emb`：

```python
mask_kwargs = {
    "config": self.config,
    "inputs_embeds": inputs_embeds,
    "attention_mask": attention_mask,
    "past_key_values": past_key_values,
    "position_ids": text_position_ids,
}
causal_mask_mapping = {
    "full_attention": create_causal_mask(**mask_kwargs),
    "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
}
```

普通未打包输入的因果顺序由序列槽位和缓存位置决定，二维 `attention_mask` 排除补齐位置。`text_position_ids` 还为无二维遮罩、无历史缓存的打包路径提供序列边界信息，遮罩代码只在这些条件成立时检测边界：

```python
if position_ids is not None and attention_mask is None and past_key_values is None:
    batch_size = inputs_embeds.shape[0]
    if batch_size != position_ids.shape[0]:
        position_ids = position_ids.expand(batch_size, -1)
    packed_sequence_mask = find_packed_sequence_indices(position_ids)
```

`find_packed_sequence_indices()` 用相邻一维位置是否连续判断所属序列：

```python
first_dummy_value = position_ids[:, :1] - 1
position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
packed_sequence_mask = (position_diff != 1).cumsum(-1)
```

图片内部的时间坐标会重复，宽度坐标在换行时会回退，它们表达的是视觉结构。单独保留 `text_position_ids`，使这些正常的网格变化不会被当作打包序列的边界。具体打包支持还取决于注意力后端和递推层的处理，三轴位置本身不承担序列隔离。

直接调用 `forward()` 时也可以只传 `[3,B,L]`，此时三行全部用于旋转，入口不会从中取出一行作为 `text_position_ids`。纯文本的 `[B,L]` 输入则会被复制成四行，再按相同规则拆分。

`rope_deltas` 解决的是生成阶段的位置接续。视觉区段按网格最大边长推进旋转编号，缓存却按全部视觉槽位增长，因此**已处理的有效槽位数**与**下一个旋转位置**可能不同。位置函数返回的差值来自这一行：

```python
mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
```

`llm_positions.max()+1` 是已分配三轴位置之后的起点，`len(current_input_ids)` 是去除补齐后的有效序列长度。两者之差以 `[B,1]` 保存，每个样本一个，自动构造位置的分支将它存入 `self.rope_deltas`。`compute_3d_position_ids()` 在需要利用已保存差值构造后续文本位置时，先形成一维位置，再扩展成三轴并加上差值：

```python
if attention_mask is not None:
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)
    position_ids = position_ids.view(1, batch_size, -1).repeat(3, 1, 1).to(inputs_embeds.device)
else:
    position_ids = torch.arange(past_key_values_length, past_key_values_length + seq_length)
    position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1).to(inputs_embeds.device)
delta = self.rope_deltas.repeat_interleave(batch_size // self.rope_deltas.shape[0], dim=0)
position_ids = position_ids + delta.to(device=inputs_embeds.device)
```

后续生成的是普通文本，三轴使用相同位置；这个偏移将其接到已分配的多模态位置之后。`repeat_interleave()` 用于使保存的批次差值与扩展后的批大小对应。差值只调整旋转位置，不改变缓存长度，也不移动任何媒体槽位。

同一次 `generate()` 已经持有完整的 `position_ids` 时，通用生成循环直接从最后一个位置继续递增：

```python
required_dim = [1] * (position_ids.dim() - 1) + [-1]
next_position_ids = (
    torch.arange(num_new_tokens, dtype=position_ids.dtype, device=position_ids.device).view(*required_dim)
    + position_ids[..., -1:]
    + 1
)
next_position_ids = torch.cat([position_ids, next_position_ids], dim=-1)
model_kwargs[position_ids_key] = next_position_ids
```

因此，`rope_deltas` 用于从一维序列位置恢复多模态位置的入口；已经接续好的位置逐步递增时，不会再反复叠加这个差值。手工调用 `get_rope_index()` 只获得返回值，保存差值的是外层的自动构造分支；显式传入 `position_ids` 会跳过该自动分支。

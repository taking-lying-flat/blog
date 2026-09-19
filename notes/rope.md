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

1. **准备混合序列**：`Qwen3VLProcessor` 展开图片、视频占位，输出 `input_ids`、`mm_token_type_ids`、视觉网格、像素张量和 `attention_mask`。
2. **填入特征、分配位置**：`Qwen3_5Model` 将视觉特征填回占位；`get_rope_index()` 根据类型和网格生成 `[3,B,L]` 的 `T/H/W` 位置表及 `[B,1]` 的 `rope_deltas`。
3. **分开序列位置与旋转位置**：`generate()` 首次构造多模态位置时，额外保留一行 `text_position_ids`，组成 `[4,B,L]`；文本模型将首行交给遮罩构造，后三行交给 `rotary_emb`。
4. **生成系数并旋转**：三轴位置分别乘 `inv_freq`，求 `cos/sin`，按 `mrope_section` 为每对通道选轴；各 `full_attention` 层在投影和归一化后旋转 `Q/K`，再写入缓存、计算注意力。
5. **接续生成位置**：后续文本从已有旋转位置继续递增；需要从一维序列位置恢复三轴位置时，使用保存的 `rope_deltas` 补上偏移。

### 混合序列与媒体字段

- **序列布局**：文本、图片和视频共用一条序列。图片是一段连续的 `<|image_pad|>`；视频按时间片展开，每片前有文本时间戳，内部是一段连续的 `<|video_pad|>`：

  ```text
  文本 → <|vision_start|> image_pad ... image_pad <|vision_end|> → 文本
       → <时间戳> <|vision_start|> video_pad ... video_pad <|vision_end|>
       → <时间戳> <|vision_start|> video_pad ... video_pad <|vision_end|> → 文本
  ```

- **输入字段**：`mm tokens` 指媒体占位 `token`，没有独立的 `mm_tokens` 参数。占位编号存于 `input_ids`，类型存于 `mm_token_type_ids`，媒体内容存于像素张量。`B/L/D` 分别表示批大小、补齐后的序列长度、文本隐藏维度：

  | 字段 | 结构 | 用途 |
  | --- | --- | --- |
  | `input_ids` | `[B,L]` | 文本、媒体起止标记、媒体占位的词表编号 |
  | `mm_token_type_ids` | `[B,L]` | 逐槽位类型：文本 `0`、图片 `1`、视频 `2` |
  | `attention_mask` | `[B,L]` | 有效槽位 `1`，批处理补齐槽位 `0` |
  | `image_grid_thw` | `[N_image,3]` | 每张图片的 `(T_g,H_g,W_g)` 网格 |
  | `video_grid_thw` | `[N_video,3]` | 每段视频的 `(T_g,H_g,W_g)` 网格 |
  | `pixel_values` | `[ΣP_image,patch_dim]` | 图片的展平 `patch` 数据 |
  | `pixel_values_videos` | `[ΣP_video,patch_dim]` | 视频的展平时空 `patch` 数据 |
  | `inputs_embeds` | `[B,L,D]` | 文本嵌入与视觉特征合并后的语言模型输入 |

- **图片占位**：`N_image/N_video` 统计整个批次，网格按媒体出现顺序排列。`T_g/H_g/W_g` 是 `patch embedding` 后、空间合并前的网格；语言模型的高度和宽度分别除以 `spatial_merge_size`。`Qwen3VLProcessor.replace_image_token()` 按合并后的特征数量扩展占位：

  ```python
  merge_length = self.image_processor.merge_size**2
  num_image_tokens = image_inputs["image_grid_thw"][image_idx].prod() // merge_length
  return self.image_token * num_image_tokens
  ```

- **视频占位**：`replace_video_token()` 按时间片扩展，每片保留自己的时间戳和起止标记。`num_frames` 是时间网格长度，一个时间片可以包含多个原始帧；时间戳作为普通文本交给 `tokenizer`：

  ```python
  merge_length = self.video_processor.merge_size**2
  num_frames = video_inputs["video_grid_thw"][video_idx][0]
  frame_seqlen = video_inputs["video_grid_thw"][video_idx][1:].prod() // merge_length
  metadata = video_inputs["video_metadata"][video_idx]
  curr_timestamp = self._calculate_timestamps(metadata.frames_indices, metadata.fps, self.video_processor.temporal_patch_size)
  video_placeholder = ""
  for frame_idx in range(num_frames):
      curr_time = curr_timestamp[frame_idx]
      video_placeholder += f"<{curr_time:.1f} seconds>"
      video_placeholder += self.vision_start_token + self.video_token * frame_seqlen + self.vision_end_token
  ```

- **区段标记**：`create_mm_token_type_ids()` 将图片、视频占位分别标成 `1/2`；时间戳、`<|vision_start|>`、`<|vision_end|>` 保持 `0`，把相邻媒体区段隔开。媒体占位的 `attention_mask` 为 `1`，名字中的 `pad` 不表示批处理补齐。图片、视频类型的赋值为：

  ```python
  tokenizer_input = np.array(tokenizer_input)
  mm_token_types = np.zeros_like(tokenizer_input)
  mm_token_types[np.isin(tokenizer_input, self.image_token_ids)] = 1
  mm_token_types[np.isin(tokenizer_input, self.video_token_ids)] = 2
  ```

- **视觉特征填充**：`Qwen3_5Model.forward()` 先生成文本嵌入，`get_placeholder_mask()` 按媒体编号找槽位并检查特征数量，再用 `masked_scatter()` 替换向量。视频分支相同；序列长度和顺序保持不变，每列位置仍对应原槽位：

  ```python
  inputs_embeds = self.get_input_embeddings()(input_ids)
  image_outputs = self.get_image_features(pixel_values, image_grid_thw, return_dict=True, **kwargs)
  image_embeds = image_outputs.pooler_output
  image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
  image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
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

- **按类型切段**：先按 `attention_mask` 排除补齐，再用 `itertools.groupby()` 将连续相同类型组成 `(modality_type,start_idx,end_idx)`。图片、视频各用一个跨批次的网格迭代器，按样本及媒体出现顺序取网格；视频已被时间戳分隔，网格也拆成时间长度为 `1` 的片段：

  ```python
  if video_grid_thw is not None:
      video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
      video_grid_thw[:, 0] = 1
  spatial_merge_size = self.config.vision_config.spatial_merge_size
  grid_iters = {
      1: iter(image_grid_thw) if image_grid_thw is not None else None,
      2: iter(video_grid_thw) if video_grid_thw is not None else None,
  }
  ```

- **推进位置**：`start_idx/end_idx` 表示序列区间，`current_pos` 表示旋转位置起点。文本把连续编号复制到三行；视觉区段按网格生成坐标，结束后按合并网格的最大边长推进，不能用视觉 `token` 总数代替：

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

- **展平视觉网格**：`get_vision_position_ids()` 用 `meshgrid()` 为每个槽位分配 `(t,h,w)`，按宽度最快、高度次之、时间最慢的顺序展平，再加区段起点。上述调用的 `temp_merge_size=1`、`time_interval=1.0`；单张图片和每个视频时间片的时间轴长度均为 `1`，视频实际时间由时间戳文本表达：

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

- **拼回整条序列**：各段沿序列维拼接，填回 `[3,B,L]` 的有效槽位，补齐位置由遮罩排除；差值保存“下一旋转位置”与“有效序列长度”的差。正常 `forward()` 未传 `position_ids` 时，`compute_3d_position_ids()` 负责调用该接口、保存 `self.rope_deltas`，并将位置表与 `inputs_embeds` 一起传入文本模型；手工调用接口只返回结果：

  ```python
  llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
  if attention_mask is not None:
      position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(position_ids.device)
  else:
      position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
  mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
  ```

### 三维位置表如何旋转 `Q/K`

- **位置乘频率**：`Qwen3_5TextModel.forward()` 调用 `self.rotary_emb(hidden_states, position_ids)`。`Qwen3_5TextRotaryEmbedding` 为三个轴使用同一条 `inv_freq`，只改变相乘的位置；沿用前文配置，`[3,B,L] × [32]` 得到 `[3,B,L,32]` 的候选角度。相位和三角函数用 `float32` 计算，`hidden_states` 决定设备及返回精度：

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

- **每对通道选一个轴**：`recomposition_frequencies()` 先取时间轴，再按频率下标用高度、宽度轴替换部分系数。`mrope_section=[11,11,10]` 分配的是三轴各自负责的通道对数量，选轴发生在频率维，所有 `token` 使用同一套规则：

  ```python
  def recomposition_frequencies(self, freq):
      freqs_thw = freq[0]
      for dim, offset in enumerate((1, 2), start=1):
          length = self.mrope_section[dim] * 3
          idx = slice(offset, length, 3)
          freqs_thw[..., idx] = freq[dim, ..., idx]
      return torch.cat((freqs_thw, freqs_thw), dim=-1)
  ```

  | 坐标轴 | 频率下标 | 选择方式 |
  | --- | --- | --- |
  | `T` | `0,3,6,…,30` | 保留时间轴 |
  | `H` | `1,4,7,…,31` | `slice(1,33,3)` |
  | `W` | `2,5,8,…,29` | `slice(2,30,3)` |

- **对齐旋转通道**：选轴后为 `[B,L,32]`，末尾 `cat()` 将系数复制成 `[B,L,64]`，让前后半区配对的通道 `(0,32)、(1,33)、…` 使用同一个角度。文本三轴位置相同，退化为普通 `RoPE`；视觉通道对分别读取时间、行、列坐标，每对只旋转一次。

- **应用到各层 `Q/K`**：`position_embeddings=(cos,sin)` 在层循环外生成并复用。`Qwen3_5Attention.forward()` 用各层的投影得到 `Q/K`，完成 `Q/K norm` 后调用 `apply_rotary_pos_emb()`；`unsqueeze_dim=1` 将系数变成 `[B,1,L,64]`，广播到所有头。核心计算为：

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

- **旋转后的去向**：`rotate_half([x1,x2])=[-x2,x1]`，与前文二维旋转相同；`q_pass/k_pass` 原样接回，`V` 不旋转。旋转后的 `K` 写入缓存，再与 `Q/V` 计算注意力；后续只旋转新增的 `Q/K`。这条路径用于 `full_attention` 层，`GatedDeltaNet` 的 `linear_attention` 层不执行这组旋转。

### `text_position_ids` 与 `rope_deltas`

- **一维序列位置**：`text_position_ids: [B,L]` 为整条序列编号，包含文本、媒体占位和起止标记。`GenerationMixin` 通常用 `attention_mask.long().cumsum(-1)-1` 生成有效位置；`Qwen3_5ForConditionalGeneration._prepare_position_ids_for_generation()` 将它和三轴位置拼为 `[4,B,L]`。多模态分支的关键语句为：

  ```python
  text_positions = super()._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)
  vision_positions, rope_deltas = self.model.get_rope_index(inputs_tensor, **model_kwargs)
  self.model.rope_deltas = rope_deltas
  text_positions = text_positions[None, ...]
  position_ids = torch.cat([text_positions, vision_positions], dim=0)
  ```

- **序列位置与旋转位置分流**：`Qwen3_5TextModel.forward()` 将首行交给 `create_causal_mask()` / `create_recurrent_attention_mask()`，后三行交给 `rotary_emb`。直接传 `[3,B,L]` 时三行全部用于旋转；纯文本 `[B,L]` 会先复制成四行：

  ```python
  if position_ids.ndim == 3 and position_ids.shape[0] == 4:
      text_position_ids = position_ids[0]
      position_ids = position_ids[1:]
  else:
      text_position_ids = None
  ```

- **打包边界**：普通输入的因果顺序由序列槽位和缓存位置决定，二维 `attention_mask` 排除补齐。没有二维遮罩和历史缓存时，遮罩代码可用一维位置的不连续处识别打包边界；视觉坐标的重复和回退不能用于这个判断，具体隔离仍取决于后端及递推层支持。`find_packed_sequence_indices()` 的核心为：

  ```python
  first_dummy_value = position_ids[:, :1] - 1
  position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
  packed_sequence_mask = (position_diff != 1).cumsum(-1)
  ```

- **生成位置偏移**：视觉网格按最大边长推进旋转位置，缓存按全部槽位增长，因此两者可能不同。`rope_deltas = llm_positions.max()+1-有效序列长度`，每个样本一个，形状为 `[B,1]`。`compute_3d_position_ids()` 复用差值时，先从 `attention_mask` 或缓存长度构造一维位置，扩展成三轴后加偏移；无二维遮罩分支的关键语句为：

  ```python
  position_ids = torch.arange(past_key_values_length, past_key_values_length + seq_length)
  position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1).to(inputs_embeds.device)
  delta = self.rope_deltas.repeat_interleave(batch_size // self.rope_deltas.shape[0], dim=0)
  position_ids = position_ids + delta.to(device=inputs_embeds.device)
  ```

- **接续规则**：后续生成普通文本，三轴位置相同；差值只调整旋转编号，不改变缓存长度或媒体槽位。同一次 `generate()` 已持有位置表时，从 `position_ids[..., -1:]` 继续递增，不重复叠加差值；显式传入 `position_ids` 会跳过 `forward()` 的自动构造分支。

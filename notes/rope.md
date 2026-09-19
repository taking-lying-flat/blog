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

1. **对齐内容与位置**：视觉特征填入混合序列的媒体槽位，得到 `inputs_embeds: [B,L,D]`；`get_rope_index()` 为同一条序列生成 `position_ids: [3,B,L]`，同一列对应同一个 token。
2. **计算三轴的候选角度**：`position_ids` 的三行分别乘同一组 `inv_freq`，再求 `cos/sin`，得到 `[3,B,L,32]` 的三轴系数。
3. **按通道对交错选轴**：`recomposition_frequencies()` 按 `T,H,W,T,H,W,…` 为每对通道选择坐标轴，得到 `[B,L,32]` 的系数。
4. **应用二维旋转**：将系数复制为 `[B,L,64]`，使前后半区的通道对 `(i,i+32)` 使用同一个角度，旋转 `Q/K`。

### 输入序列：视觉特征与位置表逐列对应

- **处理器准备槽位与类型**：`processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt")` 按消息中的 `type` 渲染媒体标记，再由 `Qwen3VLProcessor` 按视觉特征数量展开占位。图片对应连续的 `<|image_pad|>`；视频按时间片展开，每片带文本时间戳和连续的 `<|video_pad|>`。`mm tokens` 就是这些媒体占位 token；模型参数使用 `input_ids` 和 `mm_token_type_ids`，没有单独的 `mm_tokens`。`B/L/D` 分别表示批大小、补齐后的序列长度和隐藏维度。

  | 字段 | 结构 | 与位置表的关系 |
  | --- | --- | --- |
  | `input_ids` | `[B,L]` | 确定混合序列的槽位顺序 |
  | `mm_token_type_ids` | `[B,L]` | 文本 `0`、图片 `1`、视频 `2`，决定每段如何分配坐标 |
  | `image_grid_thw` / `video_grid_thw` | `[媒体数量,3]` | 提供视觉网格的时间、高度、宽度 |
  | `attention_mask` | `[B,L]` | 标明有效槽位，排除批处理补齐位置 |
  | `inputs_embeds` | `[B,L,D]` | 保存各槽位的文本嵌入或视觉特征 |
  | `position_ids` | `[3,B,L]` | 为同一槽位提供 `T/H/W` 三个坐标 |

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

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

## 交错多模态旋转位置编码（Interleaved MRoPE）流程

1. **准备混合序列与 `mm_tokens`**：处理器展开媒体占位，`input_ids` 记录各槽位的词表编号，`mm_token_type_ids` 标记模态；对应的媒体特征随后填入占位，与文本嵌入组成内容序列。
2. **生成三轴位置**：`get_rope_index()` 根据模态类型、视觉网格和有效槽位生成 `T/H/W` 坐标；位置表的每一列与同一 token 的内容向量对应，提供其旋转所需的位置因子。
3. **计算三轴候选系数**：三轴坐标分别乘共享频率 `inv_freq`，将位置因子转换为旋转角，再求 `cos/sin`，为每对通道准备来自三个轴的候选系数。
4. **按通道对交错选轴**：按 `mrope_section` 规定的数量，沿频率下标将通道对交错分配给 `T/H/W`；每对只保留对应轴的系数，最终得到 32 组确定的旋转系数。
5. **旋转 Q/K 的成对通道**：将 32 组系数复制到 64 个旋转通道，使 `(i,i+32)` 共用一个角度；各注意力头据此旋转投影、归一化后的 `Q/K`，结果用于注意力内积。

### `mm_tokens`：混合序列与模态类型

- **媒体槽位与类型编号**：`mm_tokens` 指混合序列中的媒体占位 token。用 `Qwen3-Omni` 的文本、图片、视频、音频四种输入模态举例，可以按 `0/1/2/3` 标记每个槽位：

  | 类型编号 | 槽位内容 | 填入语言模型的向量 |
  | --- | --- | --- |
  | `0` | 文本、消息标记、媒体起止标记 | token 的文本嵌入 |
  | `1` | 图片占位 | 图片编码器输出的特征 |
  | `2` | 视频占位 | 视频帧的视觉特征 |
  | `3` | 音频占位 | 音频编码器输出的特征 |

  `input_ids` 保存词表编号；类型编号只标记这个槽位属于哪种模态。例如一段文本后接图片区段，再接文本、视频和音频，类型可以写成 `0,0,1,1,1,1,0,2,2,0,3,3,3,0`；连续的多个 `1` 表示同一图片区段中的多个特征槽位。`Qwen3.5` 使用文本、图片、视频三类，类型为 `0/1/2`，由 `mm_token_type_ids: [B,L]` 逐槽位传给 `get_rope_index()`。

- **类型按占位编号生成**：`create_mm_token_type_ids()` 先把整条序列设为 `0`，再按图片、视频、音频占位的词表编号赋值：

  ```python
  tokenizer_input = np.array(tokenizer_input)
  mm_token_types = np.zeros_like(tokenizer_input)
  mm_token_types[np.isin(tokenizer_input, self.image_token_ids)] = 1
  mm_token_types[np.isin(tokenizer_input, self.video_token_ids)] = 2
  mm_token_types[np.isin(tokenizer_input, self.audio_token_ids)] = 3
  ```

- **处理器准备槽位与类型**：`processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt")` 按消息中的 `type` 渲染媒体标记，再由 `Qwen3VLProcessor` 按视觉特征数量展开占位。图片对应连续的 `<|image_pad|>`；视频按时间片展开，每片带文本时间戳和连续的 `<|video_pad|>`。`B/L/D` 分别表示批大小、补齐后的序列长度和隐藏维度。

  | 字段 | 结构 | 与位置表的关系 |
  | --- | --- | --- |
  | `input_ids` | `[B,L]` | 确定混合序列的槽位顺序 |
  | `mm_token_type_ids` | `[B,L]` | 文本 `0`、图片 `1`、视频 `2`，决定每段如何分配坐标 |
  | `image_grid_thw` / `video_grid_thw` | `[媒体数量,3]` | 提供视觉网格的时间、高度、宽度 |
  | `attention_mask` | `[B,L]` | 标明有效槽位，排除批处理补齐位置 |
  | `inputs_embeds` | `[B,L,D]` | 保存各槽位的文本嵌入或视觉特征 |
  | `position_ids` | `[3,B,L]` | 为同一槽位提供 `T/H/W` 三个坐标 |

- **区分媒体占位和补齐**：`<|image_pad|>`、`<|video_pad|>` 是有效媒体槽位，不是 padding 或 EOS token，`attention_mask` 为 `1`。`Qwen3.5-27B` 的 padding token 是 `<|endoftext|>`，tokenizer 的 EOS 是 `<|im_end|>`；生成配置把两者都列为停止 token。时间戳、消息标记和 `<|vision_start|>/<|vision_end|>` 的媒体类型均为 `0`，按文本位置规则处理。

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

### 三维位置表：单批次示例

- **对齐内容与位置：建立内容张量与三维坐标的对应关系**。设混合序列的内容张量为 $`X\in\mathbb R^{B\times L\times D}`$，对应 `inputs_embeds`；位置张量为 $`P\in\mathbb Z^{3\times B\times L}`$，对应 `position_ids`。`X[b,l,:]` 保存第 `b` 个样本、第 `l` 个 token 的内容向量，`P[:,b,l]` 保存同一 token 的时间、高度、宽度坐标。视觉特征填入媒体槽位后，序列长度与槽位顺序保持不变；`get_rope_index()` 根据模态类型和视觉网格，为这些槽位分配坐标。内容张量用于生成 `Q/K`，位置张量用于生成旋转系数。

  取 `bs=1`、`L=10`，单张图片的网格为 `[1,4,6]`，`spatial_merge_size=2`，空间合并后为 `[1,2,3]`，对应六个图片槽位。`t0/t1` 各表示一个文本 token，`S/E` 表示媒体起止标记，`Ihw` 表示图片第 `h` 行、第 `w` 列的槽位。三轴位置表为：

  | 槽位 `l` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | token | `t0` | `S` | `I00` | `I01` | `I02` | `I10` | `I11` | `I12` | `E` | `t1` |
  | `T = P[0,0,l]` | 0 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 5 | 6 |
  | `H = P[1,0,l]` | 0 | 1 | 2 | 2 | 2 | 3 | 3 | 3 | 5 | 6 |
  | `W = P[2,0,l]` | 0 | 1 | 2 | 3 | 4 | 2 | 3 | 4 | 5 | 6 |

  图片区段的起点为 `2`。单张图片的时间坐标恒为 `2`，高度、宽度坐标分别为区段起点加行号、列号，按宽度优先的顺序展平。区段结束后，旋转位置按最大网格边长 `3` 推进，因此 `E/t1` 的坐标为 `5/6`。六个图片槽位的词表编号均为 `<|image_pad|>`，各自对应不同的视觉特征和网格坐标。对 `I12`，内容向量为 `X[0,7,:]`，位置列为 `P[:,0,7]=[2,3,4]`。**位置表中的坐标直接作为相位计算的位置因子。**

### 交错 `MRoPE`：按通道对选择 `T/H/W` 坐标

- **计算三轴候选角度：位置坐标与频率向量的外积**。`Qwen3.5-27B` 的旋转维度为 $`d_r=64`$，对应 $`F=d_r/2=32`$ 个二维旋转子空间；频率底数为 $`\beta=10^7`$。轴下标 $`a=0,1,2`$ 分别表示 `T/H/W`，频率下标 $`i=0,\ldots,31`$ 对应通道对 $`(i,i+32)`$，`inv_freq[i]` 记为 $`\omega_i`$。每个轴的每个位置与全部频率相乘，构造候选相位张量：

  ```math
  \begin{aligned}
  \omega_i&=\beta^{-2i/d_r}=10^{-7i/32},\\
  \Phi_{a,b,l,i}&=P_{a,b,l}\omega_i,\qquad
  \Phi\in\mathbb R^{3\times B\times L\times32},\\
  C^{\mathrm{axis}}_{a,b,l,i}&=\cos\Phi_{a,b,l,i},\\
  S^{\mathrm{axis}}_{a,b,l,i}&=\sin\Phi_{a,b,l,i}.
  \end{aligned}
  ```

  `Qwen3_5TextRotaryEmbedding.forward()` 将三轴共享的 `inv_freq` 扩展为 `[3,B,32,1]`，位置扩展为 `[3,B,1,L]`；相乘得到 `[3,B,32,L]`，交换最后两维得到 `freqs: [3,B,L,32]`，再求同形状的 `cos/sin`（默认 `attention_scaling=1`）。单样本示例中的相位与系数均为 `[3,1,10,32]`；`I12` 的坐标 `[2,3,4]` 对应第 `i` 个子空间的三个候选角度 $`2\omega_i`$、$`3\omega_i`$、$`4\omega_i`$，分别生成一组余弦、正弦系数。

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

- **按通道对交错选轴：将三轴候选系数映射到旋转子空间**。`mrope_section=[11,11,10]` 表示时间、高度、宽度轴分别分配 `11/11/10` 个旋转子空间。`recomposition_frequencies()` 在频率下标 `0,3,6,…,30` 取时间轴，`1,4,7,…,31` 取高度轴，`2,5,8,…,29` 取宽度轴。对这组配置，定义轴选择映射 $`\sigma(i)=i\bmod3`$，则每个通道对实际使用的相位与系数为：

  ```math
  \begin{aligned}
  \varphi_{b,l,i}&=P_{\sigma(i),b,l}\omega_i,\\
  \widehat C_{b,l,i}&=C^{\mathrm{axis}}_{\sigma(i),b,l,i}
  =\cos\varphi_{b,l,i},\\
  \widehat S_{b,l,i}&=S^{\mathrm{axis}}_{\sigma(i),b,l,i}
  =\sin\varphi_{b,l,i}.
  \end{aligned}
  ```

  选择后的 $`\widehat C/\widehat S`$ 均为 `[B,L,32]`。轴选择作用于频率维：每个二维子空间只采用一个轴的坐标，所有 token 共享同一轴分配规则。源码先计算三轴 `cos/sin`，再按上述映射选取系数。对 `I12`，选轴后最终采用的 32 个旋转角为：

  ```math
  \varphi_{0,7,:}=[2\omega_0,\ 3\omega_1,\ 4\omega_2,\ 2\omega_3,\ldots,\ 2\omega_{30},\ 3\omega_{31}].
  ```

  `2、3、4` 是 `I12` 的三轴位置坐标。在第 `i` 个二维子空间，选中的坐标充当一维 RoPE 中的位置因子 $`p_i`$，与该子空间的频率 $`\omega_i`$ 相乘，得到实际旋转角 $`\varphi_{0,7,i}=p_i\omega_i`$。例如频率下标 `i=2` 读取宽度坐标 `4`，因此通道对 `(2,34)` 使用 $`\cos(4\omega_2)`$ 和 $`\sin(4\omega_2)`$。**交错结构由频率索引到坐标轴的映射确定。**

  ```python
  def recomposition_frequencies(self, freq):
      freqs_thw = freq[0]
      for dim, offset in enumerate((1, 2), start=1):
          length = self.mrope_section[dim] * 3
          idx = slice(offset, length, 3)
          freqs_thw[..., idx] = freq[dim, ..., idx]
      return torch.cat((freqs_thw, freqs_thw), dim=-1)
  ```

  > [!IMPORTANT]
  > **交错 MRoPE 将三轴坐标映射为分块旋转，使注意力的位置项由三维相对位移决定**。第 `i` 对通道 `(i,i+32)` 读取轴 $`\sigma(i)`$ 的坐标，以 $`\omega_i`$ 为角频率；32 个二维旋转共同组成该 token 在 64 维旋转空间中的位置变换。`position_ids` 参数化这一变换，`cos/sin` 是其逐通道计算所需的系数。记 $`q_l^{(i)}`$、$`k_m^{(i)}`$ 为两个 token 的第 `i` 对通道，$`R(\theta)`$ 为二维旋转矩阵，则：
  >
  > ```math
  > \begin{aligned}
  > \Delta\varphi_i
  > &=\varphi_{m,i}-\varphi_{l,i}
  > =\omega_i\bigl(P_{\sigma(i),m}-P_{\sigma(i),l}\bigr),\\
  > (\widetilde q_l^{(i)})^\top\widetilde k_m^{(i)}
  > &=(q_l^{(i)})^\top R(\Delta\varphi_i)k_m^{(i)}.
  > \end{aligned}
  > ```
  >
  > $`P_{a,l}`$ 表示第 `l` 个 token 在轴 `a` 上的坐标。旋转后的内积由两 token 在所选轴上的坐标差调制；各子空间的贡献相加，将时间、行、列方向的相对位移纳入注意力。**交错规则确定坐标轴与频率的对应关系，使三个轴的旋转子空间沿频率序列交替分布**。
  >
  > **文本 token 的三个轴共享同一位置因子**。对于第 `l` 个文本 token，$`T_l=H_l=W_l=p_l`$，因此任意子空间选轴后的旋转角均为 $`\varphi_{l,i}=p_l\omega_i`$。在相同频率与通道配对下，$`R_{\mathrm{MRoPE}}(p_l,p_l,p_l)=R_{\mathrm{RoPE}}(p_l)`$，与一维 RoPE 等价。$`p_l`$ 随 token 的序列位置变化，$`\omega_i`$ 随子空间变化。

- **应用二维旋转：32 个角度作用于 64 个通道**。取 `I12` 的一个注意力头，通道从 `0` 编号；`Q/K` 的 64 维旋转部分按 `(0,32)、(1,33)、…、(31,63)` 组成 32 对，每对使用一个旋转角。`I12` 的位置因子按 `2、3、4、2、3、4、…` 交错排列，分别乘对应频率，得到 $`2\omega_0,\,3\omega_1,\,4\omega_2,\,2\omega_3,\ldots`$ 这 32 个角度。`cos/sin` 各生成 32 个系数，再复制为 64 个，使配对通道共用同一角度。

  具体取 `(1,33)`、`(2,34)`、`(3,35)`：它们分别读取高度坐标 `3`、宽度坐标 `4`、时间坐标 `2`，旋转角依次为 $`3\omega_1`$、$`4\omega_2`$、$`2\omega_3`$。`Qwen3_5Attention.forward()` 将投影、归一化后的 `Q/K` 交给 `apply_rotary_pos_emb()`；`rotate_half()` 将前后两半 $`[x_1,x_2]`$ 变为 $`[-x_2,x_1]`$，从而完成配对通道的乘加。以 $`q_j`$ 表示 `Q` 的第 `j` 个通道，这三对通道的输出为：

  ```math
  \begin{aligned}
  \widetilde q_1&=q_1\cos(3\omega_1)-q_{33}\sin(3\omega_1),\\
  \widetilde q_{33}&=q_1\sin(3\omega_1)+q_{33}\cos(3\omega_1),\\[4pt]
  \widetilde q_2&=q_2\cos(4\omega_2)-q_{34}\sin(4\omega_2),\\
  \widetilde q_{34}&=q_2\sin(4\omega_2)+q_{34}\cos(4\omega_2),\\[4pt]
  \widetilde q_3&=q_3\cos(2\omega_3)-q_{35}\sin(2\omega_3),\\
  \widetilde q_{35}&=q_3\sin(2\omega_3)+q_{35}\cos(2\omega_3).
  \end{aligned}
  ```

  其余 29 对通道分别代入各自的旋转角，完成整个 64 维部分的旋转；后 `192` 维原样接回，`V` 不参与旋转。各注意力头共享这些位置系数，分别处理自身的 `Q/K`，旋转结果参与注意力内积。文本 token 的三轴坐标相同，例如 `t1` 为 `[6,6,6]`，则各子空间的旋转角为 $`6\omega_i`$，与一维 RoPE 一致。

- **交错的是坐标轴分配，配对方式仍是前后半区**：选轴后剩下 `[B,L,32]`；`cat()` 将这组系数复制成 `[B,L,64]`，让通道 `i` 和 `i+32` 使用同一个角度，通过 `rotate_half()` 完成二维旋转。每个 token 都使用同一套坐标轴分配规则。文本的 `T/H/W` 坐标相同，得到普通一维 RoPE；图片和视频的坐标不同，各通道对分别编码时间、行、列位置。

### Qwen3.5：MRoPE 的接口调用

- **位置入口：`Qwen3_5Model.forward()`**。媒体特征填充后，`inputs_embeds: [B,L,D]` 与位置表按槽位对应。未显式传入 `position_ids` 时，`compute_3d_position_ids()` 在首次多模态前向中调用 `get_rope_index()`，生成 `[3,B,L]` 并保存 `rope_deltas`；已有位置表则直接传入文本模型。位置构造与传递的调用为：

  ```python
  if position_ids is None:
      position_ids = self.compute_3d_position_ids(
          input_ids=input_ids,
          image_grid_thw=image_grid_thw,
          video_grid_thw=video_grid_thw,
          inputs_embeds=inputs_embeds,
          attention_mask=attention_mask,
          past_key_values=past_key_values,
          mm_token_type_ids=mm_token_type_ids,
      )

  outputs = self.language_model(
      input_ids=None,
      position_ids=position_ids,
      attention_mask=attention_mask,
      past_key_values=past_key_values,
      inputs_embeds=inputs_embeds,
      **kwargs,
  )
  ```

- **系数接口：`Qwen3_5TextModel.forward()` → `Qwen3_5TextRotaryEmbedding.forward()`**。文本模型接收三轴位置，通过 `self.rotary_emb(hidden_states, position_ids)` 返回 `position_embeddings=(cos,sin)`，两者均为 `[B,L,64]`，一次计算后传给各层。旋转模块用 `hidden_states` 确定设备类型与输出 `dtype`，用 `position_ids` 和 `inv_freq` 计算角度。`generate()` 首次准备位置时，将覆盖全部槽位的一维序列位置与三轴位置拼成 `[4,B,L]`；文本模型将首行拆为 `text_position_ids`，用于注意力掩码构造，其余三行用于旋转系数计算。直接传入 `[3,B,L]` 时，三行均用于旋转，`text_position_ids=None`。核心传递语句为：

  ```python
  if position_ids.ndim == 3 and position_ids.shape[0] == 4:
      text_position_ids = position_ids[0]
      position_ids = position_ids[1:]
  else:
      text_position_ids = None

  hidden_states = inputs_embeds
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

- **旋转接口：`Qwen3_5Attention.forward()` → `apply_rotary_pos_emb()`**。`Qwen3_5DecoderLayer` 的 `full_attention` 分支将 `position_embeddings` 传给 `self.self_attn`。注意力层先投影、归一化得到本层的 `Q/K`，再解包系数并执行旋转；`cos/sin` 从 `[B,L,64]` 扩展为 `[B,1,L,64]`，广播到各个注意力头，旋转后的 `Q/K` 用于注意力计算。`linear_attention` 分支使用 Gated DeltaNet，不调用这组旋转接口。实际调用为：

  ```python
  cos, sin = position_embeddings
  query_states, key_states = apply_rotary_pos_emb(
      query_states, key_states, cos, sin
  )
  ```

### 条件生成模型：输入与视觉特征填充

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

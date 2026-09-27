# `ms-swift ChatML` 渲染

`messages` 先按 `ChatML` 模板拼接，再经过 `tokenizer`；多模态输入还需要按媒体特征长度展开占位 `token`。下面以 `Qwen3.5` 的生成前缀和 `Qwen3-Omni` 的本地模板编码为例，沿这条链路计算最终序列长度。

## 1. 消息如何拼成 `ChatML`

`ChatML` 为每条消息写入角色、换行和结束标记。已有 `assistant` 回答时，把回答放进消息体；等待生成时，输入停在最后一轮 `assistant` 的起点。

```text
<|im_start|>system
{SYSTEM}<|im_end|>
<|im_start|>user
{QUERY}<|im_end|>
<|im_start|>assistant
{RESPONSE}<|im_end|>
```

这些角色标记、换行和结束符都进入编码。**序列长度应在模板拼接后计算，不能只对用户问题做 tokenize。** [官方 ChatML 模板](https://github.com/modelscope/ms-swift/blob/be11a273f43dcc99f4e5d9b44e556f05dec85b39/swift/template/templates/utils.py#L13-L19)

### `Qwen3.5` 的生成前缀

`Qwen3.5` 在最后一轮 `assistant` 后预填思考或非思考前缀：

```text
thinking:
<|im_start|>assistant
<think>

non-thinking:
<|im_start|>assistant
<think>

</think>

```

空思考块属于输入前缀，长度计入 `prompt`。模型随后从该前缀末尾继续生成。`response_prefix` 可显式覆盖前缀；未覆盖时，`enable_thinking` 决定采用哪一种。已有回答的样本则编码实际的 `assistant` 内容。[官方前缀选择](https://github.com/modelscope/ms-swift/blob/be11a273f43dcc99f4e5d9b44e556f05dec85b39/swift/template/base.py#L192-L205)

## 2. 多模态标记如何进入 `ChatML`

标准样本在 `messages.content` 中放置 `<image>`、`<video>`、`<audio>`，媒体数据分别放在顶层 `images`、`videos`、`audios` 数组中。同一模态按出现顺序配对。

在 `Qwen3-Omni` 的独立媒体路径中，`replace_tag()` 将它们改写成模型标记：

| 标准标记 | 写入 ChatML 的内容 |
| --- | --- |
| `<image>` | `<\|vision_start\|><\|image_pad\|><\|vision_end\|>` |
| `<video>` | `<\|vision_start\|><\|video_pad\|><\|vision_end\|>` |
| `<audio>` | `<\|audio_start\|><\|audio_pad\|><\|audio_end\|>` |

[官方媒体标记替换](https://github.com/modelscope/ms-swift/blob/be11a273f43dcc99f4e5d9b44e556f05dec85b39/swift/template/templates/qwen.py#L924-L983)

例如，包含一张图片的用户消息在展开前是：

```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>图片里是什么？<|im_end|>
```

此时每份媒体只有一个 `pad token`。**真正占用多少 token，要等 processor 给出视觉网格或有效声学帧数后再确定。** 以下按 `use_audio_in_video=False` 的独立图片、视频和音频说明；带音轨的视频使用专门的音视频交错路径。

## 3. 图片与视频的长度

`processor` 输出 `image_grid_thw` 或 `video_grid_thw`。每行是媒体的 `(T, H, W)` patch 网格；空间合并尺寸为 `merge_size`，占位长度为：

```math
N_{\mathrm{vision}}=\frac{T\,H\,W}{m^2},\qquad m=\texttt{merge\_size}.
```

这里的 `H/W` 是预处理后的 patch 网格尺寸；时间维 `T` 不参与空间合并。`media_grid_thw` 根据当前媒体类型取图片或视频网格，`token_id` 是对应媒体 token 的单元素列表：

```python
def _get_new_tokens(i):
    token_len = media_grid_thw[i].prod() // (merge_size**2)
    return token_id * token_len
```

`merge_size=2` 时：

| 媒体 | processor 网格 `(T,H,W)` | 媒体 token 数 |
| --- | --- | --- |
| 图片 | `(1,32,48)` | `1 × 32 × 48 ÷ 4 = 384` |
| 独立视频 | `(8,24,40)` | `8 × 24 × 40 ÷ 4 = 1920` |

## 4. 音频的长度

`feature_attention_mask` 的有效位置数给出声学帧数；音频编码器经过下采样后，输出 token 数由 `_get_feat_extract_output_lengths()` 计算：

```python
feature_attention_mask = media_inputs.get('feature_attention_mask')
if feature_attention_mask is not None:
    audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
    audio_lengths = self._get_feat_extract_output_lengths(audio_feature_lengths)
else:
    audio_lengths = None
```

`Qwen3-Omni` 使用下列原方法中的 `omni_v3` 分支：

```python
def _get_feat_extract_output_lengths(self, input_lengths):
    if self.version == 'omni_v2_5':
        return ((input_lengths - 1) // 2 + 1 - 2) // 2 + 1
    elif self.version == 'omni_v3':
        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        return (
            ((feat_lengths - 1) // 2 + 1 - 1) // 2
            + 1
            + (input_lengths // 100) * 13
        )
```

例如，有效声学帧数为 `300` 时，`omni_v3` 分支返回 `39` 个音频 token。这里需要有效帧数，不能使用补齐后的音频张量宽度代替。

## 5. 占位符展开后的总长度

`_extend_tokens()` 将每个媒体的单个 `pad token` 替换为对应长度的 token 列表，左右边界保持不变。循环中的 `added_tokens_len` 累计此前替换带来的长度增量：

```python
token_len = len(new_tokens)
input_ids = (
    input_ids[: idx + added_tokens_len]
    + new_tokens
    + input_ids[added_tokens_len + idx + 1 :]
)
```

图片示例由此变为：

```text
<|vision_start|>
<|image_pad|> × 384
<|vision_end|>
```

`× 384` 表示重复 token，并非写入输入的文字。上述图片区段共有 `386` 个 token：两个边界加 `384` 个媒体 token。

设 `L₀` 为媒体展开前的 token 数，已经包含角色、换行、边界和每份媒体的一个占位 token；共有 `M` 份媒体，第 `j` 份媒体展开成 `Nⱼ` 个 token。对于前述独立媒体路径，截断和 batch padding 之前：

```math
L=L_0+\sum_{j=1}^{M}(N_j-1).
```

一张 `384` token 的图片、一段 `1920` token 的独立视频和一条 `39` token 的音频，会使序列增加 `383 + 1919 + 38 = 2340` 个 token。两侧媒体边界已包含在 `L₀` 中，不应重复累加。

`ms-swift` 最终对编码得到的序列取长度，原方法如下：

```python
@staticmethod
def _get_length(input_ids, labels):
    lengths = [0]
    if input_ids is not None:
        lengths.append(len(input_ids))
    if labels is not None:
        lengths.append(len(labels))
    length = max(lengths)
    return length
```

对这里的因果语言模型样本，长度就是 `len(input_ids)`；若同时生成 `labels`，它与 `input_ids` 等长。若触发截断，再以截断后的序列为准；普通 padding batch 的补齐长度另取批内最大值，补齐位置不增加样本的有效长度。

# GDN Chunkwise Training

# 一、输入与投影：生成后续计算需要的变量

这一部分对应外层的 `Qwen3_5GatedDeltaNet.forward`：读取缓存，将输入投影为 QKV、输出门和状态更新所需的门控变量，再完成短卷积与 head 对齐。

## 1.1 读取缓存，并执行四个投影

```python 读取缓存与输入投影
hidden_states = apply_mask_to_padding_states(
    hidden_states, attention_mask
)
batch_size, seq_len, _ = hidden_states.shape

if cache_params is not None:
    conv_state = cache_params.get_conv_state(
        self.layer_idx, state_idx=0
    )
    recurrent_state = cache_params.get_recurrent_state(
        self.layer_idx, state_idx=0
    )
else:
    conv_state, recurrent_state = None, None

mixed_qkv = self.in_proj_qkv(hidden_states)
mixed_qkv = mixed_qkv.transpose(1, 2)

z = self.in_proj_z(hidden_states)
z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

b = self.in_proj_b(hidden_states)
a = self.in_proj_a(hidden_states)
```

**这些变量分别做什么：**

| 变量 | 来源 | 后续用途 |
| --- | --- | --- |
| `mixed_qkv` | `in_proj_qkv` | 经过卷积、激活后拆出 `query`、`key`、`value` |
| `z` | `in_proj_z` | 在 GDN 读出结果之后，控制输出门 |
| `b` | `in_proj_b` | 生成更新强度 `beta` |
| `a` | `in_proj_a` | 生成对数衰减 `g` |
| `conv_state` | 缓存 | 保存短卷积需要的历史输入 |
| `recurrent_state` | 缓存 | 保存 GDN 的历史记忆矩阵 |

这里先区分两个缓存：**卷积缓存用于短卷积，递推缓存用于后面的 GDN 状态更新。**

`z` 在这一块生成，但暂时不参与 state 的计算。它一直保留到外层输出处理。

## 1.2 QKV 经过因果卷积和激活，再拆分 head

128-token 输入会进入多 token 的卷积分支。下面节选这一分支及随后的 QKV 拆分过程：

```python 因果卷积与 QKV 拆分
if cache_params is not None:
    mixed_qkv = cache_params.update_conv_state(
        mixed_qkv,
        self.layer_idx,
        conv_kernel_size=self.conv_kernel_size,
    )

mixed_qkv = causal_conv1d_fn(
    mixed_qkv,
    self.conv1d.weight.squeeze(1),
    self.conv1d.bias,
    activation=self.activation,
    **kwargs,
)

if cache_params is not None:
    mixed_qkv = mixed_qkv[:, :, -seq_len:]

mixed_qkv = mixed_qkv.transpose(1, 2)
query, key, value = torch.split(
    mixed_qkv,
    [self.key_dim, self.key_dim, self.value_dim],
    dim=-1,
)

query = query.reshape(
    batch_size, seq_len, -1, self.head_k_dim
)
key = key.reshape(
    batch_size, seq_len, -1, self.head_k_dim
)
value = value.reshape(
    batch_size, seq_len, -1, self.head_v_dim
)
```

`causal_conv1d_fn` 在序列方向做因果卷积，并执行配置指定的激活。Qwen3.5 默认配置使用 SiLU。

这个卷积按通道分别计算，使每个位置送入 GDN 的 QKV 包含附近历史 token 的信息。之后才拆出三个分支，整理成多头形状。

此时：

- `query` 和 `key` 的最后一维是 `head_k_dim`。
- `value` 的最后一维是 `head_v_dim`。
- 它们还没有进入 chunk 三角求解。

论文 §3.4 对应的是“线性投影、短卷积、SiLU，再对 query/key 做 L2 normalization”这条结构链路。[Gated Delta Networks，§3.4](https://arxiv.org/html/2412.06464v2#S3.SS4)

## 1.3 生成更新强度、衰减，并对齐 head 数

```python 更新强度、对数衰减与 head 对齐
beta = b.sigmoid()
g = (
    -self.A_log.float().exp()
    * F.softplus(a.float() + self.dt_bias)
)

if self.num_v_heads // self.num_k_heads > 1:
    query = query.repeat_interleave(
        self.num_v_heads // self.num_k_heads, dim=2
    )
    key = key.repeat_interleave(
        self.num_v_heads // self.num_k_heads, dim=2
    )
```

- `beta` 控制当前 token 的 delta 更新强度。
- `g` 保存对数域中的衰减，后面的内核通过 `exp()` 使用实际衰减。
- `A_log` 和 `dt_bias` 是模型参数，参与衰减的参数化。
- 当 value head 数更多时，这段代码扩展 query/key 的 head 维，使后续每个 value head 都有对应的 query/key。

这一块产生的变量，最终进入论文的单步更新规则。下面保留论文原始 gated delta rule 的式 (10)，不作等价展开。[Gated Delta Networks，§3.1](https://arxiv.org/html/2412.06464v2#S3.SS1)

```math
\mathbf{S}_{t}
=
\mathbf{S}_{t-1}
\left(
\alpha_{t}
\left(
\mathbf{I}
-
\beta_{t}\boldsymbol{k}_{t}\boldsymbol{k}_{t}^{\intercal}
\right)
\right)
+
\beta_{t}\boldsymbol{v}_{t}\boldsymbol{k}_{t}^{\intercal}.
\tag{10}
```

第一部分结束后，送入核心内核的是 `query`、`key`、`value`、`g`、`beta` 和入口 `recurrent_state`；**`z` 留在外层等待输出处理。**

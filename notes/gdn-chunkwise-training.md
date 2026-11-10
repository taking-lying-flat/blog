# GDN Chunkwise Training

Gated DeltaNet 在进入 chunk 计算之前，需要先把 `hidden_states` 转换为读写记忆的 QKV，以及控制记忆更新与衰减的门控量。这一部分沿着 Qwen3.5 的输入处理流程，说明这些变量如何生成；代码取自 Transformers [2026-11-04 版本的 `Qwen3_5GatedDeltaNet.forward`](https://github.com/huggingface/transformers/blob/fd3cf0140de03f7619ac95b5cfed05b711ff9e22/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L535)，以 128-token 输入对应的多 token 分支为例。

输入先按 mask 将 padding 位置置零，再经过四个独立投影：`mixed_qkv` 用于构造 QKV，`z` 控制最终输出，`b`、`a` 分别生成更新强度和衰减。若传入缓存，还会读取短卷积的历史输入 `conv_state`，以及保存历史记忆矩阵的 `recurrent_state`。

```python
hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
batch_size, seq_len, _ = hidden_states.shape
if cache_params is not None:
    conv_state = cache_params.get_conv_state(self.layer_idx, state_idx=0)
    recurrent_state = cache_params.get_recurrent_state(self.layer_idx, state_idx=0)
else:
    conv_state, recurrent_state = None, None

mixed_qkv = self.in_proj_qkv(hidden_states)
mixed_qkv = mixed_qkv.transpose(1, 2)
z = self.in_proj_z(hidden_states)
z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
b = self.in_proj_b(hidden_states)
a = self.in_proj_a(hidden_states)
```

接下来，`mixed_qkv` 沿序列方向做逐通道因果卷积，并执行 SiLU 激活，使每个位置的 QKV 融入邻近历史 token 的信息。多 token 分支通过 `update_conv_state` 接入并更新卷积缓存，卷积后只保留本次输入对应的位置，再拆分 QKV、整理 head 维度：

```python
if cache_params is not None:
    mixed_qkv = cache_params.update_conv_state(
        mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size)
mixed_qkv = causal_conv1d_fn(
    mixed_qkv, self.conv1d.weight.squeeze(1), self.conv1d.bias,
    activation=self.activation, **kwargs)
if cache_params is not None:
    mixed_qkv = mixed_qkv[:, :, -seq_len:]

mixed_qkv = mixed_qkv.transpose(1, 2)
query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
```

此时 Q/K 的每个 head 有 `head_k_dim` 个分量，V 则有 `head_v_dim` 个分量。论文 [§3.4](https://arxiv.org/html/2412.06464v2#S3.SS4) 中的 Q/K L2 normalization，在这份实现里通过 `use_qk_l2norm_in_kernel=True` 交给后续内核完成。

最后，`sigmoid(b)` 得到更新强度 `beta`；`a` 与可学习参数 `A_log`、`dt_bias` 生成对数衰减 `g`，实际衰减因子为 $`\alpha_t=\exp(g_t)`$。当 V 的 head 数更多时，这一版本会重复 Q/K 的 head，使每个 value head 都有对应的 query 和 key。

```python
beta = b.sigmoid()
g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
if self.num_v_heads // self.num_k_heads > 1:
    query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
    key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
```

这些门控量对应论文的 gated delta rule：$`\alpha_t`$ 控制历史记忆的衰减，$`\beta_t`$ 控制当前 token 的纠正与写入强度。沿用论文 [式 (10)](https://arxiv.org/html/2412.06464v2#S3.SS1) 的记号，状态更新为

```math
\mathbf{S}_{t}
=\mathbf{S}_{t-1}\left(\alpha_{t}\left(\mathbf{I}
-\beta_{t}\boldsymbol{k}_{t}\boldsymbol{k}_{t}^{\intercal}\right)\right)
+\beta_{t}\boldsymbol{v}_{t}\boldsymbol{k}_{t}^{\intercal}.
\tag{10}
```

至此，`query`、`key`、`value`、`g`、`beta` 和入口 `recurrent_state` 已准备好，随后交给 chunk 内核；**`z` 留在外层，在读出结果之后参与输出门控。**

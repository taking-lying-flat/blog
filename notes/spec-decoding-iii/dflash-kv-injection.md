**附录 A.3 进一步给出 KV 注入的具体计算：选定目标层的隐藏状态先拼接，再经一次投影与 RMSNorm，得到所有草稿层共享的目标上下文特征。**

```math
\mathbf{H}_t=\operatorname{RMSNorm}\!\left(W_c[\mathbf{H}^{(l_1)};\ldots;\mathbf{H}^{(l_5)}]\right)\tag{A.1}
```

- 在第 $`i`$ 个草稿层，查询仅由草稿 token 的隐藏状态 $`\mathbf{H}_d`$ 生成；目标特征与草稿 token 则分别通过该层的键、值投影，再沿序列维度拼接：

```math
\begin{aligned}
\mathbf{Q}_i&=W_i^Q\mathbf{H}_d,\\
\mathbf{K}_i&=[W_i^K\mathbf{H}_t;W_i^K\mathbf{H}_d]_{\mathrm{seq}},\\
\mathbf{V}_i&=[W_i^V\mathbf{H}_t;W_i^V\mathbf{H}_d]_{\mathrm{seq}}.
\end{aligned}\tag{A.2}
```

- 目标特征仅为掩蔽块中的草稿 token 提供额外的 KV 条目，不经过草稿模型的 Q 投影、输出投影、自注意力更新与 FFN。
- 唯一额外引入参数的组件是共享投影 $`W_c\in\mathbb{R}^{D\times5D}`$。以 Qwen3.5-35B-A3B 为例，$`D=2048`$，采用 BF16 权重时，额外参数占用约 42 MB，相比约 70 GB 的目标模型很小：

```math
5\times2048\times2048\times2\approx42\ \mathrm{MB}\tag{A.3}
```

- 激活开销同样较低：批量大小为 1、序列长度为 2048 时，投影输入与输出分别约占 40 MB 和 8 MB；解码块大小为 16 时，临时激活占用低于 400 KB。

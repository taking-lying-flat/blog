<aside class="cot-sampling-note" id="cot-sampling-note" aria-label="CoT-decoding 的候选生成">
<p><strong>CoT-decoding 的候选生成与路径筛选</strong></p>
<p>CoT-decoding 在首个解码位置枚举 top-k 候选 token，分别初始化候选路径；随后，各路径独立执行 greedy decoding，逐步选择条件概率最高的 token。该过程仅在首个位置展开候选，后续不再分叉，也不进行随机抽样。相比之下，top-k sampling 从截断并重新归一化的概率分布中随机抽取 token</p>
<p>候选集合可同时包含直接作答与显式推理路径。上文苹果问题示意图中，以 I 和 You 开头的第 2、4 条路径包含显式 CoT；其余路径直接作答。路径类型由生成内容体现，算法不预先指定哪些分支属于 CoT，而以最终答案区间的置信度进行筛选</p>
</aside>

以下为候选生成伪代码：`prompt` 为 token ID 列表，`next_logits` 返回给定前缀的完整词表 logits，`is_eos` 判断终止 token

```python 首步枚举与分支贪心续写 | 伪代码
def cot_decode_paths(prompt, k, max_new_tokens):
    first_logits = next_logits(prompt)
    paths = []
    for first_token in first_logits.topk(k).indices:
        tokens, margins = [], []
        logits, token = first_logits, first_token
        for step in range(max_new_tokens):
            top2 = logits.softmax(dim=-1).topk(2).values
            margins.append((top2[0] - top2[1]).item())
            tokens.append(token.item())
            if is_eos(token) or step + 1 == max_new_tokens:
                break
            logits = next_logits(prompt + tokens)
            token = logits.argmax(dim=-1)
        paths.append((tokens, margins))
    return paths
```

每个生成 token 对应一个概率差，仅对最终答案区间取平均。作者补充代码在 GSM8K 中提取最后一个数值片段，再比较各候选的答案置信度，基本形式输出得分最高的候选答案

```python 答案区间的置信度 | 作者源码节选
blocks = find_number_blocks(output_ids)
if len(blocks) > 0:
    interval = blocks[-1]
    final_score_diff = np.average(scores[interval[0]:interval[1]])
```

[作者源码：NeurIPS 补充材料中的 mistral_7b.ipynb](https://proceedings.neurips.cc/paper_files/paper/2024/file/7a8e7fd295aa04eac4b470ae27f8785c-Supplemental-Conference.zip)

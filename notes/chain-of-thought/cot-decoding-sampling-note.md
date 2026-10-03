<aside class="cot-sampling-note" id="cot-sampling-note" aria-label="CoT-decoding 的候选生成">
<p><strong>CoT-decoding 的候选生成与路径筛选</strong></p>
<p>CoT-decoding 在首个解码位置枚举 top-k 候选 token，分别初始化候选路径；随后，各路径独立执行 greedy decoding，逐步选择条件概率最高的 token。该过程仅在首个位置展开候选，后续不再分叉，也不进行随机抽样。相比之下，top-k sampling 从截断并重新归一化的概率分布中随机抽取 token</p>
</aside>

`margins` 记录各位置 top-1 与 top-2 token 的概率差；`locate_answer` 按任务规则返回最终答案 token 的下标列表。伪代码中，`prompt` 为 token ID 列表，`next_logits` 返回完整词表 logits，`is_eos` 判断终止 token

```python 候选生成与答案置信度筛选 | 伪代码
def cot_decode(prompt, k=10, max_new_tokens=128):
    first_logits = next_logits(prompt)
    best_tokens, best_score = None, float("-inf")
    for first_token in first_logits.topk(k).indices:
        tokens, margins = [], []
        logits, token = first_logits, first_token
        for step in range(max_new_tokens):
            top2 = logits.float().softmax(dim=-1).topk(2).values
            margins.append((top2[0] - top2[1]).item())
            tokens.append(token.item())
            if is_eos(token) or step + 1 == max_new_tokens:
                break
            logits = next_logits(prompt + tokens)
            token = logits.argmax(dim=-1)
        answer_positions = locate_answer(tokens)
        if not answer_positions:
            continue
        score = sum(margins[t] for t in answer_positions)
        score /= len(answer_positions)
        if score > best_score:
            best_tokens, best_score = tokens, score
    return best_tokens
```

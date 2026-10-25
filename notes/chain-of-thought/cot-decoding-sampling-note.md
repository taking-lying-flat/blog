<aside class="cot-sampling-note" id="cot-sampling-note" aria-label="CoT-decoding 的候选生成">
<p><strong>CoT-decoding 的候选生成与路径筛选</strong></p>
<p>CoT-decoding 在首个解码位置枚举 top-k 候选 token，分别初始化候选路径；随后，各路径独立执行 greedy decoding，逐步选择条件概率最高的 token。该过程仅在首个位置展开候选，后续不再分叉，也不进行随机抽样。相比之下，top-k sampling 从截断并重新归一化的概率分布中随机抽取 token</p>
</aside>

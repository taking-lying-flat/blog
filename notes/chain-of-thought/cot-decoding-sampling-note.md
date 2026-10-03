<aside class="cot-sampling-note" id="cot-sampling-note" aria-label="CoT-decoding 的候选生成">
<p><strong>CoT-decoding 的候选生成</strong></p>
<p><strong>首个位置枚举 top-k 候选 token</strong>，每个候选各建立一条分支，而非只取概率最高的一个 token。<strong>从第二个位置开始，各分支分别进行 greedy decoding</strong>，即在自己的上下文中逐步选择条件概率最高的 token</p>
<p>这里的 sampling 指按概率随机抽取 token。top-k sampling 从保留并重新归一化的候选分布中随机抽取一个 token，重复采样时首个 token 可能相同；CoT-decoding 则显式枚举不同起点，再分别贪心续写，整个候选生成过程不进行随机抽样</p>
</aside>

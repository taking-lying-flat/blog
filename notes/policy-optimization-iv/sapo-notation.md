传统强化学习研究中已有对平滑门控函数的探索。在 SAPO 中，将这一思想引入面向大语言模型的基于组的强化学习范式，并在此基础上加入两个对大规模语言模型训练至关重要的组成部分：能够自然形成序列级一致性的 token 级软信赖域；基于正、负优势 token 更新行为差异的非对称温度设计

```math
\mathcal{J}(\theta)=\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}f_{i,t}(r_{i,t}(\theta))\widehat{A}_{i,t}\right]\tag{1}
```

其中，$`r_{i,t}(\theta)=\dfrac{\pi_\theta(y_{i,t}\mid q,y_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(y_{i,t}\mid q,y_{i,<t})}`$ 为 token 级重要性比率，同一回答共享组内归一化优势 $`\widehat A_{i,t}=\widehat A_i`$

期望对查询 $`q\sim\mathcal D`$ 与回答组 $`\{y_i\}_{i=1}^{G}\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid q)`$ 计算

**结果监督**

- 对于给定问题，从旧策略中采样 $`G`$ 条候选回答，并由奖励模型为每条完整回答赋予序列级奖励 $`r_i`$。组内奖励的均值与标准差定义为：

```math
\mu_r=\frac{1}{G}\sum_{i=1}^{G}r_i,\qquad
\sigma_r=\sqrt{\frac{1}{G}\sum_{i=1}^{G}(r_i-\mu_r)^2}.\tag{5}
```

- 将序列级奖励进行组内标准化，作为对应回答的优势估计；同一回答中的所有 token 共享相同的优势值：

```math
\hat A_{i,t}=\tilde r_i=\frac{r_i-\mu_r}{\sigma_r},
\qquad t=1,\ldots,|o_i|.\tag{6}
```

**过程监督**

- 过程奖励模型对各推理步骤赋予步骤级奖励。记第 $`i`$ 条回答包含 $`K_i`$ 个推理步骤，第 $`j`$ 步的终止位置为 $`t_{i,j}`$，对应奖励为 $`r_{i,j}`$。以同一问题下全部候选回答的步骤奖励为归一化集合，其样本数、均值与标准差定义为：

```math
N=\sum_{i=1}^{G}K_i,\qquad
\mu_s=\frac{1}{N}\sum_{i=1}^{G}\sum_{j=1}^{K_i}r_{i,j},\qquad
\sigma_s=\sqrt{\frac{1}{N}\sum_{i=1}^{G}\sum_{j=1}^{K_i}(r_{i,j}-\mu_s)^2}.\tag{7}
```

- 对步骤级奖励进行标准化后，将当前 token 所属步骤及后续步骤的标准化奖励累加，得到该 token 的优势估计：

```math
\tilde r_{i,j}=\frac{r_{i,j}-\mu_s}{\sigma_s},\qquad
\hat A_{i,t}=\sum_{j=1}^{K_i}\mathbf{1}\{t\le t_{i,j}\}\,\tilde r_{i,j}.\tag{8}
```

- 两种监督方式均将上述优势估计代入 GRPO 目标。若归一化所用的标准差为零，则将相应的标准化奖励置为零。

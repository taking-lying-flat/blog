**结果监督**：对同一问题，由旧策略采样 $`G`$ 个回答。奖励模型为每条完整回答给出一个分数 $`r_i`$，并在这组回答内计算均值与标准差：

```math
\mu_r=\frac{1}{G}\sum_{i=1}^{G}r_i,\qquad
\sigma_r=\sqrt{\frac{1}{G}\sum_{i=1}^{G}(r_i-\mu_r)^2}.\tag{5}
```

将序列奖励归一化后，同一回答中的所有 token 共享这一优势：

```math
\hat A_{i,t}=\tilde r_i=\frac{r_i-\mu_r}{\sigma_r},
\qquad t=1,\ldots,|o_i|.\tag{6}
```

**过程监督**：过程奖励模型在每个推理步骤结束时打分。记第 $`i`$ 个回答有 $`K_i`$ 个步骤，第 $`j`$ 步的结束位置为 $`t_{i,j}`$，奖励为 $`r_{i,j}`$。将同一问题下所有回答的步骤奖励放在一起归一化：

```math
N=\sum_{i=1}^{G}K_i,\qquad
\mu_s=\frac{1}{N}\sum_{i=1}^{G}\sum_{j=1}^{K_i}r_{i,j},\qquad
\sigma_s=\sqrt{\frac{1}{N}\sum_{i=1}^{G}\sum_{j=1}^{K_i}(r_{i,j}-\mu_s)^2}.\tag{7}
```

每个 token 的优势，是当前位置及后续步骤的归一化奖励之和：

```math
\tilde r_{i,j}=\frac{r_{i,j}-\mu_s}{\sigma_s},\qquad
\hat A_{i,t}=\sum_{j=1}^{K_i}\mathbf{1}\{t\le t_{i,j}\}\,\tilde r_{i,j}.\tag{8}
```

标准差为零时，将对应的归一化奖励置零。两种监督方式都使用上述优势优化 GRPO 目标。

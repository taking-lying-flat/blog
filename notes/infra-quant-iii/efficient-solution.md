**步骤 I：大规模基础模型庞大的参数空间启发一项重要优化：可以按照任意顺序处理权重，而不必严格遵循由 $`L_q`$ 确定的最优顺序**

- 在量化单个权重元素时，广阔的权重空间提供足够的灵活性来补偿量化误差。GPTQ 的实证结果支持这一认识：与任意顺序相比，遵循 $`L_q`$ 的最优降序排列仅能带来有限的改进

- 以任意顺序处理权重还具有一项实际优势：它允许并行处理所有行，从而高效利用 GPU 的计算能力。因此，采用 GPTQ 的列处理顺序，即从第一列到最后一列依次处理。对于每个 $`q=1,2,3,\ldots,n`$，通过下式同时计算所有行的权重更新量 $`\Delta\mathbf{W}`$

```math
\Delta\mathbf{W}=\frac{\hat{\mathbf{W}}_{:,q}-\mathbf{W}_{:,q}}{\mathbf{H}_{qq}^{-1}}\cdot\mathbf{H}_{q,:}^{-1}+\mathbf{R}\mathbf{X}^{\top}\mathbf{H}_{-q}^{-1}\tag{11}
```

- 其中，$`\mathbf{R}=\mathbf{W}\tilde{\mathbf{X}}-\mathbf{W}\mathbf{X}`$ 表示所有行，即所有输出通道的输出残差。采用这一形式，只需维护一份 $`\mathbf{H}_{-q}^{-1}`$，因为对于给定的 $`q`$，它在所有行之间保持不变。尽管这一方法实现行之间的并行处理，但仍存在两项显著的计算挑战：每次迭代都需要估计新的残差 $`\mathbf{R}`$，以及矩阵乘法 $`\mathbf{R}\mathbf{X}^{\top}\mathbf{H}_{-q}^{-1}`$ 带来的计算负担。要使该方法在大规模模型上达到实际可用的效率，必须解决这些问题

**步骤 II：通过逐通道分解，避免每轮迭代重新计算完整输出残差 $`\mathbf{R}`$**

- 尽管 $`\mathbf{R}\mathbf{X}^{\top}\mathbf{H}_{-q}^{-1}`$ 能够有效缩小量化模型与全精度模型输出之间的差距，但每次新的迭代都需要重新估计输出残差 $`\mathbf{R}`$。将输入激活的残差记为 $`\Delta\mathbf{X}=\tilde{\mathbf{X}}-\mathbf{X}`$，其更新公式为：

```math
\mathbf{R}_{\mathrm{new}}=\mathbf{R}_{-q}-\Delta\mathbf{W}\Delta\mathbf{X}_{-q}\tag{12}
```

- 这一操作会产生较大的计算开销，其复杂度为 $`O(mnk)`$，其中 $`k`$ 为 token 长度与校准样本数量的乘积。对 $`k`$ 的依赖给大型基础模型带来特殊挑战，因为 $`k`$ 通常是 $`m`$ 和 $`n`$ 的 $`10\sim50`$ 倍。提出一种替代方法，以避免重新计算输出残差 $`\mathbf{R}`$。关键认识在于，残差可以分解为来自不同神经元的 $`n`$ 个独立分量：

```math
\mathbf{R}=\mathbf{W}\Delta\mathbf{X}=\sum_{q=1}^{n}\mathbf{W}_{:,q}\Delta\mathbf{X}_{q,:}\tag{13}
```

- 通过这一分解，只需在层校准之前估计一次 $`\mathbf{R}`$。因此，在第 $`q`$ 次迭代中，前 $`q-1`$ 列已经完成量化，专注于量化第 $`q`$ 列，同时最小化与其对应的残差分量：

```math
\min_{\Delta\mathbf{W}_{:,q:}}\left\|\Delta\mathbf{W}_{:,q:}\mathbf{X}_{q:,:}-\mathbf{W}_{:,q}\Delta\mathbf{X}_{q,:}\right\|_F^2,\quad \text{s.t.}\quad \Delta\mathbf{W}_{:,q}\mathbf{e}_q^{\top}+\mathbf{W}_{:,q}-\hat{\mathbf{W}}_{:,q}=\mathbf{0}\tag{14}
```

- 类似地，第 $`q`$ 次迭代的最优权重更新量变为：

```math
\Delta\mathbf{W}_{:,q:}=\frac{\hat{\mathbf{W}}_{:,q}-\mathbf{W}_{:,q}}{\tilde{\mathbf{H}}_{qq}^{-1}}\cdot\tilde{\mathbf{H}}_{q,:}^{-1}+\mathbf{W}_{:,q}\Delta\mathbf{X}_{q,:}\mathbf{X}_{:,q:}^{\top}\tilde{\mathbf{H}}_{-q}^{-1}\tag{15}
```

- 其中，$`\tilde{\mathbf{H}}^{-1}=\mathbf{H}_{-(q-1):}^{-1}`$ 表示经过 $`q-1`$ 次迭代消元后的 Hessian 逆矩阵，因此 $`\tilde{\mathbf{H}}_{-q}^{-1}=\mathbf{H}_{-q:}^{-1}`$。值得注意的是，式（15）的第一项与 GPTQ 相同，而第二项用于最小化第 $`q`$ 个神经元的残差误差。由于 $`\mathbf{W}_{:,q}`$ 已经通过先前的 $`\Delta\mathbf{W}`$ 得到更新，因此无需重新估计残差误差

- 此外，残差分解还带来显著的计算优势，使第二项的计算复杂度降至原来的 $`1/n`$。这是因为 $`\Delta\mathbf{X}_{q,:}\mathbf{X}_{:,q:}^{\top}\mathbf{H}_{-q:}^{-1}\in\mathbb{R}^{1\times(n-q)}`$ 是一个行向量，而 $`\mathbf{W}_{:,q}`$ 是一个列向量，因此计算第二项仅需 $`O(mn)`$ 的复杂度。进一步注意到，$`\Delta\mathbf{X}_{q,:}\mathbf{X}_{:,q:}^{\top}\mathbf{H}_{-q:}^{-1}`$ 与权重更新无关，因此可以预先计算每一行并存储在矩阵 $`\mathbf{P}`$ 中。这样，在第 $`q`$ 次迭代中，就可以通过 $`\mathbf{W}_{:,q}\mathbf{P}_{q,:}`$ 高效计算第二项

**步骤 III：通过 Cholesky 重构，在保持数值稳定性的同时高效计算残差校正矩阵 $`\mathbf{P}`$**

- 重复的高斯消元会累积数值误差。对 Hessian 逆矩阵进行 Cholesky 分解 $`\mathbf{H}^{-1}=\mathbf{L}\mathbf{L}^{\top}`$，其中 $`\mathbf{L}`$ 为下三角因子。通过该因子的切片与乘法计算 $`\mathbf{H}_{-q:}^{-1}`$，可将 $`\mathbf{P}_{q,:}`$ 改写为：

```math
\mathbf{P}_{q,:}=\Delta\mathbf{X}_{q,:}\mathbf{X}^{\top}\mathbf{L}_{q+1:,q+1:}\mathbf{L}_{q+1:,q+1:}^{\top}\tag{16}
```

- 由于 $`\mathbf{X}_{:,q:}^{\top}\mathbf{H}_{-q:}^{-1}=\mathbf{X}^{\top}\mathbf{H}_{-q:}^{-1}`$，式（16）使用完整的 $`\mathbf{X}^{\top}`$。预先计算 $`\Delta\mathbf{X}\mathbf{X}^{\top}\in\mathbb{R}^{n\times n}`$，再利用 $`\mathbf{L}`$ 的下三角结构，在 GPU 上并行计算 $`\mathbf{P}`$ 的各行

**步骤 IV：预先计算矩阵 $`\mathbf{P}`$ 后，对两个权重更新项采用延迟批量处理，以提高 GPU 利用率**

- 与 GPTQ 一样，第 $`q`$ 列的量化结果仅受该列自身更新的影响，因此在当前处理阶段，对后续列的更新与该列的量化结果无关。由此，可以对两个更新项都采用延迟批量处理，以提高 GPU 利用率。具体而言，给定一个列块 $`Q`$，首先利用切片后的 $`\mathbf{P}`$ 和 $`\mathbf{L}`$ 更新块内各列，再通过下式更新块外的其余权重：

```math
\Delta\mathbf{W}_{:,Q:}=\frac{\hat{\mathbf{W}}_{:,Q}-\mathbf{W}_{:,Q}}{\mathbf{L}_{QQ}^{\top}}\mathbf{L}_{Q,Q:}^{\top}+\mathbf{W}_{:,Q}\mathbf{P}_{Q,Q:}\tag{17}
```

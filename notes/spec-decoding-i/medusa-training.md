<!-- Sources: Medusa, Sections 2.2.2 and 3.3.3, Appendix B.2–B.4 (cai24b.pdf, pp. 4–5, 9, 14).
Official implementation checked at ctlllll/axolotl commit 76bc572d69579b9bfc764c966b805210fb99dbbe:
examples/medusa/vicuna_7b_qlora_stage1.yml, vicuna_7b_qlora_stage2.yml,
vicuna_33b_together.yml, zephyr_together.yml; src/axolotl/monkeypatch/medusa_utils.py, lines 240–295.
The paper's "backbone loss" description of lambda_0 conflicts with its equation:
lambda_0 multiplies the Medusa-head loss. The B.4 sine schedule likewise scales the head loss.
-->

<p id="u78348d38"><strong><span style="color: #665800">MEDUSA-2 通过联合优化主干模型与预测头提高候选准确率，并通过分阶段训练或损失权重调度稳定主干更新</span></strong></p>

- **联合目标：** 在多步预测损失中加入下一 token 的交叉熵损失 $`\mathcal{L}_{\mathrm{LM}}=-\log p_t^{(0)}(y_{t+1})`$，得到

```math
\mathcal{L}_{\mathrm{MEDUSA\text{-}2}}=\mathcal{L}_{\mathrm{LM}}+\lambda_0\mathcal{L}_{\mathrm{MEDUSA\text{-}1}}\tag{4}
```

- $`\lambda_0`$ 控制 MEDUSA 预测头损失的整体权重，主干下一 token 损失的系数为 1；预测头损失也会通过隐藏状态反向传播至主干的可训练参数

**Vicuna-7B/13B 先训练预测头，再以 MEDUSA-1 检查点初始化联合训练**

- **第一阶段：** 冻结 4-bit 量化的主干模型，仅训练 MEDUSA 预测头，得到 MEDUSA-1
- **第二阶段：** 加载第一阶段检查点，通过 QLoRA 联合更新主干的 LoRA 适配器与 MEDUSA 预测头，量化后的基础权重保持冻结；$`\lambda_0=0.2`$，训练过程中保持固定
- **学习率：** 主干适配器的峰值学习率为 $`5\times10^{-4}`$，预测头为 $`2\times10^{-3}`$，后者为前者的 4 倍；学习率在前 40 步预热，随后按余弦计划衰减
- Vicuna-7B 消融实验中，两阶段训练的 MT-Bench 得分为 6.18，接近基线的 6.17，优于直接联合微调的 5.925；相对标准解码的加速比由 MEDUSA-1 的 2.18 倍提高至 MEDUSA-2 的 2.83 倍

**Vicuna-33B 与 Zephyr-7B 的自蒸馏实验直接联合训练，通过正弦调度逐步引入预测头损失**

- 不单独执行 MEDUSA-1 预训练阶段，而是同时训练主干的 LoRA 适配器与预测头。主干的下一 token 目标改为对齐原始模型输出分布的 KL 蒸馏损失，预测头仍使用交叉熵损失
- 将 $`\lambda_0`$ 设为 0.01，并在预测头损失前额外乘以从 0 递增至 1 的正弦调度因子，使有效权重在训练结束时达到 0.01；主干蒸馏损失的系数始终为 1
- 主干适配器的峰值学习率为 $`10^{-4}`$，预测头学习率为其 4 倍，学习率预热为 20 步。学习率预热调节参数更新步长，正弦调度则在整个训练过程中调节预测头损失的相对贡献

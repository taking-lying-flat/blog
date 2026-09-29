# Policy Optimization-II 发布校对

桌面 `无标题文档.lake` 归档为 `source.lake`，与源文件逐字节一致。文章包含 PPO 实践、DPO 和 DeepSeekMath／GRPO，保留 167 个公式、11 张图片、3 个一级标题和 5 个二级标题。公式、文字和代码图修订分别记录于独立覆盖清单，原始导出及原图保留。

## 校对来源

- [Secrets of RLHF in Large Language Models Part I: PPO](https://openlmlab.github.io/MOSS-RLHF/paper/SecretsOfRLHFPart1.pdf)
- [Direct Preference Optimization](https://arxiv.org/html/2305.18290v3)
- [DeepSeekMath](https://arxiv.org/html/2402.03300v3)
- [OpenRLHF 固定提交](https://github.com/OpenRLHF/OpenRLHF/tree/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53)

## 公式与表述

- 原 PPO 论文的奖励建模式（1）、（2）本身存在最小化损失与对数似然的符号不一致。本文统一为负对数似然：偏好项和语言模型项均取负号。
- 区分单条回答上的对数概率比惩罚与策略期望下的 KL；补齐 PPO-Penalty 和 DPO 目标中对状态／提示的期望范围。
- PPO-Clip 的裁剪对象改为新旧策略概率比，区间包含端点；避免将 clipped objective 描述成真实概率比或 KL 的硬约束。
- 有限无折扣轨迹的策略梯度下标与轨迹编号统一；即时奖励用 r，轨迹回报用 R。补充动作无关基线条件，避免将 actor-critic 与策略梯度写成互斥。
- DPO 奖励中心化采用逐提示条件期望；中心化固定零点，本身不降低条件方差。离散采样不可直接反传，不等于期望目标对策略参数不可微。
- DeepSeekMath 奖励的前缀下标由 `<t` 修正为 `≤t`。PPO／GRPO 目标明确采样分布和期望括号；token 级 KL 使用估计量记号，并说明当前策略采样下的无偏条件及复用旧策略样本的区别。
- 长公式采用分行排版，保留各论文的公式编号；编号靠右，公式居中。

## 代码图

- 原 DPO 小节误放 KTO 损失截图，替换为 OpenRLHF `DPOLoss.forward` 原函数。`ipo=False`、`label_smoothing=0` 对应标准 DPO。
- Experience 截图替换为同一提交中的字段声明节选，避免原图中的旧字段与张量维度注释混淆。
- 两张新图直接排版官方代码，去掉注释与文档字符串、调整换行；以 Python AST 确认计算逻辑和字段声明未改。源代码、固定提交链接与处理记录均保留；Apache 2.0 许可附于资源目录。
- `compute_reward` 使用导出内的原图，补充固定版本的官方源码链接。三张代码图均有一行 GitHub 来源。

## 发布验证

- 全站构建成功，资源哈希正确，公式无 MathJax 渲染错误。
- 桌面 1280 和 1920 像素下无页面或公式横向溢出、无损坏图片、无空白段落，标题与正文锚点不重复。
- 右侧目录包含一级和二级标题，可跳转、可隐藏。
- 更新后的图片渲染器对现有 14 份 Lake 文档输出相同内容。
- 详细检查结果见 `verification.json`。

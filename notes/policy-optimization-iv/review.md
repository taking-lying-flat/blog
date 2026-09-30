# Policy Optimization-IV 发布校对

桌面 `🦫 𝓡𝓮𝓯𝓲𝓷𝓮𝓶𝓮𝓷𝓽.lake` 完整归档为 `references/original.lake`；发布源 `source.lake` 只截取 DAPO，截止于 GSPO 标题之前，GSPO 和 SAPO 后续依次从 `gspo/source.lake` 与 `sapo/source.lake` 追加

- 对照 [DAPO 论文 v2](https://arxiv.org/html/2503.14476v2) 核对 PPO／GRPO 回顾、Clip-Higher、动态采样、Token 级损失和超长回答奖励塑形
- 保留正文与四项技术的高亮公式，公式按论文顺序编号（1）—（13），内部全部使用英文
- 优势公式中的分子统一为 R_i；补齐原论文长度惩罚公式中超过 L_max 时取 −1 的第三个分支
- 明确软性超长惩罚从 L_max−L_cache 开始，在 L_max 处达到 −1；将裁剪的说明限定为抑制过大更新，明确移除 KL 是 DAPO 的设计选择
- 调整多行公式对齐，去除段末标点及空白段落，修正等宽函数名中的转义显示，保持正文行距
- 右侧目录展示一级、二级标题；首页接在策略优化 I、II、III 后面，归入大模型分类
- 全站构建通过，1280 与 1920 像素桌面下无公式或页面横向溢出、无公式渲染错误、无空白段落，目录跳转和隐藏正常，详见 `verification.json`

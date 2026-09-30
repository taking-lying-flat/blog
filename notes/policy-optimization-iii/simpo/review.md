# SimPO 集成校对

桌面最新 `🐘 𝓐𝓵𝓲𝓰𝓷𝓶𝓮𝓷𝓽.lake` 完整归档为 `references/original.lake`，发布源只截取 SimPO 部分，接续 Policy Optimization-III 的 KTO 章节

- 对照 [SimPO 论文 §2](https://arxiv.org/html/2405.14734v2#S2) 核对 DPO 回顾、平均对数似然、长度归一化奖励、目标奖励间隔及完整目标；六个公式沿用（1）—（6），公式内部全部为英文
- 目标间隔的表述改为“鼓励”，避免将优化目标描述为每个样本均满足的硬约束；其余正文和两张论文图片保留，移除空段落与段末标点
- 函数图使用 [官方 SimPOTrainer.simpo_loss 固定提交](https://github.com/princeton-nlp/SimPO/blob/1b3e8f3528a23bce3da514a2dce8ea7490d4bc75/scripts/simpo_trainer.py#L560-L595) 的 sigmoid 分支，保留设备转换与返回值，移除注释、文档字符串、hinge 分支和异常处理，紧凑排版为 13 行
- Python AST 核对该节选与官方对应分支一致；上游调用启用 average_log_prob=True，gamma_beta_ratio=γ/β，label_smoothing=0 对应正文目标，保留一行 GitHub 来源和 MIT 许可证
- 对比导入前后的渲染 HTML，既有 ORPO、KTO 内容完全一致
- 全站构建通过；1280 和 1920 像素桌面下公式无横向溢出、无渲染错误，图片正常，无空白段落和重复标题锚点，新增两级目录可跳转及隐藏
- 详细排版检查见 `verification.json`

# DeepSeek AI-II 发布校对

桌面 mHC 文档完整归档为 `references/original.lake`，正文、三张原图及全部公式接入 `source.lake`

- 对照 [mHC 论文 v2](https://arxiv.org/html/2512.24880v2) 核对残差连接、HC 与 mHC 参数化、双随机矩阵约束、Sinkhorn–Knopp、混合精度内核融合及重计算公式
- 公式保留原论文 1—20 的连续编号，使用数学渲染器的右侧编号；定义等号改用兼容的等价写法，移除公式末尾标点
- 将 Efficient Infrastructure Design 标题移至内核融合段落之前，使内核融合、重计算及 DualPipe 调度归入同一节
- 保留原文强调方式和正文行距，清除空白段落，三张图居中
- 首页归入大模型分类，DeepSeek AI-I 与 AI-II 按系列顺序相邻展示；右侧目录包含一级与二级标题，支持跳转和隐藏
- 全站构建成功；1280 与 1920 像素桌面检查无横向溢出、公式渲染错误、失效图片或空白段落，公式内无中文，详见 `verification.json`

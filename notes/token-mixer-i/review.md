# Token Mixer-I 发布核验

桌面语雀导出文件按字节保存为 `source.lake`，原始资源及 SHA-256 校验值见 `manifest.json`。正文文字、标题、列表、强调、颜色及图片顺序保持原样；沿用博客现有正文字体、公式字体、行内公式间距与桌面换行规则。

## 核对依据

- [Gated Delta Networks，ICLR 2025 会议版](https://proceedings.iclr.cc/paper_files/paper/2025/file/4904fad153f6434a7bcf04465d4be2cc-Paper-Conference.pdf)：状态递推及第 3.3 节分块算法。
- [Kimi Linear，作者官方技术报告](https://github.com/MoonshotAI/Kimi-Linear/blob/master/tech_report.pdf)：第 2 节 KDA 递推、分块表示，以及附录中的位置编码与 DPLR 讨论。
- [Gated DeltaNet-2，作者官方论文](https://github.com/NVlabs/GatedDeltaNet-2/blob/main/paper/GDN2_paper.pdf)：式 (9)–(28) 的擦除/写入解耦、分块算法和反向传播。

部分论文简写本身存在指标或衰减遗漏，以下修正同时依据原文递推关系、矩阵维度和数值对照，不直接照抄有歧义的简写。

## 公式修正

仅通过 `math-overrides.json` 修改 11 个公式卡片，原始导出与原始 SVG 不变，每项记录完整修改前后内容及依据。

- `uvJ3c`：补齐累计衰减掩码的上三角零值，修正乘积的哑指标。
- `ANuyP`：补齐 DeltaNet 状态转移乘积因子的括号。
- `pc9Nl`：Gated DeltaNet 块内输出使用带衰减的因果掩码 Gamma，补回历史写入到读取之间的衰减。
- `b9LkU`：补齐块内时刻 r，H 的下括号覆盖完整求和；左乘状态转移使用逆序乘积，较晚的转移在左。
- `rZg6k`：H 的块内求和上界由块编号 t 改为块内时刻 r。
- `Rl8ez`：辅助向量 w 的递推统一为列向量，修正转置导致的维度不一致。
- `LpN7v`、`UqP7v`、`YFcfg`：统一绝对旋转、相邻相对旋转与二维旋转块；按文中的正角旋转定义，R_t^T R_i 对应角度 (i-t) theta。
- `mlJrq`：修正乘积外 k、v 的指标为 i，补齐写入门 beta_i，明确时间乘积次序和零初始状态前提。
- `pPvIQ`：在保持相同 k、v 定义的约束 DPLR 表示中，补齐写入项 beta_t。

数值核验中，块内累计衰减取 gamma^r = alpha^1 ⊙ ... ⊙ alpha^r；从第 i 步写入后至第 r 步的衰减为 gamma^r / gamma^i。此约定保证分块表示与逐步递推一致。

## 显示修复与验证

- 267 处公式、11 张图片、16 个正文标题及 797 个原始结构与格式元素完整保留；浏览器逐项核对正文文字、结构及属性。
- 241 个原始资源全部本地化并通过 SHA-256 校验，页面不依赖语雀 CDN，无源文件下载入口。
- `P6Cze`、`PKAt2`、`hXvxu` 的原始 SVG 含无效坐标，使用相同 LaTeX 重新渲染，保留公式编号。
- 构建器启用 MathJax 颜色扩展，保留修正公式中的彩色标记；修复独立 SVG 的属性转义，防止小于号造成 XML 解析失败。页面引用的全部 SVG 均通过 XML 与有效坐标检查。
- 32 组不同块长和矩阵维度、包含非零初始状态的计算中，GDN、KDA、GDN2 的分块与逐步递推、辅助向量和约束 DPLR 表示相符，最大绝对误差小于 `1.78e-15`。
- 16 组 RoPE 相对旋转检查的最大绝对误差小于 `1.82e-15`；三角矩阵求逆梯度与中心有限差分的误差小于 `6.96e-12`。
- Chrome 在 1024、1280、1440 像素宽度下无缺图、页面溢出、重复 ID 或浏览器错误；首页入口、行内公式间距和正文字体检查通过。
- 目视检查文章顶部、分块推导、RoPE 公式、DPLR 公式、编号公式及图片；长公式沿用现有横向滚动，符号不缩小。
- 原有 14 篇文章正文保持一致，比较时仅归一化 MathJax 全局编号及生成文件名；构建输出共 15 篇文章。

详细结果见 `verification.json`。数值检查针对相关公式，不替代一般性证明。

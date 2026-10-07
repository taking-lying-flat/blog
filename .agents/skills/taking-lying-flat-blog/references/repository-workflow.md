# 仓库接入方式

仓库通常位于 `/home/cherry-cloud/blog`，发布地址为 `https://taking-lying-flat.github.io/blog/`。以实际工作区为准。

## 文件与归档

- 文章配置：`site/posts.json`。保持已有 `slug`、类别、标签与论文顺序；新内容追加到用户指定文章。
- 原始笔记：`notes/<slug>/source.lake`。同一文章可用 `appendFiles` 加入独立子目录中的 Lake 导出。
- `manifest.json` 记录标题、原始卡片顺序、计数和本地资源。Math 卡片的 `value` 是 `data:` 后的 URL 编码 JSON，不是 Base64。
- 原式或文字的**已确认修改**分别放入 `math-overrides.json`、`text-overrides.json`，用 `from` 校验原始内容。不要直接改归档 `source.lake`。
- 新导入的 Math 卡片可以不设置 `asset`，由构建器从 LaTeX 渲染；保留 `assets/math/.gitkeep`。图片资源需本地归档并关联卡片。
- 原有 manifest 中可能有历史哈希字段；无需重算、校验或大规模清理它们。新增导入不生成哈希字段。

## 插入与替换

`appendMarkdown` 只接受文章主 Lake 文件同目录下的简单 `.md` 文件名，不支持子目录路径。字符串条目加到文章末尾；对象可在指定锚点前插入：

```json
{"file":"block-diffusion-algorithms.md","before":"I7Z1c"}
```

默认锚点为标题；需要段落/列表项锚点时使用现有 `tag` 支持并核对构建器。不要为了插入内容修改 URL 或丢掉原章节。

替换截图：先从源文件找出包含图片卡片的段落 ID，在 manifest 的 `blockRemovals` 中移除该展示段落，再在同一位置插入 LaTeX Markdown。原卡片和资源仍留在归档中，计数不修改。

```json
{"blockRemovals":[{"tag":"p","id":"原截图段落ID"}]}
```

新论文追加到原文章时，把属于旧论文的算法/补充材料定位在新论文 H1 之前，避免落入错误的论文目录。

Lake 段落与列表的结构调整可复用 manifest 的 `listItemMerges`、`paragraphMerges`、`paragraphListItems`、`listItemSplits` 等机制；先核对 `site/lake.mjs` 中的字段与锚点要求，保持源文件不变。现有 Lake 列表在 `site/lake.css` 中使用 `padding-left: 2em`，不要再叠加文本缩进或手工空格。

## 数学与算法模板

Markdown 行内数学使用 `$` 后紧跟反引号的语法，例如 $`x_t`$；展示数学使用 `math` 围栏。标签可用 `\tag{n}`，也可沿用已有按论文编号配置，避免重复编号。

需要统一编号时，先检查文章配置里是否已有 `numberedEquations` 或 `numberedEquationBlocks`：前者选 Math 卡片，后者可选段落 ID 或 Markdown 公式内容，均按论文分组。不要在配置编号之外再保留第二个 `\tag`。多行推导把 `\tag` 放在整个 `aligned` 环境之后；正文中的公式引用也需检查，不能只看右侧是否出现数字。

算法复用 `.paper-algorithm`，或所在文章已有的算法样式；不新建一套不同边框和字号的组件：

````markdown
<figure class="paper-algorithm" aria-labelledby="paper-algorithm-1">
<figcaption id="paper-algorithm-1"><strong>Algorithm 1</strong> Original Algorithm Title</figcaption>

```math
\begin{array}{r l}
&\textbf{Input:}\ \text{original inputs}\\[-0.25em]
1:&\text{original first step}\\[-0.25em]
2:&\textbf{return }\text{original output}
\end{array}
```

</figure>
````

参考成品：

- `notes/diffusion-language-models/llada-algorithms.md`
- `notes/diffusion-language-models/block-diffusion-algorithms.md`
- `notes/genai-sakura-iv/cm-distillation-bound.md`
- `notes/genai-sakura-iv/cm-training-equivalence.md`

## 构建与发布

```bash
npm run build --prefix site
git diff --check
```

构建器为 `site/build.mjs`，Lake 展示变换在 `site/lake.mjs`，样式在 `site/reader.css` 和 `site/lake.css`。构建结果 `site/dist` 已忽略，不提交它。

只提交本任务需要发布的文件，保留用户其他修改以及明确留在本地的推导。当前任务已授权发布时，提交后 `git push origin main`，再用 `gh run list --workflow pages.yml` 定位本次运行并等到 build、deploy 均成功。不要把上一次运行的成功当作本次已发布。

---
name: blog-quick-publish
description: 按用户请求推送 taking-lying-flat/blog 的修改，确认该提交对应的 GitHub Pages 部署结果。
---

# 快速提交与发布

1. 检查本次差异。页面、内容或构建器有改动时，按[仓库构建说明](../../../AGENTS.md)生成静态站点，验证受影响的功能。
2. 按用户授权提交并推送。仅推送仓库文档或技能时，核对远端提交即可。
3. 发布到 `main` 时，按[工作流的路径过滤条件](../../../.github/workflows/pages.yml)判断是否触发部署。用 `gh run list --workflow pages.yml --commit <SHA>` 查找本次运行，再用 `gh run watch <ID> --exit-status` 确认构建和部署成功。

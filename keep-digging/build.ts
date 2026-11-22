import { mkdir, writeFile, copyFile, cp } from 'node:fs/promises';
import path from 'node:path';

import { escape, renderPage } from '../site/src/html.ts';
import contributions from './contributions.json' with { type: 'json' };

export async function buildKeepDigging(source: string, output: string) {
  const models = [
    { name: 'Claude', icon: 'claude-color', x: 75, y: 55 },
    { name: 'ChatGPT', icon: 'openai', x: 215, y: 55 },
    { name: 'Gemini', icon: 'gemini-color', x: 355, y: 55 },
    { name: 'DeepSeek', icon: 'deepseek-color', x: 495, y: 55 },
    { name: 'Kimi', icon: 'kimi-color', x: 635, y: 55 },
    { name: 'Qwen', icon: 'qwen-color', x: 145, y: 155 },
    { name: '智谱 GLM', icon: 'zhipu-color', x: 285, y: 155 },
    { name: 'MiniMax', icon: 'minimax-color', x: 425, y: 155 },
    { name: 'Mistral', icon: 'mistral-color', x: 565, y: 155 },
    { name: 'Muse', icon: 'muse-color', x: 705, y: 155 },
  ];
  const modelNodes = models.map(model => `<span class="model-node" style="left:${model.x / 780 * 100}%;top:${model.y / 210 * 100}%" title="${escape(model.name)}" aria-hidden="true">
    <img src="assets/model-${model.icon}.svg" width="34" height="34" alt="">
  </span>`).join('\n');

  const contributionList = [...contributions.projects].sort((a, b) => b.count - a.count).map((project) => {
    const repository = `https://github.com/${project.repo}`;
    return `<li class="contribution-item">
      <a class="contribution-project" href="${repository}" target="_blank" rel="noopener noreferrer">
        <img src="assets/${escape(project.icon)}" width="36" height="36" alt="" loading="lazy" decoding="async">
        <span><strong>${escape(project.name)}</strong><small>${escape(project.repo.split('/', 1)[0] ?? project.repo)}</small></span>
      </a>
      <span class="contribution-count">
        <strong>${project.count}</strong><span>贡献</span>
      </span>
    </li>`;
  }).join('\n');

  const contributionPage = renderPage({
    title: 'Keep Digging · Blog', description: '开源项目贡献统计。', route: 'keep-digging/',
    body: `<section class="contributions" aria-label="开源贡献">
      <div class="model-network" role="img" aria-label="${escape(models.map(model => model.name).join('、'))} 的图标连线图">
        <svg class="model-connections" viewBox="0 0 780 210" fill="none" aria-hidden="true">
          <path d="M75 55C110 55 110 155 145 155S180 55 215 55 250 155 285 155 320 55 355 55 390 155 425 155 460 55 495 55 530 155 565 155 600 55 635 55 670 155 705 155"/>
          <path class="model-connection-secondary" d="M75 55H635M145 155H705"/>
        </svg>
        ${modelNodes}
      </div>
      <ul class="contribution-list">${contributionList}</ul>
    </section>`,
  });

  const directory = path.join(output, 'keep-digging');
  await mkdir(directory, { recursive: true });
  await writeFile(path.join(directory, 'index.html'), contributionPage);
  await copyFile(path.join(source, 'style.css'), path.join(directory, 'style.css'));
  await cp(path.join(source, 'assets'), path.join(directory, 'assets'), { recursive: true });
}

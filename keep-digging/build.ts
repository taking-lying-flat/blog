import { mkdir, writeFile, copyFile, cp } from 'node:fs/promises';
import path from 'node:path';

import { escape, renderPage } from '../site/src/html.ts';
import contributions from './contributions.json' with { type: 'json' };

export async function buildKeepDigging(source: string, output: string) {
  const models = [
    { name: 'Claude', icon: 'claude-color', x: 75, y: 100 },
    { name: 'ChatGPT', icon: 'openai', x: 225, y: 45 },
    { name: 'Gemini', icon: 'gemini-color', x: 390, y: 100 },
    { name: 'DeepSeek', icon: 'deepseek-color', x: 555, y: 45 },
    { name: 'Kimi', icon: 'kimi-color', x: 705, y: 100 },
    { name: 'Qwen', icon: 'qwen-color', x: 235, y: 160 },
    { name: 'Llama', icon: 'meta-color', x: 550, y: 160 },
  ];
  const modelNodes = models.map(model => `<span class="model-node" style="left:${model.x / 780 * 100}%;top:${model.y / 210 * 100}%" title="${model.name}" aria-hidden="true">
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
      <div class="model-network" role="img" aria-label="ChatGPT、Claude、Gemini、DeepSeek、Qwen、Llama 与 Kimi 的图标连线图">
        <svg class="model-connections" viewBox="0 0 780 210" fill="none" aria-hidden="true">
          <path d="M75 100C140 100 150 45 225 45S315 100 390 100 480 45 555 45 640 100 705 100"/>
          <path d="M75 100C140 100 160 160 235 160S315 100 390 100 475 160 550 160 640 100 705 100"/>
          <path class="model-connection-secondary" d="M225 45C270 90 195 115 235 160M555 45C510 90 595 115 550 160M235 160Q390 215 550 160"/>
          <g class="model-junctions"><circle cx="143" cy="75" r="3"/><circle cx="309" cy="73" r="3"/><circle cx="475" cy="73" r="3"/><circle cx="630" cy="76" r="3"/><circle cx="390" cy="187.5" r="3"/></g>
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

import { mkdir, readFile, writeFile, copyFile, cp, rm } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { readLake } from './lake.mjs';
import MarkdownIt from 'markdown-it';
import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';
import json from 'highlight.js/lib/languages/json';
import cpp from 'highlight.js/lib/languages/cpp';
import bash from 'highlight.js/lib/languages/bash';
import diff from 'highlight.js/lib/languages/diff';
import { mathjax } from '@mathjax/src/js/mathjax.js';
import { TeX } from '@mathjax/src/js/input/tex.js';
import { SVG } from '@mathjax/src/js/output/svg.js';
import { liteAdaptor } from '@mathjax/src/js/adaptors/liteAdaptor.js';
import { RegisterHTMLHandler } from '@mathjax/src/js/handlers/html.js';
import '@mathjax/src/js/util/asyncLoad/esm.js';
import '@mathjax/src/js/input/tex/base/BaseConfiguration.js';
import '@mathjax/src/js/input/tex/ams/AmsConfiguration.js';
import '@mathjax/src/js/input/tex/boldsymbol/BoldsymbolConfiguration.js';
import '@mathjax/src/js/input/tex/newcommand/NewcommandConfiguration.js';
import '@mathjax/src/js/input/tex/color/ColorConfiguration.js';

const root = path.dirname(fileURLToPath(import.meta.url));
const output = path.join(root, 'dist');
const siteUrl = 'https://taking-lying-flat.github.io/blog/';
const posts = JSON.parse(await readFile(path.join(root, 'posts.json'), 'utf8'))
  .sort((a, b) => b.date.localeCompare(a.date));
const categories = [
  { id: 'llm', title: '大模型' },
  { id: 'generative', title: '生成模型' },
  { id: 'code', title: '源码解读' },
  { id: 'issues', title: 'GitHub Issue' },
];
for (const post of posts) {
  if (!categories.some(({ id }) => id === post.category)) {
    throw new Error(`Missing or invalid article category: ${post.slug}`);
  }
}
const seriesGroups = new Map();
for (const post of posts) {
  if (!post.series) continue;
  if (!Number.isInteger(post.seriesOrder) || post.seriesOrder < 1) {
    throw new Error(`Invalid series order: ${post.slug}`);
  }
  const group = seriesGroups.get(post.series) ?? [];
  if (group.some(other => other.category !== post.category || other.seriesOrder === post.seriesOrder)) {
    throw new Error(`Inconsistent article series: ${post.series}`);
  }
  group.push(post);
  seriesGroups.set(post.series, group);
}
function groupBySeries(categoryPosts) {
  const groups = new Map();
  for (const post of categoryPosts) {
    const key = post.series ? `series:${post.series}` : `post:${post.slug}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(post);
  }
  return [...groups.values()].flatMap(group => group.sort((a, b) =>
    (a.seriesOrder ?? 0) - (b.seriesOrder ?? 0)));
}
const readingOrder = categories.flatMap(({ id }) =>
  groupBySeries(posts.filter(post => post.category === id)));
const codeCaptions = JSON.parse(await readFile(path.join(root, 'code-captions.json'), 'utf8'));
// Render article headings as written; code captions belong to the code block.
// Repository-authored HTML tables are rendered alongside Markdown.
const markdown = new MarkdownIt({ html: true, typographer: false });
const escape = markdown.utils.escapeHtml;
for (const [name, language] of Object.entries({ python, json, cpp, bash, diff })) {
  hljs.registerLanguage(name, language);
}
const languageNames = { python: 'Python', json: 'JSON', cpp: 'C++ / CUDA', bash: 'Shell', diff: 'Diff', text: 'Text' };

markdown.inline.ruler.before('backticks', 'math_inline', (state, silent) => {
  if (!state.src.startsWith('$`', state.pos)) return false;
  const end = state.src.indexOf('`$', state.pos + 2);
  if (end === -1) throw new Error('Unclosed inline math delimiter');
  if (!silent) {
    const token = state.push('math_inline', 'span', 0);
    token.content = state.src.slice(state.pos + 2, end);
  }
  state.pos = end + 2;
  return true;
});

const adaptor = liteAdaptor({ fontSize: 18 });
RegisterHTMLHandler(adaptor);
const svg = new SVG({ fontCache: 'local', displayOverflow: 'overflow', linebreaks: { inline: false } });
const document = mathjax.document('', {
  InputJax: new TeX({
    packages: ['base', 'ams', 'boldsymbol', 'newcommand', 'color'],
    tagSide: 'right',
    formatError(_jax, error) { throw error; },
  }),
  OutputJax: svg,
});
let inlineCount = 0;
let displayCount = 0;

async function renderMath(token, display) {
  const node = await document.convertPromise(token.content, { display, em: 18, ex: 8.1, containerWidth: 1024 });
  const html = adaptor.outerHTML(node);
  if (/data-mjx-error|data-mml-node="merror"/.test(html)) throw new Error(`Invalid formula: ${token.content}`);
  token.meta = { html };
  if (display) displayCount++;
  else inlineCount++;
}

async function renderLakeMath(code, original) {
  const node = await document.convertPromise(code, { display: true, em: 16, ex: 8, containerWidth: 1024 });
  const html = /\\tag\b/.test(code) ? adaptor.outerHTML(node) : undefined;
  const svgNode = adaptor.tags(node, 'svg')[0];
  if (!svgNode) throw new Error(`Missing SVG for Lake formula: ${code}`);
  // Tagged equations use a percentage width; external images need the
  // concrete width from the export to retain their original alignment.
  if (adaptor.getAttribute(svgNode, 'width').endsWith('%')) adaptor.setAttribute(svgNode, 'width', original.width);
  // MathJax's HTML serializer leaves '<' in data-latex attributes; standalone
  // SVG images use XML, which requires those attribute characters to be escaped.
  const content = adaptor.outerHTML(svgNode).replace(/="([^"]*)"/g,
    (_attribute, value) => `="${value.replaceAll('<', '&lt;')}"`);
  if (/data-mjx-error|data-mml-node="merror"|\b(?:NaN|Infinity)\b/.test(content)) {
    throw new Error(`Invalid rendered Lake formula: ${code}`);
  }
  return {
    content,
    html,
    width: adaptor.getAttribute(svgNode, 'width'),
    height: adaptor.getAttribute(svgNode, 'height'),
    style: adaptor.getAttribute(svgNode, 'style') ?? '',
  };
}

const inlineText = (token) => (token?.children ?? []).map((child) =>
  child.type === 'softbreak' || child.type === 'hardbreak' ? ' ' : child.content).join('');
const slugify = (label) => label.toLowerCase().replace(/[^\p{L}\p{N}\p{M}_\-\s]/gu, '').replace(/\s/g, '-');
// Normalize decorative title letters so the chosen web font renders every title.
const plainTitle = (title) => title.replace(/[\u{1D400}-\u{1D7FF}\u210E]/gu,
  (letter) => letter.normalize('NFKC'));
function normalizeHeadingText(content) {
  return content.replace(/<h([1-6])\b[^>]*>[\s\S]*?<\/h\1>/gi, (heading) => {
    if (plainTitle(heading) === heading) return heading;
    const parsed = adaptor.parse(heading, 'text/html');
    const visit = (node) => {
      const kind = adaptor.kind(node);
      if (kind === '#text') {
        adaptor.replace(adaptor.text(plainTitle(adaptor.value(node))), node);
        return;
      }
      if (['#comment', 'svg', 'math', 'pre', 'code'].includes(kind) ||
          /(?:^|\s)(?:lake-math|math-inline|math-display|equation)(?:\s|$)/
            .test(adaptor.getAttribute(node, 'class') ?? '')) return;
      for (const child of [...adaptor.childNodes(node)]) visit(child);
    };
    visit(adaptor.body(parsed));
    return adaptor.innerHTML(adaptor.body(parsed));
  });
}
// English terms remain prose even when an export splits off k/p as a math card.
function normalizeEnglishTerms(content) {
  const parsed = adaptor.parse(content, 'text/html');
  const body = adaptor.body(parsed);
  const mathClass = /(?:^|\s)(?:lake-math|math-inline|math-display|equation)(?:\s|$)/;
  const inlineTags = new Set(['span', 'a', 'strong', 'b', 'em', 'i', 'u', 's', 'small', 'mark']);
  const excluded = new Set(['#comment', 'pre', 'svg', 'math', 'script', 'style']);
  let previousText = null;
  const repair = (node) => {
    const kind = adaptor.kind(node);
    if (kind === '#text') { previousText = node; return; }
    if (excluded.has(kind) || kind === 'code') { previousText = null; return; }
    if (mathClass.test(adaptor.getAttribute(node, 'class') ?? '')) {
      const image = adaptor.tags(node, 'img')[0];
      const tex = (adaptor.getAttribute(node, 'aria-label') ??
        (image ? adaptor.getAttribute(image, 'alt') : '') ?? '').trim();
      if (/^[kKpP]$/.test(tex) && previousText &&
          /(?<![A-Za-z0-9_])top[-‐‑–]\s*$/i.test(adaptor.value(previousText))) {
        const text = adaptor.value(previousText).trimEnd() + tex + ' ';
        adaptor.replace(adaptor.text(text), previousText);
        adaptor.remove(node);
      }
      previousText = null;
      return;
    }
    if (!inlineTags.has(kind)) previousText = null;
    for (const child of [...adaptor.childNodes(node)]) repair(child);
    if (!inlineTags.has(kind)) previousText = null;
  };
  repair(body);
  const wrap = (node) => {
    const kind = adaptor.kind(node);
    if (kind === '#text') {
      const text = adaptor.value(node);
      const terms = /(?<![A-Za-z0-9_])(?:top[-‐‑–](?:[kp]|\d+)|n[-‐‑–]grams?)(?![A-Za-z0-9_])/gi;
      let end = 0;
      for (const match of text.matchAll(terms)) {
        if (match.index > end) adaptor.insert(adaptor.text(text.slice(end, match.index)), node);
        adaptor.insert(adaptor.node('span', { class: 'english-term' },
          [adaptor.text(match[0])]), node);
        end = match.index + match[0].length;
      }
      if (end) {
        if (end < text.length) adaptor.insert(adaptor.text(text.slice(end)), node);
        adaptor.remove(node);
      }
      return;
    }
    if (excluded.has(kind) || mathClass.test(adaptor.getAttribute(node, 'class') ?? '') ||
        /(?:^|\s)english-term(?:\s|$)/.test(adaptor.getAttribute(node, 'class') ?? '')) return;
    for (const child of [...adaptor.childNodes(node)]) wrap(child);
  };
  wrap(body);
  return adaptor.innerHTML(body);
}
const dateLabel = (date) => new Intl.DateTimeFormat('zh-CN', {
  dateStyle: 'long', timeZone: 'UTC',
}).format(new Date(`${date}T00:00:00Z`));
const metadata = (post) => `<time datetime="${post.date}">${dateLabel(post.date)}</time><span>约 ${post.readingMinutes} 分钟</span>`;

function removeManualToc(tokens) {
  for (let i = 0; i < tokens.length - 3; i++) {
    if (!['heading_open', 'paragraph_open'].includes(tokens[i].type)) continue;
    if (inlineText(tokens[i + 1]).trim() !== '目录') continue;
    if (!['bullet_list_open', 'ordered_list_open'].includes(tokens[i + 3].type)) continue;
    let end = i + 3;
    let depth = 0;
    do { depth += tokens[end++].nesting; } while (depth > 0 && end < tokens.length);
    tokens.splice(i, end - i);
    i--;
  }
}

function prepareCallouts(tokens) {
  for (let i = 0; i < tokens.length; i++) {
    if (tokens[i].type !== 'blockquote_open' || tokens[i + 1]?.type !== 'paragraph_open') continue;
    const inline = tokens[i + 2];
    const match = inline?.content.match(/^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\](?:\n|$)/);
    if (!match) continue;
    tokens[i].attrSet('class', `callout callout-${match[1].toLowerCase()}`);
    inline.content = inline.content.slice(match[0].length);
    inline.children = [];
    markdown.inline.parse(inline.content, markdown, {}, inline.children);
  }
}

function cleanProsePunctuation(tokens) {
  // Leave code, math, link targets and HTML attributes intact.
  for (const token of tokens) {
    for (const child of token.children ?? []) {
      if (child.type === 'text') child.content = child.content.replace(/[“”‘’「」『』]/gu, '');
    }
  }
  for (let i = 0; i < tokens.length; i++) {
    if (tokens[i].type !== 'paragraph_open' || tokens[i + 1]?.type !== 'inline') continue;
    const children = tokens[i + 1].children ?? [];
    for (let j = children.length - 1; j >= 0; j--) {
      const child = children[j];
      if (child.nesting === -1 || ['softbreak', 'hardbreak'].includes(child.type)) continue;
      if (child.type === 'text' && !child.content.trim()) continue;
      if (child.type === 'html_inline' && /^<\/[\w:-]+\s*>$/.test(child.content)) continue;
      if (child.type === 'text') child.content = child.content.replace(/[。：:]+(?=\s*$)/u, '');
      break;
    }
  }
}

const tableWrapper = '<div class="table-scroll" role="region" aria-label="表格" tabindex="0">';
function renderCodeLines(highlighted, source) {
  const openSpans = [];
  const lines = [];
  let line = '';
  // Close and reopen highlighting spans so multiline strings retain their colors.
  for (const part of highlighted.replace(/\n$/, '').split(/(\n|<span\b[^>]*>|<\/span>)/)) {
    if (part === '\n') {
      lines.push(line + '</span>'.repeat(openSpans.length));
      line = openSpans.join('');
    } else {
      if (part.startsWith('<span')) openSpans.push(part);
      else if (part === '</span>') openSpans.pop();
      line += part;
    }
  }
  lines.push(line);
  const sourceLines = source.replace(/\n$/, '').split('\n');
  return lines.map((html, index) => {
    const indent = Math.min(sourceLines[index].match(/^ */)[0].length + 4, 16);
    return `<span class="code-line" data-line="${index + 1}" style="--wrap-indent:${indent}ch">${html}</span>`;
  }).join('\n') + (source.endsWith('\n') ? '\n' : '');
}

markdown.renderer.rules.table_open = () => `${tableWrapper}<table>\n`;
markdown.renderer.rules.table_close = () => '</table></div>\n';
markdown.renderer.rules.html_block = (items, i) => items[i].content.includes('<table')
  ? `${tableWrapper}${items[i].content}</div>\n` : items[i].content;
markdown.renderer.rules.math_inline = (items, i) =>
  `<span class="math-inline" role="math" aria-label="${escape(items[i].content)}">${items[i].meta.html}</span>`;
markdown.renderer.rules.fence = (items, i, _options, env) => {
  const token = items[i];
  const language = token.info.trim().split(/\s+/)[0] || 'text';
  if (language === 'math') {
    return `<div class="equation" tabindex="0" role="math" aria-label="${escape(token.content)}">${token.meta.html}</div>\n`;
  }
  const id = token.attrGet('id');
  const label = languageNames[language] ?? language;
  const [declaredCaption = '', declaredSource = ''] = token.info.trim().replace(/^\S+\s*/, '').split('|').map((part) => part.trim());
  const details = codeCaptions[env.slug]?.[token.map?.[0]];
  const isDiagram = language === 'text' || details?.kind === 'diagram';
  if (codeCaptions[env.slug] && !isDiagram && (!details?.caption || !details?.url)) {
    throw new Error(`Missing code source: ${env.slug}:${token.map[0] + 1}`);
  }
  if (details && createHash('sha256').update(token.content).digest('hex') !== details.sha256) {
    throw new Error(`Code caption needs updating: ${env.slug}:${token.map[0] + 1}`);
  }
  const caption = details?.caption ?? declaredCaption;
  const source = isDiagram ? '' : (details?.sourceLabel ?? declaredSource);
  const sourceUrl = isDiagram ? undefined : (details?.url ?? env.codeSourceUrl);
  const name = caption || (env.slug === 'rope'
    ? (language === 'json' ? 'config.json · text_config' : language === 'text' ? '张量维度'
      : token.content.includes('def rotate_half') ? 'rotate_half / apply_rotary_pos_emb'
      : token.content.includes('base =') && token.content.includes('inv_freq =')
        ? 'compute_default_rope_parameters' : label)
    : label);
  const code = hljs.getLanguage(language)
    ? hljs.highlight(token.content, { language, ignoreIllegals: true }).value : escape(token.content);
  const displayedCode = isDiagram ? code : renderCodeLines(code, token.content);
  const primarySource = sourceUrl
    ? `<a href="${escape(sourceUrl)}">${escape(source === 'GitHub' ? source : `${source} · GitHub`)}</a>` : escape(source);
  const sourceLabel = [primarySource, ...(details?.additionalSources ?? []).map((item) =>
    `<a href="${escape(item.url)}">${escape(item.label)}</a>`)].join(' · ');
  return `<figure class="code-block${isDiagram ? ' code-diagram' : ''}"${id ? ` id="${escape(id)}"` : ''}>
    <figcaption class="code-caption">
      <span class="code-caption-text">${escape(name)}${source ? `<span class="code-source">${sourceLabel}</span>` : ''}</span>
      <span class="code-actions"><button type="button" class="copy-button" hidden>复制</button></span>
    </figcaption>
    <pre tabindex="0"><code class="language-${escape(language)}">${displayedCode}</code></pre>
  </figure>\n`;
};

const ropeSections = [
  { id: 'frequency', prefix: 'RoPE（Rotary' },
  { id: 'configuration', fence: 'json' },
  { id: 'rotation', prefix: 'RoFormer §3.2.1' },
  { id: 'relative-position', prefix: 'RoFormer 式（15）' },
  { id: 'partial-rope', prefix: 'Qwen3.5 的 partial RoPE' },
  { id: 'implementation', prefix: '对普通 RoPE' },
];

for (const post of posts) {
  post.route = `posts/${post.slug}/`;
  if (post.format === 'lake') {
    Object.assign(post, await readLake(path.join(root, post.file), escape, renderLakeMath));
    post.directories = [post.directory];
    for (const file of post.appendFiles ?? []) {
      const addition = await readLake(path.join(root, file), escape, renderLakeMath);
      post.content += addition.content;
      post.readingMinutes += addition.readingMinutes;
      post.directories.push(addition.directory);
      post.generatedAssets.push(...addition.generatedAssets);
    }
    post.titleId = slugify(post.title);
    const inserts = JSON.parse(await readFile(path.join(post.directory, 'code-inserts.json'), 'utf8')
      .catch((error) => { if (error.code === 'ENOENT') return '[]'; throw error; }));
    for (const insert of inserts) {
      const tag = insert.tag ?? 'p';
      if (!/^[\w-]+$/.test(insert.id) || !/^[\w-]+$/.test(insert.after) ||
          !['p', 'li'].includes(tag) ||
          (insert.description !== undefined && typeof insert.description !== 'string')) {
        throw new Error(`Invalid Lake code anchor: ${post.slug}`);
      }
      const source = await readFile(path.join(post.directory, insert.file), 'utf8');
      const tokens = markdown.parse(source, {});
      if (tokens.length !== 1 || tokens[0].type !== 'fence') {
        throw new Error(`Expected one code block: ${insert.file}`);
      }
      tokens[0].attrSet('id', insert.id);
      const content = markdown.renderer.render(tokens, markdown.options, { slug: post.slug, codeSourceUrl: insert.source });
      const intro = insert.description ? `<p class="code-intro">${escape(insert.description)}</p>` : '';
      const addition = intro + content.trim();
      const anchor = new RegExp(`(<${tag}\\b[^>]*\\sid="${insert.after}"[^>]*>)([\\s\\S]*?)(</${tag}>)`, 'g');
      let matches = 0;
      post.content = post.content.replace(anchor, (block, opening, body, closing) => {
        matches++;
        // Keep code for a list item inside that item, preserving valid list markup.
        return tag === 'li' ? opening + body + addition + closing : block + addition;
      });
      if (matches !== 1) throw new Error(`Missing or repeated Lake code anchor: ${insert.after}`);
    }
    const replacements = JSON.parse(await readFile(path.join(post.directory, 'section-replacements.json'), 'utf8')
      .catch((error) => { if (error.code === 'ENOENT') return '[]'; throw error; }));
    for (const replacement of replacements) {
      if (![replacement.id, replacement.start, replacement.before].every((id) => /^[\w-]+$/.test(id)) ||
          !/^[\w-]+\.md$/.test(replacement.file)) {
        throw new Error(`Invalid Lake section replacement: ${post.slug}`);
      }
      const source = await readFile(path.join(post.directory, replacement.file), 'utf8');
      const tokens = markdown.parse(source, {});
      for (const token of tokens) {
        if (token.type === 'fence' && token.info.trim() === 'math') await renderMath(token, true);
        for (const child of token.children ?? []) {
          if (child.type === 'math_inline') await renderMath(child, false);
        }
      }
      const html = markdown.renderer.render(tokens, markdown.options, { slug: post.slug });
      const anchor = new RegExp(`<p\\b[^>]*\\sid="${replacement.start}"[^>]*>[\\s\\S]*?(?=<p\\b[^>]*\\sid="${replacement.before}"[^>]*>)`, 'g');
      if ([...post.content.matchAll(anchor)].length !== 1) {
        throw new Error(`Missing or repeated Lake section: ${replacement.id}`);
      }
      post.content = post.content.replace(anchor,
        () => `<section class="lake-revised-section" id="${replacement.id}">${html.trim()}</section>`);
    }
    for (const entry of post.appendMarkdown ?? []) {
      const { file, before } = typeof entry === 'string' ? { file: entry } : entry;
      if (!/^[\w-]+\.md$/.test(file) || (before !== undefined && !/^[\w-]+$/.test(before))) {
        throw new Error(`Invalid Lake Markdown addition: ${file}`);
      }
      const source = await readFile(path.join(post.directory, file), 'utf8');
      const tokens = markdown.parse(source, {});
      cleanProsePunctuation(tokens);
      for (let i = 0; i < tokens.length; i++) {
        const token = tokens[i];
        if (token.type === 'heading_open') {
          const id = slugify(plainTitle(inlineText(tokens[i + 1])).replace(/[\uFE0E\uFE0F]/g, '')).replace(/^-+|-+$/g, '');
          if (!id || post.content.includes(` id="${id}"`)) throw new Error(`Duplicate Lake heading: ${id}`);
          token.attrSet('id', id);
          if (token.tag === 'h2') token.attrSet('style', 'text-align: center');
        }
        if (token.type === 'fence' && token.info.trim() === 'math') await renderMath(token, true);
        for (const child of token.children ?? []) {
          if (child.type === 'math_inline') await renderMath(child, false);
        }
      }
      const html = markdown.renderer.render(tokens, markdown.options, { slug: post.slug });
      const addition = `<section class="lake-revised-section lake-appendix">${html.trim()}</section>`;
      if (before !== undefined) {
        const anchor = new RegExp(`<h[1-6]\\b[^>]*\\sid="${before}"[^>]*>`, 'g');
        if ([...post.content.matchAll(anchor)].length !== 1) {
          throw new Error(`Missing or repeated Lake Markdown anchor: ${before}`);
        }
        post.content = post.content.replace(anchor, (heading) => addition + heading);
      } else {
        post.content += addition;
      }
      const prose = tokens.filter(token => token.type === 'inline').map(inlineText).join(' ');
      post.readingMinutes += Math.ceil((prose.match(/\p{Script=Han}/gu)?.length ?? 0) / 300 +
        (prose.match(/[A-Za-z]+/g)?.length ?? 0) / 200);
    }
    if (post.removeEmptyParagraphs) {
      // Exported spacer paragraphs add a blank line on top of section margins.
      // Keep media, formulas and anchors; discard only empty text formatting.
      post.content = post.content.replace(/<p\b[^>]*>([\s\S]*?)<\/p>/gi, (paragraph, inner) => {
        const visible = inner
          .replace(/<\/?(?:span|strong|em|b|i|u)\b[^>]*>|<br\s*\/?>/gi, '')
          .replace(/&nbsp;|&#(?:160|x0*a0|8203|x0*200b);/gi, '')
          .replace(/[\s\u200b-\u200d\ufeff]/g, '');
        return visible ? paragraph : '';
      });
    }
    continue;
  }
  post.source = await readFile(path.join(root, post.file), 'utf8');
  const tokens = markdown.parse(post.source, {});
  if (tokens[0]?.type !== 'heading_open' || tokens[0].tag !== 'h1') throw new Error(`Expected document title: ${post.file}`);
  post.title = inlineText(tokens[1]);
  post.titleId = slugify(post.title);
  tokens.splice(0, 3);
  removeManualToc(tokens);
  prepareCallouts(tokens);
  cleanProsePunctuation(tokens);
  const usedIds = new Map([[post.titleId, 1]]);
  for (let i = 0; i < tokens.length; i++) {
    if (tokens[i].type !== 'heading_open') continue;
    const label = inlineText(tokens[i + 1]);
    const baseId = slugify(label);
    const occurrence = usedIds.get(baseId) ?? 0;
    usedIds.set(baseId, occurrence + 1);
    const id = occurrence ? `${baseId}-${occurrence}` : baseId;
    tokens[i].attrSet('id', id);
  }
  if (post.slug === 'rope') {
    for (const section of ropeSections) {
      const index = tokens.findIndex((token, i) => section.fence
        ? token.type === 'fence' && token.info === section.fence
        : token.type === 'paragraph_open' && inlineText(tokens[i + 1]).startsWith(section.prefix));
      if (index < 0) throw new Error(`Missing section: ${section.id}`);
      tokens[index].attrSet('id', section.id);
    }
  }
  const previousDisplays = displayCount;
  for (const token of tokens) {
    if (token.type === 'fence' && token.info.trim() === 'math') await renderMath(token, true);
    for (const child of token.children ?? []) {
      if (child.type === 'math_inline') await renderMath(child, false);
    }
  }
  const proseText = tokens.filter((token) => token.type === 'inline').map(inlineText).join(' ');
  post.readingMinutes = Math.max(1, Math.ceil(
    (proseText.match(/\p{Script=Han}/gu)?.length ?? 0) / 300 +
    (proseText.match(/[A-Za-z]+/g)?.length ?? 0) / 200 + (displayCount - previousDisplays) * 0.3 + 2
  ));
  const firstParagraph = tokens.findIndex((token, i) => token.type === 'paragraph_open' && tokens[i + 1]?.content);
  post.description = inlineText(tokens[firstParagraph + 1]).slice(0, 180) || post.title;
  post.content = markdown.renderer.render(tokens, markdown.options, { slug: post.slug });
}

// Keep the existing anchors while presenting all titles in the same typeface.
for (const post of posts) {
  post.title = plainTitle(post.title);
  post.content = normalizeEnglishTerms(normalizeHeadingText(post.content));
}

function renderPaperToc(post) {
  if (!post.tocDepth) return '';
  if (![1, 2].includes(post.tocDepth)) throw new Error(`Invalid TOC depth: ${post.slug}`);
  const parsed = adaptor.parse(post.content, 'text/html');
  const headings = [];
  const visit = (node) => {
    const kind = adaptor.kind(node);
    if (kind === 'h1' || (kind === 'h2' && post.tocDepth === 2)) {
      headings.push({
        level: Number(kind.slice(1)),
        id: adaptor.getAttribute(node, 'id'),
        title: plainTitle(adaptor.textContent(node)).replace(/\s+/g, ' ').trim(),
      });
      return;
    }
    if (['#text', '#comment', 'svg', 'pre', 'code'].includes(kind)) return;
    for (const child of adaptor.childNodes(node)) visit(child);
  };
  visit(adaptor.body(parsed));
  if (!headings.length || headings.some(h => !h.id || !h.title) ||
      new Set(headings.map(h => h.id)).size !== headings.length) {
    throw new Error(`Missing or duplicate paper headings: ${post.slug}`);
  }
  const groups = [];
  for (const heading of headings) {
    if (heading.level === 1) groups.push({ heading, sections: [] });
    else {
      if (!groups.length) throw new Error(`Paper section precedes its title: ${post.slug}`);
      groups.at(-1).sections.push(heading);
    }
  }
  const link = (heading) => `<a class="toc-link" href="#${escape(encodeURIComponent(heading.id))}">${escape(heading.title)}</a>`;
  return `<aside class="post-toc" aria-label="文章目录">
    <details class="toc-panel" open>
      <summary><span>目录</span><span class="toc-hide">隐藏</span><span class="toc-show">显示</span></summary>
      <nav aria-label="论文及章节">
        <ol class="toc-list">${groups.map(({ heading, sections }) => `<li>${link(heading)}${sections.length
          ? `<ol>${sections.map(section => `<li>${link(section)}</li>`).join('')}</ol>` : ''}</li>`).join('')}</ol>
      </nav>
    </details>
  </aside>`;
}

const template = await readFile(path.join(root, 'template.html'), 'utf8');
const primer = path.join(root, 'node_modules/@primer/primitives');
const assets = new Map([
  ...['reader.css', 'reader.js', 'theme.js', 'favicon.svg', 'lake.css', 'lake.js', 'body-serif.css', 'toc.css', 'toc.js'].map((file) => [file, path.join(root, file)]),
  ['anime-readers.png', path.join(root, 'illustrations/anime-readers.png')],
  ['gdn-architecture.png', path.join(root, 'illustrations/gdn-architecture.png')],
  ['gdn-chunk-parallel.png', path.join(root, 'illustrations/gdn-chunk-parallel.png')],
  ['gdn-model-flow.png', path.join(root, 'illustrations/gdn-model-flow.png')],
  ...['light', 'dark'].map((mode) => [`github-${mode}-tritanopia.css`,
    path.join(primer, `dist/css/functional/themes/${mode}-tritanopia.css`)]),
]);
const assetVersion = createHash('sha256');
for (const file of assets.values()) assetVersion.update(await readFile(file));
const version = assetVersion.digest('hex').slice(0, 10);

function page({ title, description, route = '', body, type = 'website', pageClass = '', hasToc = false }) {
  const values = {
    TITLE: escape(title), DESCRIPTION: escape(description), TYPE: type,
    URL: `${siteUrl}${route}`, ROOT: '../'.repeat(route.split('/').filter(Boolean).length) || './',
    POSTS_CURRENT: route === '' ? ' aria-current="page"' : '',
    ARCHIVES_CURRENT: route === 'archives/' ? ' aria-current="page"' : '',
    PAGE_CLASS: pageClass, TOC_CLASS: hasToc ? 'has-paper-toc' : '', BODY: body,
  };
  let html = template.replace(/\{\{([A-Z_]+)\}\}/g, (_, key) => {
    if (!(key in values)) throw new Error(`Unknown template field: ${key}`);
    return values[key];
  });
  for (const file of assets.keys()) html = html.replaceAll(`assets/${file}"`, `assets/${file}?v=${version}"`);
  return html;
}

const home = page({
  title: 'Blog · 技术笔记', description: '关于模型、论文与源码的技术笔记。', pageClass: 'home-page',
  body: `<section aria-labelledby="post-list-title">
    <div class="post-list-heading"><h1 id="post-list-title" aria-live="polite">全部文章 <span>${String(posts.length).padStart(2, '0')}</span></h1></div>
    <nav class="category-nav" aria-label="文章分类">
      <button type="button" data-category="all" aria-pressed="true">全部</button>
      ${categories.map(({ id, title }) =>
      `<button type="button" data-category="${id}" aria-pressed="false">${escape(title)}</button>`).join('')}</nav>
    ${categories.map(({ id, title }) => {
      const categoryPosts = readingOrder.filter(post => post.category === id);
      return `<section class="post-group" data-category="${id}" aria-labelledby="category-${id}">
      <div class="post-group-heading"><h2 id="category-${id}">${escape(title)}</h2><span>${categoryPosts.length} 篇</span></div>
      <div class="post-list">${categoryPosts.map((post) => `<article class="post-entry">
      <h3><a href="${post.route}" aria-label="${escape(plainTitle(post.title))}">${escape(post.title)}</a></h3>
      <footer class="entry-footer">
        <div class="entry-details">
          <div class="post-meta">${metadata(post)}</div>
          <ul class="entry-topics" aria-label="文章主题">${post.tags.map((tag) => `<li>${escape(tag)}</li>`).join('')}</ul>
        </div>
        <svg class="entry-arrow" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12h14m-6-6 6 6-6 6"/></svg>
      </footer>
    </article>`).join('\n')}</div></section>`;
    }).join('\n')}
  </section>`,
});

const years = [...new Set(posts.map((post) => post.date.slice(0, 4)))];
const archives = page({
  title: '归档 · Blog', description: 'Blog 技术笔记归档。', route: 'archives/', pageClass: 'archive-page',
  body: `<header class="page-header"><h1>归档</h1><p>${posts.length} 篇文章</p></header>
    ${years.map((year) => {
      const yearPosts = posts.filter((post) => post.date.startsWith(year));
      const months = [...new Set(yearPosts.map((post) => post.date.slice(0, 7)))];
      return `<section class="archive-year" aria-labelledby="year-${year}">
        <h2 id="year-${year}">${year} <span>${yearPosts.length}</span></h2>
        ${months.map((month) => `<div class="archive-month">
          <h3>${new Intl.DateTimeFormat('zh-CN', { month: 'long', timeZone: 'UTC' }).format(new Date(`${month}-01T00:00:00Z`))}</h3>
          <div class="archive-entries">${yearPosts.filter((post) => post.date.startsWith(month)).map((post) => `<article class="archive-entry">
            <h4><a href="../${post.route}">${escape(post.title)}</a></h4>
            <div class="post-meta">${metadata(post)}</div>
          </article>`).join('\n')}</div>
        </div>`).join('\n')}
      </section>`;
    }).join('\n')}`,
});

await rm(output, { recursive: true, force: true });
await mkdir(path.join(output, 'archives'), { recursive: true });
await mkdir(path.join(output, 'assets'), { recursive: true });
await writeFile(path.join(output, 'index.html'), home);
await writeFile(path.join(output, 'archives/index.html'), archives);
for (const [index, post] of readingOrder.entries()) {
  const previous = readingOrder[index - 1];
  const next = readingOrder[index + 1];
  const toc = renderPaperToc(post);
  const article = page({
    title: `${post.title} · Blog`, description: post.description, route: post.route, type: 'article', pageClass: 'post-page', hasToc: Boolean(toc),
    body: `<article class="post-single" data-post="${escape(post.slug)}">
      <header class="post-header">
        <h1 id="${escape(post.titleId)}">${escape(post.title)}</h1>
        <div class="post-meta">${metadata(post)}<span>taking-lying-flat</span></div>
      </header>
      <div class="${post.format === 'lake' ? 'lake-document' : 'prose'}">${post.content}</div>
      <footer class="post-footer">
        <div class="post-topics" aria-label="文章主题">${post.tags.map((tag) => `<span>${escape(tag)}</span>`).join('')}</div>
        <div class="post-actions"><a href="#top">返回顶部 ↑</a></div>
        <nav class="post-pagination" aria-label="文章翻页">
          ${previous ? `<a href="../${previous.slug}/"><span>← 上一篇</span>${escape(previous.title)}</a>` : ''}
          ${next ? `<a class="post-next" href="../${next.slug}/"><span>下一篇 →</span>${escape(next.title)}</a>` : ''}
        </nav>
        <a class="back-link" href="../../">← 全部文章</a>
      </footer>
    </article>${toc}`,
  });
  await mkdir(path.join(output, post.route), { recursive: true });
  await writeFile(path.join(output, post.route, 'index.html'), article);
  if (post.format === 'lake') {
    for (const directory of post.directories) {
      await cp(path.join(directory, 'assets'), path.join(output, post.route, 'assets'), { recursive: true });
    }
    for (const asset of post.generatedAssets) await writeFile(path.join(output, post.route, asset.file), asset.content);
  }
}

await writeFile(path.join(output, 'assets/math.css'), adaptor.cssText(svg.styleSheet(document)));
for (const [name, file] of assets) await copyFile(file, path.join(output, 'assets', name));
const fonts = path.join(root, 'node_modules/@fontsource-variable/jetbrains-mono');
await cp(path.join(fonts, 'files'), path.join(output, 'assets/fonts/files'), { recursive: true });
await copyFile(path.join(fonts, 'index.css'), path.join(output, 'assets/fonts/index.css'));
await copyFile(path.join(fonts, 'LICENSE'), path.join(output, 'assets/fonts/LICENSE'));
const readingFonts = path.join(root, 'node_modules/@fontsource-variable/noto-sans-sc');
await cp(path.join(readingFonts, 'files'), path.join(output, 'assets/fonts/files'), { recursive: true });
await copyFile(path.join(readingFonts, 'wght.css'), path.join(output, 'assets/fonts/noto-sans-sc.css'));
await mkdir(path.join(output, 'assets/licenses'), { recursive: true });
await copyFile(path.join(readingFonts, 'LICENSE'), path.join(output, 'assets/licenses/NotoSansSC.txt'));
const chineseSerifFonts = path.join(root, 'node_modules/@fontsource-variable/noto-serif-sc');
await cp(path.join(chineseSerifFonts, 'files'), path.join(output, 'assets/fonts/files'), { recursive: true });
await copyFile(path.join(chineseSerifFonts, 'wght.css'), path.join(output, 'assets/fonts/noto-serif-sc.css'));
await copyFile(path.join(chineseSerifFonts, 'LICENSE'), path.join(output, 'assets/licenses/NotoSerifSC.txt'));
const englishFonts = path.join(root, 'node_modules/@fontsource/source-serif-4');
for (const weight of [400, 700]) {
  for (const style of ['normal', 'italic']) {
    const file = `source-serif-4-latin-${weight}-${style}.woff2`;
    await copyFile(path.join(englishFonts, 'files', file), path.join(output, 'assets/fonts/files', file));
  }
}
await copyFile(path.join(englishFonts, 'LICENSE'), path.join(output, 'assets/licenses/SourceSerif4.txt'));
await copyFile(path.join(primer, 'LICENSE'), path.join(output, 'assets/licenses/Primer.txt'));
await copyFile(path.join(root, 'node_modules/@mathjax/src/LICENSE'), path.join(output, 'assets/licenses/MathJax.txt'));
const mathFont = JSON.parse(await readFile(path.join(root, 'node_modules/@mathjax/mathjax-newcm-font/package.json'), 'utf8'));
await writeFile(path.join(output, 'assets/licenses/NewCM.txt'),
  `${mathFont.name} ${mathFont.version}\n${mathFont.repository.url}\nLicense: ${mathFont.license}\n\n` +
  await readFile(path.join(root, 'node_modules/@mathjax/src/LICENSE'), 'utf8'));
await copyFile(path.join(root, 'node_modules/highlight.js/LICENSE'), path.join(output, 'assets/licenses/highlight.js.txt'));
await writeFile(path.join(output, '.nojekyll'), '');
await writeFile(path.join(output, 'sitemap.xml'), `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${['', 'archives/', ...posts.map((post) => post.route)].map((route) => `<url><loc>${siteUrl}${route}</loc></url>`).join('')}</urlset>\n`);
const rope = posts.find((post) => post.slug === 'rope');
await mkdir(path.join(output, 'rope'), { recursive: true });
await copyFile(path.join(root, rope.file), path.join(output, rope.route, 'rope.md'));
await copyFile(path.join(root, rope.file), path.join(output, 'rope/rope.md'));
await writeFile(path.join(output, 'rope/index.html'), `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="0;url=../${rope.route}"><link rel="canonical" href="${siteUrl}${rope.route}"><title>${escape(rope.title)} · Blog</title></head>
<body><a href="../${rope.route}">阅读 ${escape(rope.title)}</a><script>location.replace('../${rope.route}' + location.search + location.hash);</script></body></html>\n`);
console.log(`Built ${posts.length} articles, home, archives, and legacy redirect: ${displayCount} display formulas, ${inlineCount} inline formulas.`);

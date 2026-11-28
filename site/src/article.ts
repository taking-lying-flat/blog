import path from 'node:path';
import { escape, type ArticleRoute } from './html.ts';
import { readLake, renderLake } from './lake.ts';
import { adaptor, type MathRenderer } from './math.ts';
import type { ArticleEntry } from './catalog.ts';
import TurndownService from 'turndown';
import { tables } from 'turndown-plugin-gfm';
import type { LiteElement, LiteNode } from '@mathjax/src/js/adaptors/lite/Element.js';
import { LiteText } from '@mathjax/src/js/adaptors/lite/Text.js';

interface TocSection { heading: string; links: string[] }

// Prose transforms do not edit MathJax internals. Keep those large fragments
// intact while parsing only the surrounding document structure.
function parseProse(content: string) {
  const formulas: string[] = [];
  const compact = content.replace(/<mjx-container\b[\s\S]*?<\/mjx-container>/g, formula => {
    formulas.push(formula);
    return `<svg data-build-math="${formulas.length - 1}"></svg>`;
  });
  const body = adaptor.body(adaptor.parse(compact, 'text/html'));
  return {
    body,
    html: () => adaptor.innerHTML(body).replace(/<svg data-build-math="(\d+)"><\/svg>/g,
      (placeholder, index) => formulas[Number(index)] ?? placeholder),
  };
}

const slugify = (label: string) => label.toLowerCase().replace(/[^\p{L}\p{N}\p{M}_\-\s]/gu, '').replace(/\s/g, '-');
// Normalize decorative title letters so the chosen web font renders every title.
const plainTitle = (title: string) => title.replace(/[\u{1D400}-\u{1D7FF}\u210E]/gu,
  (letter) => letter.normalize('NFKC'));
const mathClass = /(?:^|\s)(?:lake-math|math-inline|math-display|equation)(?:\s|$)/;
const inlineTags = new Set(['span', 'a', 'strong', 'b', 'em', 'i', 'u', 's', 'small', 'mark']);

// English terms remain prose even when an export splits off k/p as a math card.
function normalizeEnglishTerms(body: LiteElement) {
  const excluded = new Set(['#comment', 'pre', 'svg', 'math', 'script', 'style']);
  let previousText: LiteText | null = null;
  const repair = (node: LiteNode, inHeading: boolean): void => {
    const kind = adaptor.kind(node);
    if (node instanceof LiteText) {
      previousText = kind === '#comment' ? null : node;
      if (inHeading && previousText) node.value = plainTitle(node.value);
      return;
    }
    if (excluded.has(kind) || kind === 'code') { previousText = null; return; }
    if (mathClass.test(adaptor.getAttribute(node, 'class') ?? '')) {
      if (previousText && /(?<![A-Za-z0-9_])(?:top[-‐‑–]|pass@)\s*$/i.test(previousText.value)) {
        const image = adaptor.getElement('img', node);
        const tex = (adaptor.getAttribute(node, 'aria-label') ??
          (image ? adaptor.getAttribute(image, 'alt') : '') ?? '').trim();
        if (/^[kp]$/i.test(tex)) {
          previousText.value = previousText.value.trimEnd() + tex + ' ';
          adaptor.remove(node);
        }
      }
      previousText = null;
      return;
    }
    if (!inlineTags.has(kind)) previousText = null;
    const heading = inHeading || /^h[1-6]$/.test(kind);
    for (const child of adaptor.childNodes(node)) repair(child, heading);
    if (!inlineTags.has(kind)) previousText = null;
  };
  repair(body, false);
  const wrap = (node: LiteNode): void => {
    const kind = adaptor.kind(node);
    if (node instanceof LiteText) {
      if (kind === '#comment') return;
      const original = adaptor.value(node);
      const text = original
        .replace(/[ \t]*词元[ \t]*/g, ' token ')
        .replace(/[ \t]+([，。；：！？])/gu, '$1');
      const terms = /(?<![A-Za-z0-9_])(?:top[-‐‑–](?:[kp]|\d+)|n[-‐‑–]grams?|pass@(?:k|\d+))(?![A-Za-z0-9_])/gi;
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
      } else if (text !== original) {
        adaptor.replace(adaptor.text(text), node);
      }
      return;
    }
    if (excluded.has(kind) || mathClass.test(adaptor.getAttribute(node, 'class') ?? '') ||
        /(?:^|\s)english-term(?:\s|$)/.test(adaptor.getAttribute(node, 'class') ?? '')) return;
    for (const child of adaptor.childNodes(node)) wrap(child);
  };
  wrap(body);
}

function markLakeMathSpacing(body: LiteElement) {
  const isMath = (node: LiteElement) => /(?:^|\s)lake-math(?:\s|$)/.test(adaptor.getAttribute(node, 'class') ?? '');
  let afterMath = false;
  const spaces: LiteText[] = [];
  const reset = (math = false) => { afterMath = math; spaces.length = 0; };
  const visit = (node: LiteNode): void => {
    const kind = adaptor.kind(node);
    if (node instanceof LiteText) {
      if (kind === '#comment') return;
      if (adaptor.value(node).trim()) reset();
      else if (afterMath) spaces.push(node);
      return;
    }
    if (isMath(node)) {
      const numbered = /(?:^|\s)lake-math-numbered(?:\s|$)/.test(adaptor.getAttribute(node, 'class') ?? '');
      if (afterMath && !numbered) {
        adaptor.addClass(node, 'lake-math-adjacent');
        for (const space of spaces) adaptor.remove(space);
      }
      reset(!numbered);
      return;
    }
    const boundary = !inlineTags.has(kind);
    if (boundary) reset();
    if (kind === 'svg' || kind === 'pre') return;
    const children = adaptor.childNodes(node);
    for (const child of children) visit(child);
    if (kind === 'p') {
      // CSS sibling and :only-child selectors ignore intervening prose text.
      const visible = children.filter(child => !(child instanceof LiteText) ||
        (adaptor.kind(child) !== '#comment' && child.value.trim()));
      const [only] = visible;
      if (visible.length === 1 && only && !(only instanceof LiteText) && isMath(only)) {
        adaptor.addClass(only, 'lake-math-standalone');
      }
    }
    if (boundary) reset();
  };
  visit(body);
}
// Keep MathJax's internal equation anchors unique when several independently
// exported papers are assembled into one reading page. Visible paper numbers
// still restart at each h1; only the DOM anchors receive a paper prefix.
function namespaceMathJaxEquationAnchors(content: string, slug: string) {
  if (!content.includes('mjx-eqn:')) return content;
  let paper: string | undefined;
  return content.replace(
    /<h1\b[^>]*>|\s(id|href|xlink:href)="(#?mjx-eqn:[^"]+)"/g,
    (opening, attribute, value) => {
      if (opening.startsWith('<h1')) {
        paper = opening.match(/\bid="([^"]+)"/)?.[1];
        return opening;
      }
      if (!paper) throw new Error(`Equation anchor before paper heading: ${slug}`);
      const prefix = value.startsWith('#') ? '#' : '';
      return ` ${attribute}="${prefix}${paper}-${value.slice(prefix.length)}"`;
    },
  );
}

export type PostInfo = ArticleEntry & { title: string; titleId: string; route: ArticleRoute };
export type PostSummary = PostInfo & { readingMinutes: number };

export async function openArticle(root: string, { dateIndex, ...entry }: ArticleEntry & { dateIndex: number }) {
  const lake = await readLake(path.join(root, entry.file));
  const info: PostInfo = {
    ...entry, route: `posts/${entry.slug}/`, titleId: slugify(lake.title),
    title: plainTitle(lake.title).replace(/\s*-\s*([IVXLCDM]+)$/u, ' $1'),
  };
  return { info, lake, dateIndex };
}

export async function renderArticle({ info, lake }: Awaited<ReturnType<typeof openArticle>>, renderMath: MathRenderer) {
  const { content, directory, assets } = await renderLake(lake, renderMath);
  const prose = parseProse(namespaceMathJaxEquationAnchors(content, info.slug));
  normalizeEnglishTerms(prose.body);
  markLakeMathSpacing(prose.body);
  const readingMinutes = lake.readingMinutes ?? estimateMinutes(prose.body);
  const index = preparePublication(info, prose.body);
  return {
    info: { ...info, readingMinutes }, directory, assets, content: prose.html(), ...index,
  };
}

function estimateMinutes(body: LiteElement) {
  const text = adaptor.textContent(body);
  return Math.max(1, Math.ceil(
    (text.match(/\p{Script=Han}/gu)?.length ?? 0) / 300 +
    (text.match(/[A-Za-z]+/g)?.length ?? 0) / 200
  ));
}

export type Post = Awaited<ReturnType<typeof renderArticle>>;

export function markdownDownload(post: Post) {
  const writer = new TurndownService({ headingStyle: 'atx', codeBlockStyle: 'fenced' });
  writer.use(tables);
  writer.addRule('formula', {
    filter: node => node.classList.contains('lake-math') || node.classList.contains('equation'),
    replacement(_content, node) {
      const tex = (node.getAttribute('aria-label') || node.querySelector('img')?.getAttribute('alt') || '').trim();
      return node.classList.contains('equation') ? `\n\n\`\`\`math\n${tex}\n\`\`\`\n\n` : `$\`${tex}\`$`;
    },
  });
  return `# ${post.info.title}\n\n${writer.turndown(post.content)}\n`;
}

function renderPaperToc(slug: string, sections: TocSection[]) {
  if (!sections.length) throw new Error(`Missing paper headings: ${slug}`);
  return `<aside class="post-toc" aria-label="文章目录">
    <details class="toc-panel" open>
      <summary><span>目录</span><span class="toc-hide">隐藏</span><span class="toc-show">显示</span></summary>
      <nav aria-label="论文及章节">
        <ol class="toc-list">${sections.map(({ heading, links }) => `<li>${heading}${links.length
          ? `<ol>${links.join('')}</ol>` : ''}</li>`).join('')}</ol>
      </nav>
    </details>
  </aside>`;
}


// Lake preserves whitespace. Follow list and section boundaries so source
// indentation cannot create an extra line between a frame and adjacent prose.
const blockContainers = new Set(['section', 'ul', 'ol', 'li', 'div', 'blockquote']);
function normalizeCodeGaps(frame: LiteElement, body: LiteElement) {
  for (const before of [true, false]) {
    const next = (node: LiteNode) => before ? adaptor.previous(node) : adaptor.next(node);
    const skipIndent = (node: LiteNode | null) => {
      while (node instanceof LiteText && node.kind === '#text' && /^[\t\r\n ]+$/.test(node.value)) {
        const sibling = next(node);
        adaptor.remove(node);
        node = sibling;
      }
      return node;
    };
    let cursor = frame;
    let neighbor = skipIndent(next(cursor));
    while (!neighbor && adaptor.parent(cursor) !== body && blockContainers.has(adaptor.kind(adaptor.parent(cursor)))) {
      cursor = adaptor.parent(cursor);
      neighbor = skipIndent(next(cursor));
    }
    while (neighbor && !(neighbor instanceof LiteText) && !adaptor.hasClass(neighbor, 'code-block')) {
      adaptor.addClass(neighbor, before ? 'before-code' : 'after-code');
      if (!blockContainers.has(adaptor.kind(neighbor))) break;
      neighbor = skipIndent(before ? adaptor.lastChild(neighbor) : adaptor.firstChild(neighbor));
    }
  }
}

// Final pass over the prose before serialization.
function preparePublication(post: PostInfo, body: LiteElement) {
  const { slug } = post;
  const sections: TocSection[] = [];
  const headingIds = new Set<string>();
  function visit(node: LiteNode, inToc: boolean) {
    const kind = adaptor.kind(node);
    if (node instanceof LiteText) return;
    if (['svg', 'style', 'script', 'button'].includes(kind)) return;
    if (adaptor.hasClass(node, 'code-block')) normalizeCodeGaps(node, body);
    // Keep native anchors and discard Lake editor metadata from the reading page.
    adaptor.removeAttribute(node, 'data-lake-id');
    adaptor.removeAttribute(node, 'data-lake-style');
    adaptor.removeAttribute(node, 'fid');
    if (kind === 'ul' || kind === 'ol') adaptor.removeAttribute(node, 'list');
    if (inToc && (kind === 'h1' || kind === 'h2')) {
      const id = adaptor.getAttribute(node, 'id');
      const title = plainTitle(adaptor.textContent(node)).replace(/\s+/g, ' ').trim();
      if (!id || !title || headingIds.has(id)) throw new Error(`Missing or duplicate paper heading: ${slug}`);
      headingIds.add(id);
      const link = `<a class="toc-link" href="#${escape(encodeURIComponent(id))}">${escape(title)}</a>`;
      if (kind === 'h1') sections.push({ heading: link, links: [] });
      else {
        const section = sections.at(-1);
        if (!section) throw new Error(`Paper section precedes its title: ${slug}`);
        section.links.push(`<li>${link}</li>`);
      }
      inToc = false;
    }
    if (kind === 'img' || adaptor.hasClass(node, 'equation') || adaptor.hasClass(node, 'math-inline')) return;
    for (const child of adaptor.childNodes(node)) visit(child, inToc && kind !== 'pre' && kind !== 'code');
  }
  visit(body, true);
  return {
    toc: renderPaperToc(slug, sections),
  };
}

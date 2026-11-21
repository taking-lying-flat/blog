import { createHighlighterCoreSync, getTokenStyleObject, stringifyTokenStyle, type ThemedToken } from '@shikijs/core';
import { createJavaScriptRegexEngine } from '@shikijs/engine-javascript';
import cpp from '@shikijs/langs/cpp';
import diff from '@shikijs/langs/diff';
import json from '@shikijs/langs/json';
import jsonl from '@shikijs/langs/jsonl';
import python from '@shikijs/langs/python';
import shell from '@shikijs/langs/shellscript';
import githubLight from '@shikijs/themes/github-light-default';
import { escape } from './html.ts';

export interface CodeCard {
  id: string;
  code: string;
  language?: string;
  caption?: string;
  codeDetails?: {
    caption?: string;
    kind?: string;
    sourceLabel?: string;
    url?: string;
    additionalSources?: { url: string; label: string }[];
  };
}

// Give receivers in Python member access the theme's module scope.
const pythonGrammar = python.map(grammar => grammar.name === 'python' ? {
  ...grammar,
  injections: { ...grammar.injections, 'L:source.python -comment -string': {
    patterns: [{ name: 'support.type.object.module.python',
      match: String.raw`\b[[:alpha:]_]\w*(?=\s*\.\s*[[:alpha:]_])|(?<=\bfrom\s|\bimport\s)[[:alpha:]_]\w*` }],
  } },
} : grammar);
const highlighter = createHighlighterCoreSync({
  themes: [{
    ...githubLight,
    // TextMate gives Python call sites a different scope from declarations.
    tokenColors: githubLight.tokenColors!.map(rule => rule.scope === 'entity.name.function'
      ? { ...rule, scope: ['entity.name.function', 'meta.function-call.generic.python'] } : rule),
  }],
  langs: [...pythonGrammar, ...cpp, ...shell, ...diff, ...json, ...jsonl],
  langAlias: { ipython: 'python', gyp: 'python', cc: 'cpp', cxx: 'cpp', hpp: 'cpp', patch: 'diff' },
  engine: createJavaScriptRegexEngine(),
});
const languages = new Set(highlighter.getLoadedLanguages());
const theme = highlighter.getTheme('github-light-default');
const styles = new Map<string, string>();

function renderToken(token: ThemedToken) {
  const text = escape(token.content);
  if (!text.trim() || (!token.fontStyle && !token.bgColor && token.color?.toLowerCase() === theme.fg.toLowerCase())) return text;
  const style = stringifyTokenStyle(getTokenStyleObject(token));
  let name = styles.get(style);
  if (!name) {
    name = `tok-${styles.size}`;
    styles.set(style, name);
  }
  return `<span class="${name}">${text}</span>`;
}

export function codeStylesheet() {
  const tokens = Array.from(styles, ([style, name]) => `.${name}{${style}}`).join('\n');
  return `.code-block{color:${theme.fg}}\n@media screen {\n${tokens}\n}\n`;
}

function highlightCode(source: string, language: string) {
  const lang = language.toLowerCase();
  if (!languages.has(lang)) return escape(source);
  return highlighter.codeToTokensBase(source, { lang, theme: theme.name })
    .map(line => line.map(renderToken).join('')).join('\n');
}

export function renderLakeCode(value: CodeCard) {
  const { id, code: content, language, codeDetails: details } = value;
  if (!language) {
    return `<div class="code-block" id="${escape(id)}"><pre><code>${escape(content)}</code></pre></div>`;
  }
  const [declaredCaption = '', declaredSource = ''] = (value.caption ?? '').split('|').map(part => part.trim());
  const isDiagram = language === 'text' || details?.kind === 'diagram';
  const caption = details?.caption ?? declaredCaption;
  const source = isDiagram ? '' : (details?.sourceLabel ?? declaredSource);
  const sourceUrl = isDiagram ? undefined : details?.url;
  const displayedCode = isDiagram ? escape(content) : highlightCode(content, language);
  const primarySource = sourceUrl
    ? `<a href="${escape(sourceUrl)}">${escape(source === 'GitHub' ? source : `${source} · GitHub`)}</a>` : escape(source);
  const sourceLabel = [primarySource, ...(details?.additionalSources ?? []).map((item) =>
    `<a href="${escape(item.url)}">${escape(item.label)}</a>`)].join(' · ');
  const captionText = `${caption ? `<span>${escape(caption)}</span>` : ''}${source ? `<span class="code-source">${sourceLabel}</span>` : ''}`;
  return `<figure class="code-block" id="${escape(id)}">
    ${captionText ? `<figcaption class="code-caption">${captionText}</figcaption>` : ''}
    <pre tabindex="0"><code class="language-${escape(language)}">${displayedCode}</code></pre>
  </figure>`;
}

import { createHash } from 'node:crypto';
import { writeFile } from 'node:fs/promises';
import path from 'node:path';
import { escape } from './html.ts';
import { mathjax } from '@mathjax/src/js/mathjax.js';
import { TeX } from '@mathjax/src/js/input/tex.js';
import { SVG } from '@mathjax/src/js/output/svg.js';
import { liteAdaptor } from '@mathjax/src/js/adaptors/liteAdaptor.js';
import { RegisterHTMLHandler } from '@mathjax/src/js/handlers/html.js';
import '@mathjax/src/js/util/asyncLoad/esm.js';
import '@mathjax/src/js/input/tex/ams/AmsConfiguration.js';
import '@mathjax/src/js/input/tex/boldsymbol/BoldsymbolConfiguration.js';
import '@mathjax/src/js/input/tex/newcommand/NewcommandConfiguration.js';
import '@mathjax/src/js/input/tex/color/ColorConfiguration.js';
import '@mathjax/src/js/input/tex/cancel/CancelConfiguration.js';

import type { LiteDocument } from '@mathjax/src/js/adaptors/lite/Document.js';
import type { LiteElement } from '@mathjax/src/js/adaptors/lite/Element.js';
import type { LiteText } from '@mathjax/src/js/adaptors/lite/Text.js';

export const adaptor = liteAdaptor({ fontSize: 18 });
RegisterHTMLHandler(adaptor);

type Formula = (cardStyle: string) => string;
export type MathRenderer = (tex: string, style: string) => Promise<string>;

export function createMathRenderer(directory: string) {
  const svg = new SVG<LiteElement, LiteText, LiteDocument>({ fontCache: 'local', localID: 'glyph', displayOverflow: 'overflow', linebreaks: { inline: false } });
  const input = new TeX({
    packages: ['base', 'ams', 'boldsymbol', 'newcommand', 'color', 'cancel'],
    tagSide: 'right',
    formatError(_jax: unknown, error: Error) { throw error; },
  });
  // alt and aria-label carry the full TeX. Drop parser annotations before SVG layout.
  input.postFilters.add(() => {
    input.parseOptions.root.walkTree(node => node.attributes.unset('data-latex'));
  });
  const document = mathjax.document('', { InputJax: input, OutputJax: svg });

  const cache = new Map<string, Formula>();
  const writtenFiles = new Set<string>();
  let inlineId = 0;
  async function convert(tex: string): Promise<Formula> {
    const node = await document.convertPromise(tex, { display: true, em: 16, ex: 8, containerWidth: 1024 });
    const image = adaptor.getElement('svg', node);
    if (!image) throw new Error(`Missing SVG for Lake formula: ${tex}`);
    const numbered = /\\tag\b/.test(tex);
    if (!numbered && adaptor.getAttribute(image, 'width').endsWith('%')) adaptor.setAttribute(image, 'width', '100%');
    const markup = numbered ? adaptor.outerHTML(node) : adaptor.serializeXML(image);
    if (/data-mjx-error|data-mml-node="merror"|\b(?:NaN|Infinity)\b/.test(markup)) {
      throw new Error(`Invalid rendered Lake formula: ${tex}`);
    }
    const label = escape(tex);
    if (numbered) {
      const template = `<span class="lake-math lake-math-numbered" role="math" aria-label="${label}">${markup}</span>`;
      return () => template.replace(/((?:id|href)="#?)MJX-glyph-/g, `$1MJX-lake-${++inlineId}-`);
    }
    const file = createHash('sha256').update(markup).digest('hex').slice(0, 32) + '.svg';
    if (!writtenFiles.has(file)) {
      await writeFile(path.join(directory, file), markup);
      writtenFiles.add(file);
    }
    const width = adaptor.getAttribute(image, 'width');
    const height = adaptor.getAttribute(image, 'height');
    const style = adaptor.getAttribute(image, 'style') ?? '';
    const wide = parseFloat(width) > 20;
    const baseline = style.match(/vertical-align:\s*([^;]+)/)?.[1] ?? '0';
    const sizing = wide ? `--lake-math-width:${width};vertical-align:${baseline};` : '';
    const img = `<img src="../../assets/math/${file}" alt="${label}" style="width:${escape(width)};height:${escape(height)};${escape(style)}" decoding="async">`;
    return cardStyle => {
      const spanStyle = sizing + cardStyle;
      return `<span class="lake-math${wide ? ' lake-math-wide' : ''}"${spanStyle ? ` style="${escape(spanStyle)}"` : ''}>${img}</span>`;
    };
  }
  const render: MathRenderer = async (tex, style) => {
    let formula = cache.get(tex);
    if (!formula) {
      formula = await convert(tex);
      cache.set(tex, formula);
    }
    return formula(style);
  };
  return { render, stylesheet: () => adaptor.cssText(svg.styleSheet(document)) };
}

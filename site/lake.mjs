import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';

const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

// Keep the archived export intact. Re-render equations only for display fixes
// or explicit, checked corrections to their LaTeX.
export async function readLake(file, escape, renderMath) {
  const directory = path.dirname(file);
  const manifest = JSON.parse(await readFile(path.join(directory, 'manifest.json'), 'utf8'));
  const mathRemovals = new Set(manifest.mathRemovals ?? []);
  for (const id of mathRemovals) {
    if (!manifest.cards.some(card => card.kind === 'math' && card.value.id === id)) {
      throw new Error(`Missing Lake formula selected for removal: ${id}`);
    }
  }
  const source = await readFile(file);
  if (sha256(source) !== manifest.source.sha256) throw new Error(`Lake source changed: ${file}`);
  const overrides = JSON.parse(await readFile(path.join(directory, 'math-overrides.json'), 'utf8')
    .catch((error) => { if (error.code === 'ENOENT') return '{}'; throw error; }));
  for (const [id, override] of Object.entries(overrides)) {
    const card = manifest.cards.find((card) => card.kind === 'math' && card.value.id === id);
    if (!card || card.value.code !== override.from || typeof override.to !== 'string') {
      throw new Error(`Invalid Lake math correction: ${id}`);
    }
  }
  const assets = new Map();
  const invalidMath = new Set();
  for (const asset of manifest.assets) {
    const bytes = await readFile(path.join(directory, asset.file));
    if (sha256(bytes) !== asset.sha256) throw new Error(`Lake asset changed: ${asset.file}`);
    assets.set(asset.file, asset);
    if (asset.file.endsWith('.svg') && /\b(?:NaN|Infinity)\b/.test(bytes.toString())) invalidMath.add(asset.file);
  }
  const imageOverrides = JSON.parse(await readFile(path.join(directory, 'image-overrides.json'), 'utf8')
    .catch((error) => { if (error.code === 'ENOENT') return '{}'; throw error; }));
  for (const [id, correction] of Object.entries(imageOverrides)) {
    const card = manifest.cards.find((card) => card.kind === 'image' && card.value.id === id);
    if (!card || correction.from !== card.asset ||
        !/^https:\/\//.test(correction.source) || typeof correction.label !== 'string' ||
        (correction.asset && (!assets.has(correction.asset) ||
          !Number.isFinite(correction.width) || correction.width <= 0 ||
          !Number.isFinite(correction.height) || correction.height <= 0 ||
          (card.value.crop ?? [0, 0, 1, 1]).some((value, index) => value !== [0, 0, 1, 1][index])))) {
      throw new Error(`Invalid Lake image correction: ${id}`);
    }
  }
  // Legacy SVGs use ornate script glyphs for \mathcal. Use the same readable
  // calligraphic font as the repaired equations, preserving the math commands.
  const renderedCards = new Map();
  const generated = new Map();
  for (const card of manifest.cards) {
    if (card.kind !== 'math') continue;
    const code = overrides[card.value.id]?.to ?? card.value.code;
    // Numbered equations need responsive MathJax layout, not a fixed-width image.
    const original = assets.get(card.asset) ?? { width: '100%' };
    // An explicit override can repair a stale SVG without changing its LaTeX.
    if (card.asset && !invalidMath.has(card.asset) && !Object.hasOwn(overrides, card.value.id) &&
        !/\\(?:mathcal|mathscr|tag)\b/.test(code)) continue;
    const key = JSON.stringify([code, original.width]);
    if (!generated.has(key)) {
      const rendered = await renderMath(code, original);
      generated.set(key, { ...rendered, file: `assets/math/${sha256(rendered.content).slice(0, 32)}.rendered.svg` });
    }
    renderedCards.set(card.value.id, generated.get(key));
  }

  const header = /^<!doctype lake><title>[\s\S]*?<\/title>(?:<meta\b[^>]*>)+/i;
  if (!header.test(source.toString())) throw new Error(`Unexpected Lake header: ${file}`);
  let content = source.toString().replace(header, '');
  // Apply reviewed wording corrections without modifying the archived export.
  const textOverrides = JSON.parse(await readFile(path.join(directory, 'text-overrides.json'), 'utf8')
    .catch((error) => { if (error.code === 'ENOENT') return '{}'; throw error; }));
  for (const [id, override] of Object.entries(textOverrides)) {
    const parts = Array.isArray(override.to) ? override.to : [override.to];
    if (!/^[\w-]+$/.test(id) || typeof override.from !== 'string' || !parts.length ||
      parts.some(part => typeof part !== 'string' && (!part || typeof part.math !== 'string'))) {
      throw new Error(`Invalid Lake text correction: ${id}`);
    }
    let replacement = '';
    for (const part of parts) {
      if (typeof part === 'string') { replacement += escape(part); continue; }
      const rendered = await renderMath(part.math, { width: '100%' });
      const asset = { ...rendered, file: `assets/math/${sha256(rendered.content).slice(0, 32)}.rendered.svg` };
      generated.set(`text:${id}:${part.math}`, asset);
      replacement += `<span class="lake-math" data-text-math="${escape(id)}"><img src="${asset.file}" alt="${escape(part.math)}" style="width:${escape(asset.width)};height:${escape(asset.height)};${escape(asset.style)}" decoding="async"></span>`;
    }
    const pattern = new RegExp(`(<([a-z][\\w:-]*)\\b[^>]*\\sid="${id}"[^>]*>)([^<>]*)(<\\/\\2>)`, 'g');
    let matches = 0;
    content = content.replace(pattern, (_original, opening, _tag, text, closing) => {
      if (text !== escape(override.from)) throw new Error(`Lake text correction source changed: ${id}`);
      matches++;
      return `${opening}${replacement}${closing}`;
    });
    if (matches !== 1) throw new Error(`Lake text correction must match once: ${id}`);
    if (override.removeEmptyBlock) {
      const block = override.removeEmptyBlock;
      if (!['li', 'p'].includes(block) || replacement !== '') throw new Error(`Invalid Lake block deletion: ${id}`);
      const emptyFormatting = '(?:\\s|<br\\s*\\/?>|<\\/?(?:span|strong|em|b|i|u)\\b[^>]*>)*';
      const emptyBlock = new RegExp(`<${block}\\b[^>]*>${emptyFormatting}<span\\b[^>]*\\sid="${id}"[^>]*><\\/span>${emptyFormatting}<\\/${block}>`, 'g');
      if ([...content.matchAll(emptyBlock)].length !== 1) throw new Error(`Lake block deletion must match once: ${id}`);
      content = content.replace(emptyBlock, '');
    }
  }
  // Restore list boundaries at explicit, direct-child text anchors.
  for (const { item, before, id } of manifest.listItemSplits ?? []) {
    if (![item, before, id].every(value => typeof value === 'string' && /^[\w-]+$/.test(value)) ||
        content.includes(` id="${id}"`)) {
      throw new Error(`Invalid Lake list split: ${item}`);
    }
    const pattern = new RegExp(`(<li\\b[^>]*\\sid="${item}"[^>]*>)([\\s\\S]*?)<\\/li>`, 'g');
    if ([...content.matchAll(pattern)].length !== 1) {
      throw new Error(`Lake list split must match once: ${item}`);
    }
    content = content.replace(pattern, (_match, opening, body) => {
      const anchors = [...body.matchAll(new RegExp(`<span\\b[^>]*\\sid="${before}"[^>]*>`, 'g'))];
      if (anchors.length !== 1) throw new Error(`Missing Lake list split anchor: ${before}`);
      const offset = anchors[0].index;
      return `${opening}${body.slice(0, offset)}</li><li id="${id}">${body.slice(offset)}</li>`;
    });
  }
  // Join consecutive explanations without losing their formula cards or anchors.
  for (const { into, items } of manifest.listItemMerges ?? []) {
    if (!/^[\w-]+$/.test(into) || !Array.isArray(items) || !items.length ||
        items.some(id => !/^[\w-]+$/.test(id) || id === into)) {
      throw new Error(`Invalid Lake list merge: ${into}`);
    }
    const inline = '(?:(?!<\\/?(?:p|ul|ol|li)\\b)[\\s\\S])*';
    for (const id of items) {
      const pattern = new RegExp(`(<li\\b[^>]*\\sid="${into}"[^>]*>)(${inline})<\\/li>\\s*(<li\\b[^>]*\\sid="${id}"[^>]*>)(${inline})<\\/li>`, 'g');
      if ([...content.matchAll(pattern)].length !== 1) {
        throw new Error(`Lake list merge must match adjacent items once: ${into}, ${id}`);
      }
      content = content.replace(pattern, (_match, opening, body, itemOpening, itemBody) =>
        `${opening}${body}${itemOpening.replace(/^<li\b/, '<span')}${itemBody}</span></li>`);
    }
  }
  // Use semantic list items for indented explanations, preserving inline cards.
  for (const id of manifest.paragraphListItems ?? []) {
    if (typeof id !== 'string' || !/^[\w-]+$/.test(id)) {
      throw new Error(`Invalid Lake paragraph list item: ${id}`);
    }
    const pattern = new RegExp(`<p\\b([^>]*\\sid="${id}"[^>]*)>([\\s\\S]*?)<\\/p>`, 'g');
    if ([...content.matchAll(pattern)].length !== 1) {
      throw new Error(`Lake paragraph list item must match once: ${id}`);
    }
    content = content.replace(pattern, '<ul><li$1>$2</li></ul>');
  }
  // Join an adjacent single-item list to its paragraph, preserving inline cards.
  for (const { paragraph, listItem } of manifest.paragraphMerges ?? []) {
    if (!/^[\w-]+$/.test(paragraph) || !/^[\w-]+$/.test(listItem)) {
      throw new Error(`Invalid Lake paragraph merge: ${paragraph}`);
    }
    const inline = '(?:(?!<\\/?(?:p|ul|ol|li)\\b)[\\s\\S])*';
    const pattern = new RegExp(`(<p\\b[^>]*\\sid="${paragraph}"[^>]*>)(${inline})<\\/p>\\s*<ul\\b[^>]*>\\s*(<li\\b[^>]*\\sid="${listItem}"[^>]*>)(${inline})<\\/li>\\s*<\\/ul>`, 'g');
    if ([...content.matchAll(pattern)].length !== 1) {
      throw new Error(`Lake paragraph merge must match once: ${paragraph}`);
    }
    content = content.replace(pattern, (_match, opening, body, itemOpening, itemBody) =>
      `${opening}${body}${itemOpening.replace(/^<li\b/, '<span')}${itemBody}</span></p>`);
  }
  // Attach a list item's opening objective to its preceding method statement.
  for (const { paragraph, listItem, before } of manifest.listLeadIns ?? []) {
    if (![paragraph, listItem, before].every(value => typeof value === 'string' && /^[\w-]+$/.test(value))) {
      throw new Error(`Invalid Lake list lead-in: ${listItem}`);
    }
    const inline = '(?:(?!<\\/?(?:p|ul|ol|li)\\b)[\\s\\S])*';
    const pattern = new RegExp(`(<p\\b[^>]*\\sid="${paragraph}"[^>]*>)(${inline})<\\/p>(\\s*<ul\\b[^>]*>\\s*<li\\b[^>]*\\sid="${listItem}"[^>]*>)(${inline})(<span\\b[^>]*\\sid="${before}"[^>]*>)`, 'g');
    if ([...content.matchAll(pattern)].length !== 1) {
      throw new Error(`Lake list lead-in must match once: ${listItem}`);
    }
    content = content.replace(pattern, (_match, opening, body, listOpening, lead, anchor) =>
      `${opening}${body}${lead}</p>${listOpening}${anchor}`);
  }
  // Split prose at explicit anchors without changing its inline formulas.
  for (const { paragraph, before, id } of manifest.paragraphSplits ?? []) {
    if (![paragraph, before, id].every(value => typeof value === 'string' && /^[\w-]+$/.test(value)) ||
        content.includes(` id="${id}"`)) {
      throw new Error(`Invalid Lake paragraph split: ${paragraph}`);
    }
    const pattern = new RegExp(`(<p\\b[^>]*\\sid="${paragraph}"[^>]*>)([\\s\\S]*?)<\\/p>`, 'g');
    if ([...content.matchAll(pattern)].length !== 1) {
      throw new Error(`Lake paragraph split must match once: ${paragraph}`);
    }
    content = content.replace(pattern, (_match, opening, body) => {
      const anchors = [...body.matchAll(new RegExp(`<span\\b[^>]*\\sid="${before}"[^>]*>`, 'g'))];
      if (anchors.length !== 1) throw new Error(`Missing Lake paragraph split anchor: ${before}`);
      const offset = anchors[0].index;
      return `${opening}${body.slice(0, offset)}</p><p id="${id}">${body.slice(offset)}</p>`;
    });
  }
  let index = 0;
  let mathCount = 0;
  let imageCount = 0;
  let separatorCount = 0;
  let codeCount = 0;
  content = content.replace(/<card\b[^>]*>[\s\S]*?<\/card>/g, (original) => {
    const card = manifest.cards[index++];
    if (!card || !original.includes(`value="${card.attributes.value}"`)) {
      throw new Error(`Lake card order changed: ${index}`);
    }
    const id = escape(card.value.id);
    if (card.kind === 'hr') {
      separatorCount++;
      if (manifest.separatorRemovals?.includes(card.value.id)) return '';
      return `<hr class="lake-separator" data-card-id="${id}">`;
    }
    if (card.kind === 'codeblock') {
      if (typeof card.value.code !== 'string') throw new Error(`Invalid Lake code block: ${id}`);
      codeCount++;
      return `<div class="code-block code-diagram lake-code-diagram" id="${id}" data-card-id="${id}"><pre><code>${escape(card.value.code)}</code></pre></div>`;
    }
    const asset = renderedCards.get(card.value.id) ?? assets.get(card.asset);
    if (!asset) throw new Error(`Missing Lake asset: ${card.asset}`);
    if (card.kind === 'math') {
      mathCount++;
      if (mathRemovals.has(card.value.id)) return '';
      if (asset.html) {
        const code = overrides[card.value.id]?.to ?? card.value.code;
        return `<span class="lake-math lake-math-numbered" data-card-id="${id}" role="math" aria-label="${escape(code)}">${asset.html}</span>`;
      }
      const wide = parseFloat(asset.width) > 20;
      const baseline = asset.style?.match(/vertical-align:\s*([^;]+)/)?.[1] ?? '0';
      const sizing = wide ? `--lake-math-width:${asset.width};vertical-align:${baseline};` : '';
      const style = ` style="${sizing}${escape(card.attributes.style ?? '')}"`;
      const code = overrides[card.value.id]?.to ?? card.value.code;
      return `<span class="lake-math${wide ? ' lake-math-wide' : ''}" data-card-id="${id}"${style}><img src="${escape(asset.file)}" alt="${escape(code)}" style="width:${escape(asset.width)};height:${escape(asset.height)};${escape(asset.style ?? '')}" decoding="async"></span>`;
    }
    if (card.kind === 'image') {
      imageCount++;
      const correction = imageOverrides[card.value.id];
      const value = correction?.asset ? { ...card.value, ...correction } : card.value;
      const imageFile = correction?.asset ?? card.asset;
      const { crop = [0, 0, 1, 1], originWidth, originHeight, width, height } = value;
      if (crop.some((value, index) => value !== [0, 0, 1, 1][index])) {
        const [left, top, right, bottom] = crop;
        const cropWidth = right - left;
        const cropHeight = bottom - top;
        // Lake stores the uncropped height, but displays the selected region
        // at the authored width. Keep the source image and crop only its view.
        const sizing = `width:${100 / cropWidth}%;height:auto;left:${-100 * left / cropWidth}%;top:${-100 * top / cropHeight}%;`;
        return `<span class="lake-image lake-image-cropped" data-card-id="${id}" style="width:${width}px;aspect-ratio:${originWidth * cropWidth}/${originHeight * cropHeight}"><img src="${escape(card.asset)}" alt="${escape(card.value.title ?? '')}" width="${width}" height="${height}" style="${sizing}" decoding="async"></span>`;
      }
      const image = `<img class="lake-image" data-card-id="${id}" src="${escape(imageFile)}" alt="${escape(correction?.alt ?? card.value.title ?? '')}" width="${width}" height="${height}" style="width:${width}px;aspect-ratio:${width}/${height}" decoding="async">`;
      return image + (correction ? `<span class="lake-image-source"><a href="${escape(correction.source)}">${escape(correction.label)} · GitHub</a></span>` : '');
    }
    throw new Error(`Unsupported Lake card: ${card.kind}`);
  });
  if (index !== manifest.cards.length || mathCount !== manifest.counts.math || imageCount !== manifest.counts.images || separatorCount !== (manifest.counts.separators ?? 0) || codeCount !== (manifest.counts.code ?? 0)) {
    throw new Error(`Lake card count mismatch: ${file}`);
  }
  // Lake's paired light/dark color syntax is not a CSS color. The document uses
  // its original light palette; retain the pair for a future viewer as well.
  content = content.replace(/style="([^"<>]*color:\s*)(rgb\([^)]+\)),\s*rgb\([^)]+\)([^"<>]*)"/g,
    (original, prefix, light, suffix) => `data-lake-style="${original.slice(7, -1)}" style="${prefix}${light}${suffix}"`);
  const figures = JSON.parse(await readFile(path.join(directory, 'figure-inserts.json'), 'utf8')
    .catch((error) => { if (error.code === 'ENOENT') return '[]'; throw error; }));
  for (const figure of figures) {
    const asset = assets.get(figure.asset);
    if (!/^[\w-]+$/.test(figure.id) || !/^[\w-]+$/.test(figure.before) ||
        !asset || !(asset.source === 'user-upload' || /^https:\/\//.test(asset.source))) {
      throw new Error(`Invalid Lake figure: ${figure.id}`);
    }
    let image = `<img src="${escape(asset.file)}" alt="${escape(figure.alt)}" width="${asset.width}" height="${asset.height}" decoding="async">`;
    let figureStyle = '';
    if (figure.crop !== undefined) {
      // Show separate regions of a diagram while keeping the source image intact.
      const { x, y, width, height } = figure.crop;
      if (![asset.width, asset.height, x, y, width, height].every(Number.isFinite) ||
          x < 0 || y < 0 || width <= 0 || height <= 0 ||
          x + width > asset.width || y + height > asset.height) {
        throw new Error(`Invalid Lake figure crop: ${figure.id}`);
      }
      const sizing = `width:${100 * asset.width / width}%;left:${-100 * x / width}%;top:${-100 * y / height}%;`;
      image = `<span class="lake-figure-crop" style="aspect-ratio:${width}/${height}"><img src="${escape(asset.file)}" alt="${escape(figure.alt)}" width="${asset.width}" height="${asset.height}" style="${sizing}" decoding="async"></span>`;
      figureStyle = ` style="width:${width}px"`;
    }
    const html = `<figure class="lake-figure" id="${figure.id}"${figureStyle}><a href="${escape(asset.file)}">${image}</a>${figure.caption ? `<figcaption>${escape(figure.caption)} · <a href="${escape(figure.source)}">官方来源</a></figcaption>` : ''}</figure>`;
    const anchor = new RegExp(`<p\\b[^>]*\\sid="${figure.before}"[^>]*>`, 'g');
    let matches = 0;
    content = content.replace(anchor, (opening) => { matches++; return html + opening; });
    if (matches !== 1) throw new Error(`Missing or repeated Lake figure anchor: ${figure.before}`);
  }
  // Independently exported sections can reuse anchors. Rename only the
  // declared anchors and their local links, preserving the source archive.
  for (const [from, to] of Object.entries(manifest.idOverrides ?? {})) {
    if (!/^[\w-]+$/.test(from) || !/^[\w-]+$/.test(to)) {
      throw new Error(`Invalid Lake anchor correction: ${from}`);
    }
    const anchor = new RegExp(`(\\sid=")${from}("(?=[\\s>]))`, 'g');
    if ([...content.matchAll(anchor)].length !== 1 || content.includes(` id="${to}"`)) {
      throw new Error(`Lake anchor correction must be unique: ${from}`);
    }
    content = content.replace(anchor, `$1${to}$2`)
      .replaceAll(`href="#${from}"`, `href="#${to}"`);
  }
  // Remove explicitly bounded sections while preserving the archived export.
  for (const { from, before } of manifest.sectionRemovals ?? []) {
    const boundaries = [from, before].map((id) => {
      if (typeof id !== 'string' || !/^[\w-]+$/.test(id)) {
        throw new Error(`Invalid Lake section removal: ${id}`);
      }
      const heading = new RegExp(`<h[1-6]\\b[^>]*\\sid="${id}"[^>]*>`, 'g');
      const matches = [...content.matchAll(heading)];
      if (matches.length !== 1) {
        throw new Error(`Lake section removal boundary must match once: ${id}`);
      }
      return matches[0].index;
    });
    if (boundaries[0] >= boundaries[1]) {
      throw new Error(`Invalid Lake section removal order: ${from}, ${before}`);
    }
    content = content.slice(0, boundaries[0]) + content.slice(boundaries[1]);
  }
  // Remove selected blocks, or their tails from an explicit inline anchor.
  for (const { tag, id, trimFrom } of manifest.blockRemovals ?? []) {
    if (!['p', 'li', 'blockquote', 'h2'].includes(tag) || !/^[\w-]+$/.test(id) ||
        (trimFrom !== undefined && !/^[\w-]+$/.test(trimFrom))) {
      throw new Error(`Invalid Lake block removal: ${id}`);
    }
    const block = new RegExp(`<${tag}\\b[^>]*\\sid="${id}"[^>]*>[\\s\\S]*?<\\/${tag}>`, 'g');
    if ([...content.matchAll(block)].length !== 1) {
      throw new Error(`Lake block removal must match once: ${id}`);
    }
    content = content.replace(block, (html) => {
      if (trimFrom === undefined) return '';
      const start = new RegExp(`<span\\b[^>]*\\sid="${trimFrom}"[^>]*>`, 'g');
      const matches = [...html.matchAll(start)];
      if (matches.length !== 1) {
        throw new Error(`Lake tail removal anchor must match once: ${id}/${trimFrom}`);
      }
      return html.slice(0, matches[0].index) + `</${tag}>`;
    });
    if (tag === 'li') {
      content = content.replace(/<(ul|ol)\b[^>]*>\s*<\/\1>/g, '');
    }
  }
  // Restore complete lists to ordinary prose without changing their contents.
  for (const entry of manifest.listItemParagraphs ?? []) {
    const ids = Array.isArray(entry) ? entry : [entry];
    if (!ids.length || ids.some(id => typeof id !== 'string' || !/^[\w-]+$/.test(id))) {
      throw new Error(`Invalid Lake list paragraph: ${entry}`);
    }
    const inline = '(?:(?!<\\/?(?:ul|ol|li)\\b)[\\s\\S])*';
    const items = ids.map(id => `<li\\b[^>]*\\sid="${id}"[^>]*>${inline}<\\/li>\\s*`).join('');
    const pattern = new RegExp(`<ul\\b[^>]*>\\s*${items}<\\/ul>`, 'g');
    if ([...content.matchAll(pattern)].length !== 1) {
      throw new Error(`Lake list paragraphs must match one complete list: ${ids.join(', ')}`);
    }
    content = content.replace(pattern, html => html
      .replace(/^<ul\b[^>]*>\s*|<\/ul>$/g, '')
      .replace(/<li\b/g, '<p').replace(/<\/li>/g, '</p>'));
  }
  // Join adjacent paragraphs without changing inline formatting or formulas.
  for (const { into, next, separator = '' } of manifest.paragraphJoins ?? []) {
    if (![into, next].every(id => typeof id === 'string' && /^[\w-]+$/.test(id)) ||
        into === next || typeof separator !== 'string') {
      throw new Error(`Invalid Lake paragraph join: ${into}`);
    }
    const inline = '(?:(?!<\\/?p\\b)[\\s\\S])*';
    const pattern = new RegExp(`(<p\\b[^>]*\\sid="${into}"[^>]*>)(${inline})<\\/p>\\s*(<p\\b[^>]*\\sid="${next}"[^>]*>)(${inline})<\\/p>`, 'g');
    if ([...content.matchAll(pattern)].length !== 1) {
      throw new Error(`Lake paragraph join must match adjacent paragraphs once: ${into}, ${next}`);
    }
    content = content.replace(pattern, (_match, opening, body, nextOpening, nextBody) =>
      `${opening}${body}${escape(separator)}${nextOpening.replace(/^<p\b/, '<span')}${nextBody}</span></p>`);
  }
  // Move whole ranges after rendering cards so their source order stays valid.
  for (const { from, until, before } of manifest.blockRangeMoves ?? []) {
    const positions = [from, until, before].map(id => {
      if (typeof id !== 'string' || !/^[\w-]+$/.test(id)) {
        throw new Error(`Invalid Lake block move anchor: ${id}`);
      }
      const pattern = new RegExp(`<(?:p|h[1-6]|ul|ol|blockquote)\\b[^>]*\\sid="${id}"[^>]*>`, 'g');
      const matches = [...content.matchAll(pattern)];
      if (matches.length !== 1) throw new Error(`Lake block move anchor must match once: ${id}`);
      return matches[0].index;
    });
    const [start, end, target] = positions;
    if (start >= end || (target >= start && target < end)) {
      throw new Error(`Invalid Lake block move range: ${from}, ${until}, ${before}`);
    }
    const moved = content.slice(start, end);
    content = content.slice(0, start) + content.slice(end);
    const destination = target >= end ? target - moved.length : target;
    content = content.slice(0, destination) + moved + content.slice(destination);
  }
  const text = content.replace(/<[^>]*>/g, '');
  return {
    title: manifest.title,
    content,
    readingMinutes: Math.max(1, Math.ceil(
      (text.match(/\p{Script=Han}/gu)?.length ?? 0) / 300 +
      (text.match(/[A-Za-z]+/g)?.length ?? 0) / 200
    )),
    directory,
    generatedAssets: [...generated.values()],
  };
}

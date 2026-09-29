import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';

const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

// Keep the archived export intact. Re-render equations only for display fixes
// or explicit, checked corrections to their LaTeX.
export async function readLake(file, escape, renderMath) {
  const directory = path.dirname(file);
  const manifest = JSON.parse(await readFile(path.join(directory, 'manifest.json'), 'utf8'));
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
    if (card.asset && !invalidMath.has(card.asset) && code === card.value.code &&
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
  let index = 0;
  let mathCount = 0;
  let imageCount = 0;
  let separatorCount = 0;
  content = content.replace(/<card\b[^>]*>[\s\S]*?<\/card>/g, (original) => {
    const card = manifest.cards[index++];
    if (!card || !original.includes(`value="${card.attributes.value}"`)) {
      throw new Error(`Lake card order changed: ${index}`);
    }
    const id = escape(card.value.id);
    if (card.kind === 'hr') {
      separatorCount++;
      return `<hr class="lake-separator" data-card-id="${id}">`;
    }
    const asset = renderedCards.get(card.value.id) ?? assets.get(card.asset);
    if (!asset) throw new Error(`Missing Lake asset: ${card.asset}`);
    if (card.kind === 'math') {
      mathCount++;
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
  if (index !== manifest.cards.length || mathCount !== manifest.counts.math || imageCount !== manifest.counts.images || separatorCount !== (manifest.counts.separators ?? 0)) {
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
    const html = `<figure class="lake-figure" id="${figure.id}"><a href="${escape(asset.file)}"><img src="${escape(asset.file)}" alt="${escape(figure.alt)}" width="${asset.width}" height="${asset.height}" decoding="async"></a>${figure.caption ? `<figcaption>${escape(figure.caption)} · <a href="${escape(figure.source)}">官方来源</a></figcaption>` : ''}</figure>`;
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

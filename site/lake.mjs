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
  // Legacy SVGs use ornate script glyphs for \mathcal. Use the same readable
  // calligraphic font as the repaired equations, preserving the math commands.
  const renderedCards = new Map();
  const generated = new Map();
  for (const card of manifest.cards) {
    if (card.kind !== 'math') continue;
    const code = overrides[card.value.id]?.to ?? card.value.code;
    if (!invalidMath.has(card.asset) && code === card.value.code && !/\\math(?:cal|scr)\b/.test(code)) continue;
    const key = JSON.stringify([code, assets.get(card.asset).width]);
    if (!generated.has(key)) {
      const rendered = await renderMath(code, assets.get(card.asset));
      generated.set(key, { ...rendered, file: `assets/math/${sha256(rendered.content).slice(0, 32)}.rendered.svg` });
    }
    renderedCards.set(card.value.id, generated.get(key));
  }

  const header = /^<!doctype lake><title>[\s\S]*?<\/title>(?:<meta\b[^>]*>)+/i;
  if (!header.test(source.toString())) throw new Error(`Unexpected Lake header: ${file}`);
  let content = source.toString().replace(header, '');
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
      const wide = parseFloat(asset.width) > 20;
      const baseline = asset.style?.match(/vertical-align:\s*([^;]+)/)?.[1] ?? '0';
      const sizing = wide ? `--lake-math-width:${asset.width};vertical-align:${baseline};` : '';
      const style = ` style="${sizing}${escape(card.attributes.style ?? '')}"`;
      const code = overrides[card.value.id]?.to ?? card.value.code;
      return `<span class="lake-math${wide ? ' lake-math-wide' : ''}" data-card-id="${id}"${style}><img src="${escape(asset.file)}" alt="${escape(code)}" style="width:${escape(asset.width)};height:${escape(asset.height)};${escape(asset.style ?? '')}" decoding="async"></span>`;
    }
    if (card.kind === 'image') {
      imageCount++;
      const { crop = [0, 0, 1, 1], originWidth, originHeight, width, height } = card.value;
      if (crop.some((value, index) => value !== [0, 0, 1, 1][index])) {
        const [left, top, right, bottom] = crop;
        const cropWidth = right - left;
        const cropHeight = bottom - top;
        // Lake stores the uncropped height, but displays the selected region
        // at the authored width. Keep the source image and crop only its view.
        const sizing = `width:${100 / cropWidth}%;height:auto;left:${-100 * left / cropWidth}%;top:${-100 * top / cropHeight}%;`;
        return `<span class="lake-image lake-image-cropped" data-card-id="${id}" style="width:${width}px;aspect-ratio:${originWidth * cropWidth}/${originHeight * cropHeight}"><img src="${escape(card.asset)}" alt="${escape(card.value.title ?? '')}" width="${width}" height="${height}" style="${sizing}" decoding="async"></span>`;
      }
      return `<img class="lake-image" data-card-id="${id}" src="${escape(card.asset)}" alt="${escape(card.value.title ?? '')}" width="${card.value.width}" height="${card.value.height}" style="width:${card.value.width}px;aspect-ratio:${card.value.width}/${card.value.height}" decoding="async">`;
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

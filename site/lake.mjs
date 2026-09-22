import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';

const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

// Lake exports contain the authored HTML and references to already-rendered
// equations. Preserve the HTML and valid equation images without conversion.
export async function readLake(file, escape, renderMath) {
  const directory = path.dirname(file);
  const manifest = JSON.parse(await readFile(path.join(directory, 'manifest.json'), 'utf8'));
  const source = await readFile(file);
  if (sha256(source) !== manifest.source.sha256) throw new Error(`Lake source changed: ${file}`);
  const assets = new Map();
  const invalidMath = new Set();
  for (const asset of manifest.assets) {
    const bytes = await readFile(path.join(directory, asset.file));
    if (sha256(bytes) !== asset.sha256) throw new Error(`Lake asset changed: ${asset.file}`);
    assets.set(asset.file, asset);
    if (asset.file.endsWith('.svg') && /\b(?:NaN|Infinity)\b/.test(bytes.toString())) invalidMath.add(asset.file);
  }
  // Some exported SVGs contain non-finite glyph coordinates. Re-render only
  // those images from their original LaTeX; retain the untouched export.
  const repaired = new Map();
  for (const card of manifest.cards) {
    if (card.kind !== 'math' || !invalidMath.has(card.asset) || repaired.has(card.asset)) continue;
    const rendered = await renderMath(card.value.code, assets.get(card.asset));
    repaired.set(card.asset, { ...rendered, file: card.asset.replace(/\.svg$/, '.repaired.svg') });
  }

  const header = /^<!doctype lake><title>[\s\S]*?<\/title>(?:<meta\b[^>]*>)+/i;
  if (!header.test(source.toString())) throw new Error(`Unexpected Lake header: ${file}`);
  let content = source.toString().replace(header, '');
  let index = 0;
  let mathCount = 0;
  let imageCount = 0;
  content = content.replace(/<card\b[^>]*>[\s\S]*?<\/card>/g, (original) => {
    const card = manifest.cards[index++];
    if (!card || !original.includes(`value="${card.attributes.value}"`)) {
      throw new Error(`Lake card order changed: ${index}`);
    }
    const asset = repaired.get(card.asset) ?? assets.get(card.asset);
    if (!asset) throw new Error(`Missing Lake asset: ${card.asset}`);
    const id = escape(card.value.id);
    if (card.kind === 'math') {
      mathCount++;
      const wide = parseFloat(asset.width) > 20;
      const baseline = asset.style?.match(/vertical-align:\s*([^;]+)/)?.[1] ?? '0';
      const sizing = wide ? `--lake-math-width:${asset.width};vertical-align:${baseline};` : '';
      const style = ` style="${sizing}${escape(card.attributes.style ?? '')}"`;
      return `<span class="lake-math${wide ? ' lake-math-wide' : ''}" data-card-id="${id}"${style}><img src="${escape(asset.file)}" alt="${escape(card.value.code)}" style="width:${escape(asset.width)};height:${escape(asset.height)};${escape(asset.style ?? '')}" decoding="async"></span>`;
    }
    if (card.kind === 'image') {
      imageCount++;
      return `<img class="lake-image" data-card-id="${id}" src="${escape(card.asset)}" alt="${escape(card.value.title ?? '')}" width="${card.value.width}" height="${card.value.height}" style="width:${card.value.width}px;aspect-ratio:${card.value.width}/${card.value.height}" decoding="async">`;
    }
    throw new Error(`Unsupported Lake card: ${card.kind}`);
  });
  if (index !== manifest.cards.length || mathCount !== manifest.counts.math || imageCount !== manifest.counts.images) {
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
    generatedAssets: [...repaired.values()],
  };
}

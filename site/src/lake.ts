import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { escape } from './html.ts';
import { renderLakeCode, type CodeCard } from './code.ts';
import type { MathRenderer } from './math.ts';

interface ImageCard {
  title?: string;
  width: number;
  height: number;
  originWidth: number;
  originHeight: number;
  crop?: [number, number, number, number];
}

const isObject = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null;
const optionalText = (value: unknown) => value === undefined || typeof value === 'string';

function isCodeCard(value: Record<string, unknown>): value is CodeCard & Record<string, unknown> {
  if (typeof value.code !== 'string' || typeof value.id !== 'string' ||
      !optionalText(value.language) || !optionalText(value.caption)) return false;
  const details = value.codeDetails;
  if (details === undefined) return true;
  if (!isObject(details) || !['kind', 'caption', 'sourceLabel', 'url'].every(key => optionalText(details[key]))) return false;
  const sources = details.additionalSources;
  return sources === undefined || (Array.isArray(sources) && sources.every(source =>
    isObject(source) && typeof source.url === 'string' && typeof source.label === 'string'));
}

function isImageCard(value: Record<string, unknown>): value is ImageCard & Record<string, unknown> {
  return optionalText(value.title) &&
    ['width', 'height', 'originWidth', 'originHeight'].every(key => Number.isFinite(value[key])) &&
    (value.crop === undefined || (Array.isArray(value.crop) && value.crop.length === 4 && value.crop.every(Number.isFinite)));
}

const entities: Record<string, string> = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: '\u00a0' };
const decode = (value: string) => value.replace(/&(#x[\da-f]+|#\d+|amp|lt|gt|quot|apos|nbsp);/gi,
  (match, name) => name[0] === '#'
    ? String.fromCodePoint(name[1].toLowerCase() === 'x' ? parseInt(name.slice(2), 16) : Number(name.slice(1)))
    : entities[name.toLowerCase()] ?? match);
function attributesOf(opening: string): Record<string, string> {
  const attributes: Record<string, string> = {};
  for (const [, key, value] of opening.matchAll(/([\w:-]+)="([^"]*)"/g)) {
    if (key && value !== undefined) attributes[key] = decode(value);
  }
  return attributes;
}

function localAsset(file: string | undefined) {
  if (!file?.startsWith('assets/') || file.includes('..') || file.includes('\\')) {
    throw new Error(`Invalid local Lake asset: ${file}`);
  }
  return file;
}

function bindMathPunctuation(source: string) {
  // Keep punctuation attached to inline formulas, including styled prose spans.
  return source.replace(
    /(<card\b[^>]*\bname="math"[^>]*>\s*<\/card>)((?:<(?:span|strong|b|em|i)\b[^>]*>)*)\s*([，。；：！？、）】》」』,.!?;:)\]]+)/g,
    (_, card: string, opening: string, punctuation: string) => {
      const styling = opening.replace(/\sid="[^"]*"/g, '');
      const closing = [...opening.matchAll(/<([\w]+)\b/g)].reverse().map(([, tag]) => `</${tag}>`).join('');
      return `<span class="lake-math-punctuation">${card}${styling}${punctuation}${closing}</span>${opening}`;
    }
  );
}

export async function readLake(file: string) {
  const source = await readFile(file, 'utf8');
  const header = /^<!doctype lake><title>([\s\S]*?)<\/title>(?:<meta\b[^>]*>)+/i;
  const match = source.match(header);
  if (match?.[1] === undefined) throw new Error(`Unexpected Lake header: ${file}`);
  const readingTime = [...match[0].matchAll(/<meta\b[^>]*>/g)]
    .map(([opening]) => attributesOf(opening))
    .find(meta => meta.name === 'reading-minutes')?.content;
  if (readingTime !== undefined && !/^[1-9]\d*$/.test(readingTime)) {
    throw new Error(`Invalid Lake reading time: ${file}`);
  }
  return {
    file, title: decode(match[1]), body: source.slice(match[0].length).trimEnd(),
    readingMinutes: readingTime === undefined ? undefined : Number(readingTime),
  };
}

type LakeSource = Awaited<ReturnType<typeof readLake>>;

export async function renderLake({ file, body: source }: LakeSource, renderMath: MathRenderer) {
  const assets = new Set<string>();
  // Directly authored figures and image cards both use article-local paths.
  for (const [, ref] of source.matchAll(/\b(?:src|href|data-license)="(assets\/[^"]+)"/g)) {
    if (ref) assets.add(localAsset(decode(ref)));
  }
  const body = bindMathPunctuation(source);
  const fragments: string[] = [];
  const opening = /<card\b[^>]*>/g;
  let last = 0;
  for (let card; (card = opening.exec(body));) {
    fragments.push(body.slice(last, card.index));
    const end = body.indexOf('</card>', opening.lastIndex);
    if (end < 0 || /<card\b/.test(body.slice(opening.lastIndex, end))) {
      throw new Error(`Unclosed Lake card: ${file}`);
    }
    last = opening.lastIndex = end + '</card>'.length;
    const attr = attributesOf(card[0]);
    if (!attr.value?.startsWith('data:')) throw new Error(`Invalid Lake card: ${file}`);
    const value: unknown = JSON.parse(decodeURIComponent(attr.value.slice(5)));
    if (!isObject(value) || typeof value.id !== 'string') throw new Error(`Missing Lake card id: ${file}`);
    const id = value.id;
    switch (attr.name) {
      case 'hr':
        fragments.push('<hr class="lake-separator">');
        break;
      case 'codeblock':
        if (!isCodeCard(value)) throw new Error(`Invalid code block: ${file}#${id}`);
        fragments.push(renderLakeCode(value));
        break;
      case 'math':
        if (typeof value.code !== 'string') throw new Error(`Missing TeX: ${file}#${id}`);
        fragments.push(await renderMath(value.code, attr.style ?? ''));
        break;
      case 'image': {
        if (!isImageCard(value)) throw new Error(`Invalid image card: ${file}#${id}`);
        const image = localAsset(attr['data-asset']);
        assets.add(image);
        fragments.push(renderImage(value, escape(image)));
        break;
      }
      default:
        throw new Error(`Unknown Lake card: ${file}#${id} (${attr.name})`);
    }
  }
  fragments.push(body.slice(last));
  return {
    content: fragments.join(''), assets,
    directory: path.resolve(path.dirname(file), '..'),
  };
}

function renderImage(value: ImageCard, image: string) {
  const { crop = [0, 0, 1, 1], originWidth, originHeight, width, height } = value;
  const [left, top, right, bottom] = crop;
  if (left !== 0 || top !== 0 || right !== 1 || bottom !== 1) {
    const cropWidth = right - left;
    const cropHeight = bottom - top;
    const sizing = `width:${100 / cropWidth}%;height:auto;left:${-100 * left / cropWidth}%;top:${-100 * top / cropHeight}%;`;
    return `<span class="lake-image lake-image-cropped" style="width:${width}px;aspect-ratio:${originWidth * cropWidth}/${originHeight * cropHeight}"><img src="${image}" alt="${escape(value.title ?? '')}" width="${width}" height="${height}" style="${sizing}" decoding="async"></span>`;
  }
  return `<img class="lake-image" src="${image}" alt="${escape(value.title ?? '')}" width="${width}" height="${height}" style="width:${width}px;aspect-ratio:${width}/${height}" decoding="async">`;
}

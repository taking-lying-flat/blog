import { copyFile, mkdir, mkdtemp, rename, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { buildKeepDigging } from '../keep-digging/build.ts';
import catalog from './posts.ts';
import { markdownDownload, openArticle, renderArticle, type Post, type PostSummary } from './src/article.ts';
import { copySiteAssets } from './src/assets.ts';
import { orderCategories, type CategoryGroup } from './src/catalog.ts';
import { codeStylesheet } from './src/code.ts';
import { escape, siteUrl, version } from './src/html.ts';
import { createMathRenderer } from './src/math.ts';
import { renderArchives, renderHome, renderPost } from './src/pages.ts';

const root = path.resolve(import.meta.dirname, '../../..');

async function writePost(output: string, post: Post, html: string) {
  const { info } = post;
  const directory = path.join(output, info.route);
  await mkdir(directory, { recursive: true });
  await writeFile(path.join(directory, 'index.html'), html);
  for (const file of post.assets) {
    const target = path.join(directory, file);
    await mkdir(path.dirname(target), { recursive: true });
    await copyFile(path.join(post.directory, file), target);
  }
  for (const alias of info.aliases ?? []) {
    const target = escape(new URL(info.route, siteUrl).href);
    const script = escape(path.posix.relative(alias, 'assets/reader.js'));
    await mkdir(path.join(output, alias), { recursive: true });
    await writeFile(path.join(output, alias, 'index.html'), `<!doctype html>
<html lang="zh-CN" data-redirect><head><meta charset="utf-8">
<link rel="canonical" href="${target}"><title>${escape(info.title)}</title>
<noscript><meta http-equiv="refresh" content="0;url=${target}"></noscript>
<script type="module" src="${script}?v=${version}"></script></head>
<body><a href="${target}">${escape(info.title)}</a></body></html>`);
  }
  if (info.downloads?.length) {
    const markdown = markdownDownload(post);
    for (const file of info.downloads) {
      await mkdir(path.dirname(path.join(output, file)), { recursive: true });
      await writeFile(path.join(output, file), markdown);
    }
  }
}

async function build(output: string) {
  const mathDir = path.join(output, 'assets/math');
  await mkdir(mathDir, { recursive: true });
  const math = createMathRenderer(mathDir);
  const entries = catalog.toSorted((a, b) => b.date.localeCompare(a.date))
    .map((entry, dateIndex) => ({ ...entry, dateIndex }));
  const archives: PostSummary[] = [];
  const categories: CategoryGroup<PostSummary>[] = [];
  for (const { id, title, posts: group } of orderCategories(entries)) {
    const posts = await Promise.all(group.map(entry => openArticle(root, entry)));
    const summaries: PostSummary[] = [];
    for (const [i, source] of posts.entries()) {
      const post = await renderArticle(source, math.render);
      await writePost(output, post, renderPost(post, posts, i));
      summaries.push(post.info);
      archives[source.dateIndex] = post.info;
    }
    categories.push({ id, title, posts: summaries });
  }
  await copySiteAssets(root, output);
  await writeFile(path.join(output, 'assets/code-theme.css'), codeStylesheet());
  await writeFile(path.join(output, 'assets/math.css'), math.stylesheet());
  await writeFile(path.join(output, 'index.html'), renderHome(categories));
  await mkdir(path.join(output, 'archives'), { recursive: true });
  await writeFile(path.join(output, 'archives/index.html'), renderArchives(archives));
  await buildKeepDigging(path.join(root, '../keep-digging'), output);
  await writeFile(path.join(output, '.nojekyll'), '');
  await writeFile(path.join(output, 'sitemap.xml'), `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${['', 'archives/', 'keep-digging/', ...archives.map((post) => post.route)].map((route) => `<url><loc>${siteUrl}${route}</loc></url>`).join('')}</urlset>\n`);
  return archives.length;
}

const staging = await mkdtemp(path.join(root, '.cache/pages-'));
try {
  const count = await build(staging);
  const destination = path.join(root, 'dist');
  await rm(destination, { recursive: true, force: true });
  await rename(staging, destination);
  console.log(`Built ${count} Lake articles, home, archives, and Keep Digging.`);
} finally {
  await rm(staging, { recursive: true, force: true });
}

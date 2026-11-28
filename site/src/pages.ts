import { escape, version, renderPage } from './html.ts';
import type { CategoryGroup } from './catalog.ts';
import type { Post, PostInfo, PostSummary } from './article.ts';

const dates = new Intl.DateTimeFormat('zh-CN', { dateStyle: 'long', timeZone: 'UTC' });
const months = new Intl.DateTimeFormat('zh-CN', { month: 'long', timeZone: 'UTC' });
const metadata = (post: PostSummary) => `<time datetime="${post.date}">${dates.format(new Date(post.date))}</time><span>约 ${post.readingMinutes} 分钟</span>`;

// Keep ambient movement in the gutters, clear of the reading column.
const ambientGutters = (kind: 'day' | 'night') => ['left', 'right'].map((side, sideIndex) => {
  const points: [number, number][] = kind === 'day'
    ? [[21, 35], [66, 51], [42, 71], [79, 83], [28, 92]]
    : [[19, 16], [65, 27], [34, 39], [83, 48], [21, 61], [62, 75], [42, 87]];
  const particles = points.map(([x, y], index) =>
    `<i style="--x:${sideIndex ? 100 - x : x}%;--y:${(y + sideIndex * 9) % 94}%;--duration:${(kind === 'day' ? 12 : 5) + index * 1.3}s;--delay:-${index * 2.7 + sideIndex * 3.4}s"></i>`).join('');
  return `<span class="home-sky-gutter home-sky-${side}">${particles}${kind === 'night' ? '<b class="home-meteor"></b>' : ''}</span>`;
}).join('');

export function renderHome(categories: CategoryGroup<PostSummary>[]) {
  return renderPage({
    title: 'Blog · 技术笔记', description: '关于模型、论文与源码的技术笔记。', route: '',
    body: `<div class="home-garden" aria-hidden="true">
    <img class="home-background-day" src="assets/garden-background.png?v=${version}" width="1536" height="1024" alt="" decoding="async" fetchpriority="high" draggable="false">
    <img class="home-background-night" src="assets/starry-background.png?v=${version}" width="1536" height="1024" alt="" decoding="async" draggable="false">
    <div class="home-atmosphere home-atmosphere-day">${ambientGutters('day')}</div>
    <div class="home-atmosphere home-atmosphere-night">${ambientGutters('night')}</div>
  </div>
  <section class="garden-articles" aria-label="文章目录">
    <nav class="category-nav" aria-label="文章分类">
      <button type="button" data-category="all" aria-pressed="true">全部</button>
      ${categories.map(({ id, title }) =>
      `<button type="button" data-category="${id}" aria-pressed="false">${escape(title)}</button>`).join('')}</nav>
    ${categories.map(({ id, title, posts: categoryPosts }) => {
      return `<section class="post-group" data-category="${id}" aria-labelledby="category-${id}">
      <div class="post-group-heading"><h2 id="category-${id}">${escape(title)}</h2><span>${categoryPosts.length} 篇</span></div>
      <div class="post-list">${categoryPosts.map((post) => `<article class="post-entry" data-slug="${escape(post.slug)}">
      <h3><a href="${post.route}">${escape(post.title)}</a></h3>
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
}

export function renderArchives(posts: PostSummary[]) {
  const years = Map.groupBy(posts, post => post.date.slice(0, 4));
  return renderPage({
    title: '归档 · Blog', description: 'Blog 技术笔记归档。', route: 'archives/',
    body: `<header class="page-header"><h1>归档</h1><p>${posts.length} 篇文章</p></header>
    ${[...years].map(([year, yearPosts]) => {
      const byMonth = Map.groupBy(yearPosts, post => post.date.slice(0, 7));
      return `<section class="archive-year" aria-labelledby="year-${year}">
        <h2 id="year-${year}">${year} <span>${yearPosts.length}</span></h2>
        ${[...byMonth].map(([month, monthPosts]) => `<div class="archive-month">
          <h3>${months.format(new Date(`${month}-01T00:00:00Z`))}</h3>
          <div class="archive-entries">${monthPosts.map((post) => `<article class="archive-entry">
            <h4><a href="../${post.route}">${escape(post.title)}</a></h4>
            <div class="post-meta">${metadata(post)}</div>
          </article>`).join('\n')}</div>
        </div>`).join('\n')}
      </section>`;
    }).join('\n')}`,
  });
}

export function renderPost({ info: post, content, toc }: Post, group: readonly { info: PostInfo }[], index: number) {
  const previous = group.at(index - 1)!.info;
  const next = group[(index + 1) % group.length]!.info;
  return renderPage({
    title: `${post.title} · Blog`, description: post.description, route: post.route,
    body: `<article class="post-single" data-post="${escape(post.slug)}">
      <header class="post-header">
        <h1 id="${escape(post.titleId)}">${escape(post.title)}</h1>
        <div class="post-meta">${metadata(post)}<span>taking-lying-flat</span></div>
      </header>
      <div class="lake-document">${content}</div>
      <footer class="post-footer">
        <div class="post-topics" aria-label="文章主题">${post.tags.map((tag) => `<span>${escape(tag)}</span>`).join('')}</div>
        <div class="post-actions"><a href="#top">返回顶部 ↑</a></div>
        ${group.length > 1 ? `<nav class="post-pagination" aria-label="同类文章翻页">
          <a href="../${previous.slug}/"><span>← 上一篇</span>${escape(previous.title)}</a>
          <a class="post-next" href="../${next.slug}/"><span>下一篇 →</span>${escape(next.title)}</a>
        </nav>` : ''}
        <a class="back-link" href="../../">← 全部文章</a>
      </footer>
    </article>${toc}`,
  });
}

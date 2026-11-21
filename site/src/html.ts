export type ArticleRoute = `posts/${string}/`;
type IndexRoute = '' | 'archives/' | 'keep-digging/';
type PageOptions = { title: string; description: string; body: string; route: IndexRoute | ArticleRoute };
const pageClasses: Record<IndexRoute, string> & Partial<Record<ArticleRoute, string>> = {
  '': 'home-page', 'archives/': 'archive-page', 'keep-digging/': 'contributions-page',
};

export const siteUrl = 'https://taking-lying-flat.github.io/blog/';
export const version = Date.now().toString(36);
const htmlEntities: Record<string, string> = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' };
export const escape = (value: string) => value.replace(/[&<>"]/g, character => htmlEntities[character] ?? character);

export function renderPage({ title, description, route, body }: PageOptions) {
  const pageClass = pageClasses[route] ?? 'post-page';
  const article = pageClass === 'post-page';
  const root = '../'.repeat(route.split('/').filter(Boolean).length) || './';
  const current = (section: IndexRoute) => section === route ? ' aria-current="page"' : '';
  const styles = (article ? ['math.css', 'code-theme.css', 'reader.css', 'lake.css', 'toc.css'] : ['reader.css'])
    .map(file => root + 'assets/' + file);
  if (route === 'keep-digging/') styles.push('style.css');
  const links = styles.map(file => `<link rel="stylesheet" href="${file}?v=${version}">`).join('\n  ');
  return `<!doctype html>
<html lang="zh-CN" data-color-mode="auto" data-light-theme="light_tritanopia" data-dark-theme="dark_tritanopia">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#ffffff">
  <meta name="description" content="${escape(description)}">
  <meta property="og:type" content="${article ? 'article' : 'website'}">
  <meta property="og:title" content="${escape(title)}">
  <meta property="og:description" content="${escape(description)}">
  <meta property="og:url" content="${siteUrl}${route}">
  <meta name="twitter:card" content="summary">
  <title>${escape(title)}</title>
  <link rel="canonical" href="${siteUrl}${route}">
  <link rel="icon" href="${root}assets/github-avatar.jpg?v=${version}" type="image/jpeg" sizes="160x160">
  <link rel="stylesheet" href="${root}assets/fonts/index.css">
  <link rel="stylesheet" href="${root}assets/fonts/noto-sans-sc.css">
  <link rel="stylesheet" href="${root}assets/fonts/noto-serif-sc.css">
  <link rel="stylesheet" href="${root}assets/github-light-tritanopia.css?v=${version}">
  <link rel="stylesheet" href="${root}assets/github-dark-tritanopia.css?v=${version}">
  ${links}
  <script type="module" blocking="render" src="${root}assets/reader.js?v=${version}"></script>
</head>
<body id="top" class="${pageClass}-shell">
  <a class="skip-link" href="#main">跳转到正文</a>
  <header class="site-header">
    <nav class="nav" aria-label="主导航">
      <div class="logo">
        <a href="${root}">Blog</a>
        <button class="theme-toggle" type="button" aria-label="切换配色" title="切换配色" hidden>
          <svg class="moon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><path d="M20.8 13.1A9 9 0 0 1 10.9 3.2a9 9 0 1 0 9.9 9.9Z"/></svg>
          <svg class="sun" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
        </button>
      </div>
      <ul class="menu">
        <li><a href="${root}"${current('')}>文章</a></li>
        <li><a href="${root}archives/"${current('archives/')}>归档</a></li>
        <li><a href="${root}keep-digging/"${current('keep-digging/')}>Keep Digging</a></li>
      </ul>
    </nav>
  </header>
  <main class="main ${pageClass}" id="main">${body}</main>
  <footer class="site-footer">
    ${article ? '' : `<img class="footer-art" src="${root}assets/anime-readers.png?v=${version}" width="2172" height="724" alt="" aria-hidden="true" loading="lazy" decoding="async">`}
    <span>© 2026 <a href="${root}">Blog</a></span>
    <span>托管于 <a href="https://github.com/taking-lying-flat/blog">GitHub Pages</a></span>
  </footer>
</body>
</html>\n`;
}

(() => {
  const key = 'blog-bookmarks-v1';
  const read = () => {
    try {
      const value = JSON.parse(localStorage.getItem(key) ?? '[]');
      return new Set(Array.isArray(value) ? value.filter(slug => typeof slug === 'string') : []);
    } catch { return new Set(); }
  };
  window.blogBookmarks = { read };
  const announce = () => window.dispatchEvent(new Event('blog-bookmarks-change'));
  const buttons = [...document.querySelectorAll('[data-bookmark]')];
  const sync = () => {
    const saved = read();
    for (const button of buttons) {
      const active = saved.has(button.dataset.bookmark);
      button.hidden = false;
      button.setAttribute('aria-pressed', String(active));
      button.querySelector('.bookmark-label').textContent = active ? '已收藏' : '收藏';
      button.title = active ? '从当前浏览器的收藏中移除' : '收藏到当前浏览器';
    }
  };
  for (const button of buttons) {
    button.addEventListener('click', () => {
      const saved = read();
      const slug = button.dataset.bookmark;
      if (saved.has(slug)) saved.delete(slug);
      else saved.add(slug);
      const status = document.querySelector('#engagement-status');
      try {
        localStorage.setItem(key, JSON.stringify([...saved]));
        if (status) status.textContent = saved.has(slug) ? '已收藏，可在首页「我的收藏」查看。' : '已取消收藏。';
        announce();
      } catch {
        if (status) status.textContent = '浏览器未允许保存收藏，请允许此网站存储数据后重试。';
      }
    });
  }
  window.addEventListener('blog-bookmarks-change', sync);
  window.addEventListener('storage', event => {
    if (event.key === key || event.key === null) announce();
  });
  sync();

  // Counts are public daily snapshots; clicking the link never invents a like.
  const like = document.querySelector('[data-like-slug]');
  if (like) {
    const refresh = async () => {
      try {
        const response = await fetch('https://raw.githubusercontent.com/taking-lying-flat/blog/main/data/article-likes.json');
        if (!response.ok) return;
        const data = await response.json();
        const count = data.posts?.[like.dataset.likeSlug];
        if (!Number.isSafeInteger(count) || count < 0) return;
        like.querySelector('.like-count').textContent = String(count);
        like.title = `在 GitHub 为本文点 ❤️ · ${data.date} 更新`;
      } catch { /* The last build's snapshot remains visible offline. */ }
    };
    const observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) { observer.disconnect(); refresh(); }
    });
    observer.observe(like);
  }
})();

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

  const like = document.querySelector('[data-like-slug]');
  if (like) {
    let pending = false;
    let initialized = false;
    const status = document.querySelector('#engagement-status');
    const setPending = value => {
      pending = value;
      like.disabled = value;
      like.setAttribute('aria-busy', String(value));
    };
    const request = async action => {
      const url = new URL(like.dataset.likeApi);
      url.searchParams.set('action', action);
      url.searchParams.set('id', like.dataset.likeId);
      url.searchParams.set('_', String(Date.now()));
      const response = await fetch(url, {
        credentials: 'omit', cache: 'no-store',
        signal: AbortSignal.timeout(15000),
      });
      if (!response.ok) throw new Error('Like service unavailable');
      const result = await response.json();
      const data = result.data;
      if (!result.success || data?.id !== like.dataset.likeId ||
          !Number.isSafeInteger(data.total) || data.total < 0 || typeof data.liked !== 'boolean') {
        throw new Error('Invalid like response');
      }
      return data;
    };
    const render = data => {
      like.querySelector('.like-count').textContent = String(data.total);
      like.querySelector('.like-label').textContent = data.liked ? '已赞' : '点赞';
      like.setAttribute('aria-pressed', String(data.liked));
      like.title = data.liked ? '取消点赞' : '点赞';
      initialized = true;
    };
    const refresh = async () => {
      if (pending) return;
      setPending(true);
      try { render(await request('get')); }
      catch { /* Keep the last confirmed snapshot if the service is unavailable. */ }
      finally { setPending(false); }
    };
    like.addEventListener('click', async () => {
      if (pending) return;
      setPending(true);
      if (status) status.textContent = '';
      try {
        if (!initialized) render(await request('get'));
        const data = await request('toggle');
        render(data);
        if (status) status.textContent = data.liked ? '谢谢喜欢！' : '已取消点赞。';
        try { localStorage.setItem('blog-likes-change', JSON.stringify({ slug: like.dataset.likeSlug, time: Date.now() })); }
        catch { /* Likes are stored online and work without local storage. */ }
      } catch {
        // A timed-out write may have succeeded; read back instead of retrying a toggle.
        try { render(await request('get')); } catch { /* Preserve confirmed state. */ }
        if (status) status.textContent = '连接暂时中断，请确认点赞状态后重试。';
      } finally { setPending(false); }
    });
    window.addEventListener('storage', event => {
      if (event.key === 'blog-likes-change') refresh();
    });
    const observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) { observer.disconnect(); refresh(); }
    });
    observer.observe(like);
  }
})();

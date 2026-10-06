const themeButton = document.querySelector('.theme-toggle');
if (themeButton) {
  const syncThemeLabel = () => {
    const dark = document.documentElement.dataset.colorMode === 'dark';
    const label = dark ? '切换到 Light Tritanopia' : '切换到 Dark Tritanopia';
    themeButton.setAttribute('aria-label', label);
    themeButton.title = `${dark ? 'Dark Tritanopia' : 'Light Tritanopia'} · ${label}`;
  };
  themeButton.hidden = false;
  syncThemeLabel();
  themeButton.addEventListener('click', () => {
    const theme = document.documentElement.dataset.colorMode === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = theme;
    document.documentElement.dataset.colorMode = theme;
    document.querySelector('meta[name="theme-color"]').content = getComputedStyle(document.documentElement).getPropertyValue('--bgColor-muted').trim();
    try { localStorage.setItem('blog-theme', theme); } catch {}
    syncThemeLabel();
  });
}

const status = document.querySelector('#copy-status');
if (navigator.clipboard?.writeText) {
  document.querySelectorAll('.copy-button').forEach((button) => {
    button.hidden = false;
    button.addEventListener('click', async () => {
      const code = button.closest('.code-block').querySelector('code').textContent;
      try {
        await navigator.clipboard.writeText(code);
        button.textContent = '已复制';
        status.textContent = '代码已复制';
      } catch {
        button.textContent = '请手动复制';
        status.textContent = '无法访问剪贴板，请选中代码手动复制';
      }
      setTimeout(() => { button.textContent = '复制'; }, 2000);
    });
  });
}

const categoryNav = document.querySelector('.category-nav');
if (categoryNav) {
  const buttons = [...categoryNav.querySelectorAll('[data-category]')];
  const groups = [...document.querySelectorAll('.post-group')];
  const heading = document.querySelector('#post-list-title');
  const form = document.querySelector('.post-search');
  const input = form.querySelector('input');
  const clear = form.querySelector('.search-clear');
  const searchStatus = document.querySelector('#search-status');
  const empty = document.querySelector('.search-empty');
  const savedFilter = document.querySelector('.saved-filter');
  const normalize = text => text.normalize('NFKC').toLowerCase();
  const entries = [...document.querySelectorAll('.post-entry')].map(element => {
    const snippet = document.createElement('p');
    snippet.className = 'search-snippet';
    snippet.hidden = true;
    element.querySelector('h3').after(snippet);
    return {
      element, snippet, slug: element.dataset.slug,
      metadata: normalize(`${element.querySelector('h3').textContent} ${element.querySelector('.entry-topics').textContent}`),
      text: '', normalizedText: '',
    };
  });
  let indexReady = false;
  let indexRequest;
  let indexFailed = false;
  let debounce;

  const renderSnippet = (entry, terms) => {
    const term = terms.find(word => entry.normalizedText.includes(word));
    if (!term) { entry.snippet.hidden = true; return; }
    const match = entry.normalizedText.indexOf(term);
    const start = Math.max(0, match - 40);
    const end = Math.min(entry.text.length, Math.max(start + 150, match + term.length));
    const excerpt = entry.text.slice(start, end);
    entry.snippet.replaceChildren();
    if (start) entry.snippet.append('…');
    const offset = match - start;
    const mark = document.createElement('mark');
    mark.textContent = excerpt.slice(offset, offset + term.length);
    entry.snippet.append(excerpt.slice(0, offset), mark, excerpt.slice(offset + term.length));
    if (end < entry.text.length) entry.snippet.append('…');
    entry.snippet.hidden = false;
  };
  const renderResults = () => {
    const requested = location.hash.replace('#category-', '');
    const selected = buttons.find(button => button.dataset.category === requested) ?? buttons[0];
    const category = selected.dataset.category;
    const savedOnly = new URLSearchParams(location.search).get('saved') === '1';
    const saved = window.blogBookmarks.read();
    savedFilter.hidden = false;
    savedFilter.setAttribute('aria-pressed', String(savedOnly));
    savedFilter.querySelector('span').textContent = String(entries.filter(entry => saved.has(entry.slug)).length);
    const query = input.value.trim();
    const terms = normalize(query).split(/\s+/).filter(Boolean);
    for (const entry of entries) {
      entry.element.hidden = (savedOnly && !saved.has(entry.slug)) || !terms.every(term =>
        entry.metadata.includes(term) || entry.normalizedText.includes(term));
      entry.snippet.hidden = true;
      if (terms.length && !entry.element.hidden) renderSnippet(entry, terms);
    }
    let count = 0;
    for (const group of groups) {
      const matches = group.querySelectorAll('.post-entry:not([hidden])').length;
      group.hidden = (category !== 'all' && group.dataset.category !== category) || matches === 0;
      group.querySelector('.post-group-heading span').textContent = `${matches} 篇`;
      if (!group.hidden) count += matches;
    }
    for (const button of buttons) {
      button.setAttribute('aria-pressed', String(button === selected));
    }
    heading.firstChild.textContent = `${category === 'all' ? (savedOnly ? '我的收藏' : '全部文章') : selected.textContent} `;
    heading.querySelector('span').textContent = String(count).padStart(2, '0');
    clear.hidden = !input.value;
    searchStatus.hidden = !query;
    searchStatus.textContent = query ? indexFailed
      ? `正文搜索暂时无法加载，当前仅搜索标题和标签（${count} 篇）。重新输入可重试。`
      : !indexReady ? '正在加载正文搜索…'
        : `找到 ${count} 篇相关文章${category === 'all' ? '' : ` · ${selected.textContent}`}` : '';
    empty.textContent = savedOnly
      ? '当前筛选下没有收藏文章。可到文章底部点击「收藏」，或切换分类与关键词。'
      : '没有找到相关文章，试试其他关键词或切换到「全部」。';
    empty.hidden = (!query && !savedOnly) || count > 0 || (Boolean(query) && !indexReady && !indexFailed);
    form.setAttribute('aria-busy', String(Boolean(query) && !indexReady && !indexFailed));
  };
  const loadIndex = () => {
    if (indexReady || indexRequest) return;
    indexFailed = false;
    indexRequest = fetch(form.dataset.index)
      .then(response => {
        if (!response.ok) throw new Error('Search index unavailable');
        return response.json();
      })
      .then(records => {
        const bySlug = new Map(records.map(record => [record.slug, record.text]));
        if (entries.some(entry => typeof bySlug.get(entry.slug) !== 'string')) {
          throw new Error('Incomplete search index');
        }
        for (const entry of entries) {
          entry.text = bySlug.get(entry.slug);
          entry.normalizedText = normalize(entry.text);
        }
        indexReady = true;
      })
      .catch(() => { indexFailed = true; })
      .finally(() => { indexRequest = undefined; renderResults(); });
  };
  const search = () => {
    clearTimeout(debounce);
    const url = new URL(location.href);
    if (input.value.trim()) url.searchParams.set('q', input.value.trim());
    else url.searchParams.delete('q');
    history.replaceState(null, '', `${url.pathname}${url.search}${url.hash}`);
    if (input.value.trim()) loadIndex();
    renderResults();
  };
  const syncLocation = () => {
    clearTimeout(debounce);
    input.value = new URLSearchParams(location.search).get('q') ?? '';
    if (input.value.trim()) loadIndex();
    renderResults();
    if (location.hash === '#search-input') input.focus();
  };
  form.hidden = false;
  input.addEventListener('focus', loadIndex);
  input.addEventListener('input', event => {
    clearTimeout(debounce);
    clear.hidden = !input.value;
    if (!event.isComposing) debounce = setTimeout(search, 120);
  });
  input.addEventListener('compositionend', search);
  form.addEventListener('submit', event => { event.preventDefault(); search(); });
  clear.addEventListener('click', () => { input.value = ''; search(); input.focus(); });
  savedFilter.addEventListener('click', () => {
    search();
    const url = new URL(location.href);
    if (url.searchParams.get('saved') === '1') url.searchParams.delete('saved');
    else url.searchParams.set('saved', '1');
    history.pushState(null, '', `${url.pathname}${url.search}${url.hash}`);
    renderResults();
  });
  window.addEventListener('blog-bookmarks-change', renderResults);
  categoryNav.addEventListener('click', event => {
    const button = event.target.closest('button[data-category]');
    if (!button || button.getAttribute('aria-pressed') === 'true') return;
    search();
    const hash = button.dataset.category === 'all' ? '' : `#category-${button.dataset.category}`;
    history.pushState(null, '', `${location.pathname}${location.search}${hash}`);
    renderResults();
  });
  window.addEventListener('popstate', syncLocation);
  window.addEventListener('hashchange', syncLocation);
  syncLocation();
}

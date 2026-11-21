import type { SearchRecord } from './src/catalog.ts';

if (document.documentElement.hasAttribute('data-redirect')) {
  const canonical = document.querySelector<HTMLLinkElement>('link[rel="canonical"]')!;
  location.replace(canonical.href + location.search + location.hash);
} else {
  setupTheme();
  const categories = document.querySelector<HTMLElement>('.category-nav');
  if (categories) setupSearch(categories);
  const contents = document.querySelector<HTMLElement>('.post-toc');
  if (contents) setupContents(contents);
}

function setupTheme() {
  type Theme = 'light' | 'dark';
  const root = document.documentElement;
  const apply = (theme: Theme) => {
    root.dataset.colorMode = theme;
    document.querySelector<HTMLMetaElement>('meta[name="theme-color"]')!.content =
      getComputedStyle(root).getPropertyValue('--bgColor-muted').trim();
  };
  let stored: string | null = null;
  try { stored = localStorage.getItem('blog-theme'); } catch {}
  apply(stored === 'light' || stored === 'dark' ? stored :
    matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');

  const button = document.querySelector<HTMLButtonElement>('.theme-toggle')!;
  const updateLabel = () => {
    const dark = root.dataset.colorMode === 'dark';
    const label = dark ? '切换到 Light Tritanopia' : '切换到 Dark Tritanopia';
    button.setAttribute('aria-label', label);
    button.title = `${dark ? 'Dark Tritanopia' : 'Light Tritanopia'} · ${label}`;
  };
  updateLabel();
  button.hidden = false;
  button.addEventListener('click', () => {
    const theme = root.dataset.colorMode === 'dark' ? 'light' : 'dark';
    apply(theme);
    updateLabel();
    try { localStorage.setItem('blog-theme', theme); } catch {}
  });
}

function isSearchRecord(value: unknown): value is SearchRecord {
  return typeof value === 'object' && value !== null &&
    'slug' in value && typeof value.slug === 'string' &&
    'text' in value && typeof value.text === 'string';
}

function setupSearch(categoryNav: HTMLElement) {
  const buttons = [...categoryNav.querySelectorAll<HTMLButtonElement>('[data-category]')];
  const all = buttons.find(button => button.dataset.category === 'all');
  if (!all) return;
  const heading = document.querySelector<HTMLElement>('#post-list-title')!;
  const form = document.querySelector<HTMLFormElement>('.post-search')!;
  const input = form.querySelector('input')!;
  const clear = form.querySelector<HTMLButtonElement>('.search-clear')!;
  const searchStatus = document.querySelector<HTMLElement>('#search-status')!;
  const empty = document.querySelector<HTMLElement>('.search-empty')!;
  const normalize = (text: string) => text.normalize('NFKC').toLowerCase();
  const headingCount = heading.querySelector('span')!;
  const groups = [...document.querySelectorAll<HTMLElement>('.post-group')].map(element => ({
    element,
    count: element.querySelector<HTMLElement>('.post-group-heading span')!,
    entries: [...element.querySelectorAll<HTMLElement>('.post-entry')].map(article => {
      const title = article.querySelector('h3')!;
      const metadata = normalize(`${title.textContent} ${article.querySelector('.entry-topics')!.textContent}`);
      return {
        article, title, metadata, slug: article.dataset.slug!,
        text: '', normalizedText: '', snippet: null as HTMLParagraphElement | null,
      };
    }),
  }));
  type Entry = typeof groups[number]['entries'][number];
  let indexState: 'idle' | 'loading' | 'ready' | 'failed' = 'idle';
  let debounce: number;

  function showSnippet(entry: Entry, match: number, length: number) {
    if (!entry.snippet) {
      entry.snippet = document.createElement('p');
      entry.snippet.className = 'search-snippet';
      entry.title.after(entry.snippet);
    }
    const start = Math.max(0, match - 40);
    const end = Math.min(entry.text.length, Math.max(start + 150, match + length));
    const excerpt = entry.text.slice(start, end);
    entry.snippet.replaceChildren();
    if (start) entry.snippet.append('…');
    const offset = match - start;
    const mark = document.createElement('mark');
    mark.textContent = excerpt.slice(offset, offset + length);
    entry.snippet.append(excerpt.slice(0, offset), mark, excerpt.slice(offset + length));
    if (end < entry.text.length) entry.snippet.append('…');
    entry.snippet.hidden = false;
  }
  const renderResults = () => {
    const requested = location.hash.replace('#category-', '');
    const selected = buttons.find(button => button.dataset.category === requested) ?? all;
    const category = selected.dataset.category;
    const query = input.value.trim();
    const terms = normalize(query).split(/\s+/).filter(Boolean);
    let count = 0;
    for (const group of groups) {
      if (category !== 'all' && group.element.dataset.category !== category) {
        group.element.hidden = true;
        continue;
      }
      let matches = 0;
      for (const entry of group.entries) {
        let first = -1;
        let length = 0;
        const visible = terms.every(term => {
          const index = indexState === 'ready' ? entry.normalizedText.indexOf(term) : -1;
          if (index >= 0 && first < 0) { first = index; length = term.length; }
          return index >= 0 || entry.metadata.includes(term);
        });
        entry.article.hidden = !visible;
        if (visible && first >= 0) showSnippet(entry, first, length);
        else if (entry.snippet) entry.snippet.hidden = true;
        if (visible) matches++;
      }
      group.element.hidden = matches === 0;
      group.count.textContent = `${matches} 篇`;
      count += matches;
    }
    for (const button of buttons) {
      button.setAttribute('aria-pressed', String(button === selected));
    }
    heading.firstChild!.textContent = `${category === 'all' ? '全部文章' : selected.textContent} `;
    headingCount.textContent = String(count).padStart(2, '0');
    clear.hidden = !input.value;
    searchStatus.hidden = !query;
    searchStatus.textContent = query ? indexState === 'failed'
      ? `正文搜索暂时无法加载，当前仅搜索标题和标签（${count} 篇）。重新输入可重试。`
      : indexState !== 'ready' ? '正在加载正文搜索…'
        : `找到 ${count} 篇相关文章${category === 'all' ? '' : ` · ${selected.textContent}`}` : '';
    empty.hidden = !query || count > 0 || indexState === 'loading';
    form.setAttribute('aria-busy', String(Boolean(query) && indexState === 'loading'));
  };
  async function loadIndex() {
    if (indexState === 'ready' || indexState === 'loading') return;
    indexState = 'loading';
    try {
      const response = await fetch(form.dataset.index!);
      if (!response.ok) throw new Error('Search index unavailable');
      const records: unknown = await response.json();
      if (!Array.isArray(records)) throw new Error('Invalid search index');
      const bySlug = new Map<string, string>();
      for (const record of records) {
        if (!isSearchRecord(record)) throw new Error('Invalid search record');
        bySlug.set(record.slug, record.text);
      }
      for (const group of groups) {
        for (const entry of group.entries) {
          const text = bySlug.get(entry.slug);
          if (typeof text !== 'string') throw new Error(`Missing search text: ${entry.slug}`);
          entry.text = text;
          entry.normalizedText = text.toLowerCase();
        }
      }
      indexState = 'ready';
    } catch {
      indexState = 'failed';
    }
    renderResults();
  }
  const queryUrl = () => {
    clearTimeout(debounce);
    const url = new URL(location.href);
    if (input.value.trim()) url.searchParams.set('q', input.value.trim());
    else url.searchParams.delete('q');
    return url;
  };
  const search = () => {
    history.replaceState(null, '', queryUrl());
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
    if (!(event as InputEvent).isComposing) debounce = window.setTimeout(search, 120);
  });
  input.addEventListener('compositionend', search);
  form.addEventListener('submit', event => { event.preventDefault(); search(); });
  clear.addEventListener('click', () => { input.value = ''; search(); input.focus(); });
  categoryNav.addEventListener('click', event => {
    const button = (event.target as Element).closest<HTMLButtonElement>('button[data-category]');
    if (!button || button.getAttribute('aria-pressed') === 'true') return;
    const url = queryUrl();
    history.replaceState(null, '', url);
    url.hash = button.dataset.category === 'all' ? '' : `#category-${button.dataset.category}`;
    history.pushState(null, '', url);
    syncLocation();
  });
  window.addEventListener('popstate', syncLocation);
  window.addEventListener('hashchange', syncLocation);
  syncLocation();
}

function setupContents(paperToc: HTMLElement) {
  const panel = paperToc.querySelector('details')!;
  const main = paperToc.closest('main')!;
  const navigation = paperToc.querySelector('nav')!;
  const sections = [...navigation.querySelectorAll<HTMLAnchorElement>('.toc-link')].flatMap(link => {
    const heading = document.getElementById(decodeURIComponent(link.hash.slice(1)));
    return heading ? [{ link, heading }] : [];
  });
  if (!sections.length) return;
  const preferenceKey = 'blog-paper-toc';
  let preference;
  try { preference = localStorage.getItem(preferenceKey); } catch {}
  panel.open = preference !== 'closed';
  const syncPanel = () => main.classList.toggle('toc-collapsed', !panel.open);
  syncPanel();
  panel.addEventListener('toggle', () => {
    syncPanel();
    try { localStorage.setItem(preferenceKey, panel.open ? 'open' : 'closed'); } catch {}
    schedule();
  });

  let activeLink: HTMLAnchorElement | undefined;
  function updateActive() {
    // Headings follow document order. Find the last one above the reading line.
    let start = 0;
    let end = sections.length;
    while (start < end) {
      const middle = Math.floor((start + end) / 2);
      if (sections[middle]!.heading.getBoundingClientRect().top <= 100) start = middle + 1;
      else end = middle;
    }
    const current = sections[Math.max(0, start - 1)]!.link;
    if (current !== activeLink) {
      activeLink?.removeAttribute('aria-current');
      current.setAttribute('aria-current', 'location');
      activeLink = current;
    }
    // Keep the current section visible without moving the article itself.
    if (panel.open) {
      const item = current.getBoundingClientRect();
      const viewport = navigation.getBoundingClientRect();
      if (item.top < viewport.top) navigation.scrollTop += item.top - viewport.top;
      else if (item.bottom > viewport.bottom) navigation.scrollTop += item.bottom - viewport.bottom;
    }
  }

  let scheduled = false;
  function schedule() {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => {
      scheduled = false;
      updateActive();
    });
  }
  addEventListener('scroll', schedule, { passive: true });
  addEventListener('resize', schedule);
  addEventListener('hashchange', schedule);
  document.fonts.ready.then(schedule);
  new ResizeObserver(schedule).observe(main.querySelector('.post-single')!);
  schedule();
}

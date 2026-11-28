if (document.documentElement.hasAttribute('data-redirect')) {
  const canonical = document.querySelector<HTMLLinkElement>('link[rel="canonical"]')!;
  location.replace(canonical.href + location.search + location.hash);
} else {
  setupTheme();
  const categories = document.querySelector<HTMLElement>('.category-nav');
  if (categories) setupCategories(categories);
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

function setupCategories(categoryNav: HTMLElement) {
  const buttons = [...categoryNav.querySelectorAll<HTMLButtonElement>('[data-category]')];
  const all = buttons.find(button => button.dataset.category === 'all');
  if (!all) return;
  const groups = [...document.querySelectorAll<HTMLElement>('.post-group')];
  const syncLocation = () => {
    const requested = location.hash.replace('#category-', '');
    const selected = buttons.find(button => button.dataset.category === requested) ?? all;
    for (const group of groups) {
      group.hidden = selected !== all && group.dataset.category !== selected.dataset.category;
    }
    for (const button of buttons) {
      button.setAttribute('aria-pressed', String(button === selected));
    }
  };
  categoryNav.addEventListener('click', event => {
    const button = (event.target as Element).closest<HTMLButtonElement>('button[data-category]');
    if (!button || button.getAttribute('aria-pressed') === 'true') return;
    const url = new URL(location.href);
    url.hash = button === all ? '' : `#category-${button.dataset.category}`;
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

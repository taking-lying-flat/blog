const paperToc = document.querySelector('.post-toc');

if (paperToc) {
  const panel = paperToc.querySelector('details');
  const main = paperToc.closest('main');
  const navigation = paperToc.querySelector('nav');
  const links = [...navigation.querySelectorAll('.toc-link')];
  const headings = links.map(link => document.getElementById(decodeURIComponent(link.hash.slice(1))));
  const preferenceKey = 'blog-paper-toc';
  let preference;
  try { preference = localStorage.getItem(preferenceKey); } catch {}
  panel.open = preference ? preference === 'open' : !matchMedia('(max-width: 1199px)').matches;
  const syncPanel = () => main.classList.toggle('toc-collapsed', !panel.open);
  syncPanel();
  panel.addEventListener('toggle', () => {
    syncPanel();
    try { localStorage.setItem(preferenceKey, panel.open ? 'open' : 'closed'); } catch {}
    schedule();
  });

  let activeLink;
  function updateActive() {
    let index = 0;
    for (let i = 0; i < headings.length; i++) {
      if (headings[i].getBoundingClientRect().top > 100) break;
      index = i;
    }
    const current = links[index];
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

  let frame;
  function schedule() {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(updateActive);
  }
  addEventListener('scroll', schedule, { passive: true });
  addEventListener('resize', schedule);
  addEventListener('hashchange', schedule);
  document.fonts.ready.then(schedule);
  new ResizeObserver(schedule).observe(main.querySelector('.post-single'));
  schedule();
}

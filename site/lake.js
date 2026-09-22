const lakeDocument = document.querySelector('.lake-document');

if (lakeDocument) {
  const paragraphs = [...lakeDocument.querySelectorAll('p, li')].filter(element =>
    element.textContent.trim() && !element.querySelector('p, ul, ol') &&
    !['center', 'right', 'end'].includes(getComputedStyle(element).textAlign));
  const originals = new Map(paragraphs.map(element => [element, {
    letterSpacing: element.style.letterSpacing,
    textWrap: element.style.textWrap,
  }]));
  const range = document.createRange();

  function lines(element, fontSize) {
    const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
    const fragments = [];
    while (walker.nextNode()) {
      const node = walker.currentNode;
      const start = node.textContent.search(/\S/u);
      if (start < 0) continue;
      range.setStart(node, start);
      range.setEnd(node, node.textContent.trimEnd().length);
      fragments.push(...[...range.getClientRects()].filter(rect => rect.width > 0));
    }
    fragments.sort((a, b) => a.top - b.top);
    const rows = [];
    for (const rect of fragments) {
      const row = rows.at(-1);
      if (!row || rect.top - row.top > fontSize / 2) rows.push({ top: rect.top, bottom: rect.bottom, right: rect.right });
      else { row.right = Math.max(row.right, rect.right); row.bottom = Math.max(row.bottom, rect.bottom); }
    }
    if (rows.length < 2) return { count: rows.length, orphan: false };
    const last = rows.at(-1);
    const middle = (last.top + last.bottom) / 2;
    for (const image of element.querySelectorAll('.lake-math, .lake-image')) {
      const rect = image.getBoundingClientRect();
      // A following formula/image on its own line is not a stranded word.
      if (rect.top >= last.bottom) return { count: rows.length, orphan: false };
      if (rect.top <= middle && rect.bottom >= middle) last.right = Math.max(last.right, rect.right);
    }
    const width = last.right - element.getBoundingClientRect().left - parseFloat(getComputedStyle(element).paddingLeft);
    return { count: rows.length, orphan: width < fontSize * 3.5 };
  }

  function fitParagraphs() {
    for (const element of paragraphs) Object.assign(element.style, originals.get(element));
    for (const element of paragraphs) {
      const style = getComputedStyle(element);
      const fontSize = parseFloat(style.fontSize);
      const before = lines(element, fontSize);
      if (!before.orphan) continue;
      let low = -fontSize * .04;
      let high = parseFloat(style.letterSpacing) || 0;
      element.style.letterSpacing = `${low}px`;
      if (lines(element, fontSize).count < before.count) {
        // Use the smallest adjustment that brings the short final line back.
        for (let step = 0; step < 8; step++) {
          const middle = (low + high) / 2;
          element.style.letterSpacing = `${middle}px`;
          if (lines(element, fontSize).count < before.count) low = middle;
          else high = middle;
        }
        element.style.letterSpacing = `${low}px`;
      } else {
        element.style.letterSpacing = originals.get(element).letterSpacing;
        element.style.textWrap = 'balance';
      }
    }
  }

  let pendingFrame;
  const schedule = () => {
    cancelAnimationFrame(pendingFrame);
    pendingFrame = requestAnimationFrame(fitParagraphs);
  };
  document.fonts.ready.then(() => {
    schedule();
    let previousWidth = -1;
    new ResizeObserver(([entry]) => {
      if (Math.abs(entry.contentRect.width - previousWidth) < .5) return;
      previousWidth = entry.contentRect.width;
      schedule();
    }).observe(lakeDocument);
  });
  document.fonts.addEventListener('loadingdone', schedule);
}

const lakeDocument = document.querySelector('.lake-document');

if (lakeDocument) {
  // A formula image and its closing punctuation must wrap as one unit.
  for (const math of lakeDocument.querySelectorAll('.lake-math:not(.lake-math-numbered):not(.lake-math-wide)')) {
    const next = math.nextElementSibling;
    if (math.nextSibling !== next) continue;
    const text = next?.firstChild;
    if (!next?.matches('span[data-lake-id]') || text?.nodeType !== Node.TEXT_NODE) continue;
    const punctuation = text.data.match(/^[，。！？；：、）】》」』,.!?;:)\]]+/u)?.[0];
    if (!punctuation) continue;
    const group = document.createElement('span');
    group.className = 'lake-math-punctuation';
    const tail = next.cloneNode(false);
    tail.removeAttribute('id');
    tail.removeAttribute('data-lake-id');
    tail.textContent = punctuation;
    math.before(group);
    group.append(math, tail);
    text.deleteData(0, punctuation.length);
  }
}

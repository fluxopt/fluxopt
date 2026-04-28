// Strip "In " / "Out" prefixes from notebook prompts (keep only [N]:),
// and make the whole input code block clickable to copy.
//
// Subscribes to Material's `document$` instant-nav lifecycle when available
// so both behaviours re-attach on every page transition.

function stripPromptPrefixes() {
  document.querySelectorAll('.jp-InputPrompt, .jp-OutputPrompt').forEach((el) => {
    if (el.dataset.fixed) return;
    const m = el.textContent.match(/^\s*(?:In|Out)\s*(\[\s*\d*\s*\]:?)\s*$/);
    if (m) {
      el.textContent = m[1];
      el.dataset.fixed = '1';
    }
  });
}

function attachClickToCopy() {
  // Notebook code cells AND markdown code fences (skip inline `code` and
  // mkdocstrings .doc-signature — neither is meant for copy-the-block UX)
  const blocks = document.querySelectorAll(
    '.jp-CodeCell .highlight-ipynb, .md-typeset .highlight'
  );
  blocks.forEach((block) => {
    if (block.dataset.copyAttached) return;
    if (block.closest('.doc-signature')) return;
    block.dataset.copyAttached = '1';
    block.addEventListener('click', (event) => {
      // Don't fire if the user is selecting text (or just clicked a link inside)
      if (window.getSelection().toString().length > 0) return;
      if (event.target.closest('a, button')) return;

      const codeEl = block.querySelector('pre');
      if (!codeEl) return;
      const text = codeEl.innerText;

      navigator.clipboard.writeText(text).then(() => {
        block.classList.add('copied');
        setTimeout(() => block.classList.remove('copied'), 700);
      }).catch(() => {});
    });
  });
}

function init() {
  stripPromptPrefixes();
  attachClickToCopy();
}

if (typeof document$ !== 'undefined') {
  document$.subscribe(init);
} else {
  document.addEventListener('DOMContentLoaded', init);
}

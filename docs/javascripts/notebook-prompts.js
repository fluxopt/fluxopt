// Strip "In " / "Out" prefixes from notebook prompts, keep only [N]:
// Subscribes to Material's instant-nav lifecycle when available so it
// re-runs on every page transition, not just initial load.

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

if (typeof document$ !== 'undefined') {
  document$.subscribe(stripPromptPrefixes);
} else {
  document.addEventListener('DOMContentLoaded', stripPromptPrefixes);
}

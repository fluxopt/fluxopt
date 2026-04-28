// Inject a "Download notebook" link at the top of each notebook page.
// The .ipynb source is copied into the build dir alongside index.html by
// mkdocs-jupyter's `include_source: true`, so the URL is just <slug>.ipynb
// relative to the current page.
//
// Subscribes to Material's `document$` instant-nav lifecycle so the button
// re-attaches on every page transition.

function injectNotebookDownload() {
  const wrapper = document.querySelector('.jupyter-wrapper');
  if (!wrapper) return;
  if (wrapper.parentNode.querySelector('.notebook-download')) return;

  const path = window.location.pathname.replace(/\/$/, '');
  const slug = path.split('/').pop();
  if (!slug) return;

  const link = document.createElement('a');
  link.className = 'notebook-download';
  link.href = `${slug}.ipynb`;
  link.download = `${slug}.ipynb`;
  link.title = 'Download notebook (.ipynb)';
  link.innerHTML = '<span class="notebook-download__icon">↓</span> Download notebook';
  wrapper.parentNode.insertBefore(link, wrapper);
}

if (typeof document$ !== 'undefined') {
  document$.subscribe(injectNotebookDownload);
} else {
  document.addEventListener('DOMContentLoaded', injectNotebookDownload);
}

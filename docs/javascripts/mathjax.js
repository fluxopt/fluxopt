window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

// Re-typeset on every instant-nav page swap. The startup.promise gate
// avoids racing the initial load. Resetting startup.document forgets
// the previous page's element list (so stale "already typeset" tracking
// doesn't skip new elements) without ripping rendered DOM out.
document$.subscribe(() => {
  if (typeof MathJax === "undefined" || !MathJax.startup) return;
  MathJax.startup.promise = MathJax.startup.promise
    .then(() => {
      if (MathJax.startup.document && MathJax.startup.document.clear) {
        MathJax.startup.document.clear();
      }
      return MathJax.typesetPromise();
    })
    .catch((err) => console.error("MathJax typeset failed:", err));
});

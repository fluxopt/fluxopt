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

// On every instant-nav page swap, re-typeset math — but only after MathJax
// has finished its initial startup. Without the startup.promise gate we hit
// a race on first navigations where MathJax isn't ready yet and typesetPromise
// silently fails, leaving raw \(...\) on the page until a hard reload.
document$.subscribe(() => {
  if (typeof MathJax === "undefined" || !MathJax.startup) return;
  MathJax.startup.promise.then(() => {
    MathJax.typesetClear();
    MathJax.texReset();
    return MathJax.typesetPromise();
  });
});

// Math in answers, typeset by KaTeX (vendored in /static/vendor/katex, loaded by index.html).
//
// katex.render builds DOM nodes directly (no HTML strings). The answer is untrusted, so: trust off
// (no \href, \url, \includegraphics, \htmlClass ...), errors shown as text instead of thrown, and
// limits on size and macro expansion so a hostile formula can't hang the page.
// Without KaTeX (failed to load), the formula is shown as code.

import { el } from "./dom.js";

const OPTIONS = {
  throwOnError: false,
  trust: false,
  strict: "ignore",
  maxSize: 20,
  maxExpand: 300,
  output: "htmlAndMathml", // MathML for screen readers, HTML for the eye
};

export function renderMath(tex, display) {
  const node = el(display ? "div" : "span", { class: display ? "math display" : "math" });
  const katex = globalThis.katex;
  if (!katex || tex.length > 2000) {
    node.append(el("code", {}, tex));
    return node;
  }
  try {
    katex.render(tex, node, { ...OPTIONS, displayMode: display });
  } catch {
    node.replaceChildren(el("code", {}, tex));
  }
  return node;
}

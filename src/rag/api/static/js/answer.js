// The answer text: the Markdown subset of format.js, with each [n] citation a button pointing at
// source n once the answer is final, and math typeset by KaTeX (math.js).

import { clear, el } from "./dom.js";
import { blocks } from "./format.js";
import { renderMath } from "./math.js";

// While streaming: formatted, but citations stay as text (guardrails may still change them).
export function renderStreaming(container, text) {
  clear(container);
  const nodes = blocks(text, { cite: () => null, math: renderMath });
  container.append(...(nodes.length ? nodes : [el("p")]));
}

// Final answer: citation buttons for numbers that exist; others stay as written.
export function renderFinal(container, text, nSources, onCite) {
  clear(container);
  const cite = (n) =>
    n >= 1 && n <= nSources
      ? el("button", { type: "button", class: "cite", "aria-label": `Show source ${n}`, onclick: () => onCite(n) }, `${n}`)
      : null;
  container.append(...blocks(text, { cite, math: renderMath }));
}

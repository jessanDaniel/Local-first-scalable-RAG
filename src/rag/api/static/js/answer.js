// The answer text: paragraphs, with each [n] citation turned into a button that points at source n.

import { clear, el } from "./dom.js";

// [1], [1, 3], [2-4], [2–4]; the same forms the server's citation check accepts.
const MARKER = /\[(\d+(?:\s*[,–-]\s*\d+)*)\]/g;

export function markerNumbers(body) {
  const numbers = [];
  for (const part of body.split(",")) {
    const range = part.split(/[–-]/).map((x) => parseInt(x.trim(), 10));
    if (range.length === 2 && range[0] <= range[1] && range[1] - range[0] < 50) {
      for (let n = range[0]; n <= range[1]; n++) numbers.push(n);
    } else if (!Number.isNaN(range[0])) {
      numbers.push(range[0]);
    }
  }
  return numbers;
}

function paragraphs(text) {
  return text
    .trim()
    .split(/\n{2,}/)
    .filter((p) => p.trim());
}

// While streaming: plain text, no buttons (the raw text may still change once guardrails run).
export function renderStreaming(container, text) {
  clear(container);
  const parts = paragraphs(text);
  for (const p of parts.length ? parts : [""]) container.append(el("p", {}, p));
}

// Final answer: citation buttons for numbers that exist; others stay as written.
export function renderFinal(container, text, nSources, onCite) {
  clear(container);
  for (const p of paragraphs(text)) {
    const node = el("p");
    let last = 0;
    for (const match of p.matchAll(MARKER)) {
      node.append(p.slice(last, match.index));
      for (const n of markerNumbers(match[1])) {
        if (n >= 1 && n <= nSources) {
          node.append(
            el("button", { type: "button", class: "cite", "aria-label": `Show source ${n}`, onclick: () => onCite(n) }, `${n}`),
          );
        } else {
          node.append(`[${n}]`);
        }
      }
      last = match.index + match[0].length;
    }
    node.append(p.slice(last));
    container.append(node);
  }
}

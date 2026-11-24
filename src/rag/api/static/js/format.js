// A small Markdown subset for answers, built as DOM nodes (never HTML strings):
//
//   blocks   paragraphs (blank line), "- " / "* " / "• " bullet lists, "1. " / "1) " numbered lists,
//            "#".."####" headings (shown as a bold line), display math on its own: \[ ... \] or $$ ... $$
//   inline   **bold**, `code`, math \( ... \) / \[ ... \] / $$ ... $$, citations [n] / [1, 3] / [2-4]
//
// Single $...$ is NOT math: answers about policies are full of prices ("$25 discount").
// Anything else (tables, links, images, HTML) stays as the model wrote it, as plain text.

import { el } from "./dom.js";

const INLINE = new RegExp(
  [
    "`([^`\\n]+)`", // 1 code
    "\\\\\\((.+?)\\\\\\)", // 2 \( math \)
    "\\\\\\[(.+?)\\\\\\]", // 3 \[ math \]
    "\\$\\$(.+?)\\$\\$", // 4 $$ math $$
    "\\*\\*(.+?)\\*\\*", // 5 **bold**
    "\\[(\\d+(?:\\s*[,\\u2013-]\\s*\\d+)*)\\]", // 6 [n] citation
  ].join("|"),
  "gs",
);

export function citationNumbers(body) {
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

// options: { cite(n) -> Node | null  (null: leave "[n]" as text), math(tex, display) -> Node }
export function inline(text, options) {
  const nodes = [];
  let last = 0;
  let lastCite = null; // "[1][1][3]" -> 1 3: a citation repeated right after itself is dropped
  for (const m of text.matchAll(INLINE)) {
    const between = text.slice(last, m.index);
    if (between.trim()) lastCite = null;
    nodes.push(between);
    if (m[6] !== undefined) {
      for (const n of citationNumbers(m[6])) {
        if (n === lastCite) continue;
        nodes.push(options.cite(n) ?? `[${n}]`);
        lastCite = n;
      }
      last = m.index + m[0].length;
      continue;
    }
    lastCite = null;
    if (m[1] !== undefined) nodes.push(el("code", {}, m[1]));
    else if (m[2] !== undefined) nodes.push(options.math(m[2], false));
    else if (m[3] !== undefined) nodes.push(options.math(m[3], true));
    else if (m[4] !== undefined) nodes.push(options.math(m[4], true));
    else if (m[5] !== undefined) nodes.push(el("strong", {}, ...inline(m[5], options)));
    last = m.index + m[0].length;
  }
  nodes.push(text.slice(last));
  return nodes.filter((n) => n !== "");
}

const BULLET = /^\s*[-*•]\s+(.*)$/;
const NUMBERED = /^\s*(\d+)[.)]\s+(.*)$/;
const HEADING = /^\s*#{1,4}\s+(.*)$/;
const DISPLAY_OPEN = /^\s*(\\\[|\$\$)\s*(.*)$/;

// text -> a list of block nodes.
export function blocks(text, options) {
  const lines = text.replace(/\r\n?/g, "\n").split("\n");
  const out = [];
  let paragraph = [];
  let list = null; // { node, ordered }

  const flushParagraph = () => {
    if (paragraph.length) out.push(el("p", {}, ...inline(paragraph.join("\n"), options)));
    paragraph = [];
  };
  const flushList = () => {
    if (list) out.push(list.node);
    list = null;
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    let m;
    if (!line.trim()) {
      flushParagraph();
      flushList();
    } else if ((m = DISPLAY_OPEN.exec(line)) && !line.includes(m[1] === "$$" ? "$$" : "\\]", line.indexOf(m[1]) + 2)) {
      // display math spanning lines: collect until the closing delimiter
      flushParagraph();
      flushList();
      const close = m[1] === "$$" ? "$$" : "\\]";
      const tex = [m[2]];
      while (++i < lines.length && !lines[i].includes(close)) tex.push(lines[i]);
      if (i < lines.length) tex.push(lines[i].slice(0, lines[i].indexOf(close)));
      out.push(el("div", { class: "math-block" }, options.math(tex.join("\n").trim(), true)));
    } else if ((m = HEADING.exec(line))) {
      flushParagraph();
      flushList();
      out.push(el("p", { class: "heading" }, el("strong", {}, ...inline(m[1], options))));
    } else if ((m = BULLET.exec(line)) || (m = NUMBERED.exec(line))) {
      flushParagraph();
      const ordered = m.length === 3;
      const content = ordered ? m[2] : m[1];
      if (!list || list.ordered !== ordered) {
        flushList();
        list = { node: el(ordered ? "ol" : "ul", ordered && m[1] !== "1" ? { start: m[1] } : {}), ordered };
      }
      list.node.append(el("li", {}, ...inline(content, options)));
    } else if (list && /^\s{2,}\S/.test(line)) {
      // an indented continuation of the last list item
      list.node.lastElementChild.append(" ", ...inline(line.trim(), options));
    } else {
      flushList();
      paragraph.push(line);
    }
  }
  flushParagraph();
  flushList();
  return out;
}

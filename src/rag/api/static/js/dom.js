// Building DOM without innerHTML. Answers, document text and log records can contain anything,
// including text written to look like markup or instructions (see the injection-test collection),
// so every string goes in as a text node.

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  append(node, children);
  return node;
}

export function append(node, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function clear(node) {
  node.replaceChildren();
  return node;
}

export const $ = (id) => document.getElementById(id);

export function seconds(ms) {
  if (ms === null || ms === undefined) return "–";
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms.toFixed(ms < 10 ? 1 : 0)} ms`;
}

export function pages(start, end) {
  if (!start) return "";
  return end && end !== start ? `pp. ${start}–${end}` : `p. ${start}`;
}

export const store = {
  get(key) {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(key, value);
    } catch {
      /* private mode: fine */
    }
  },
};

// "2026-11-24T11:02:09.192Z" -> local "24 Nov, 12:02:09" (the log's time stamps are UTC).
export function localTime(ts) {
  if (!ts) return "–";
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return ts;
  return d.toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

// An eval run's "20261124T080146Z" -> "2026-11-24 08:01 UTC".
export function runStamp(stamp) {
  const m = /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/.exec(stamp || "");
  return m ? `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]} UTC` : stamp || "–";
}

// "22/40" for a hit count, "0.675" for MRR, "–" when the run didn't record it.
export const ratio = (value, n) => (value === null || value === undefined ? "–" : `${value}/${n}`);
export const decimal = (value, digits = 3) => (value === null || value === undefined ? "–" : Number(value).toFixed(digits));

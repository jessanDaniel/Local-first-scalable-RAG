// The Documents tab when no collection is picked: every collection in one table, with its size,
// parse doubts, folders, last sync, and the latest retrieval eval of a set that covers it.

import { getCollections, getEvalRuns } from "./api.js";
import { $, clear, el, ratio, runStamp } from "./dom.js";

// "2026-11-24 07:41:42" (SQLite, UTC) -> local date and time.
function sqliteTime(value) {
  if (!value) return "–";
  const d = new Date(`${value.replace(" ", "T")}Z`);
  return Number.isNaN(d.getTime())
    ? value
    : d.toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

// collection -> the newest run of each set whose questions are in it (runs come newest first).
function latestEvals(runs) {
  const out = new Map();
  for (const run of runs) {
    for (const collection of run.collections || []) {
      const list = out.get(collection) || [];
      if (!list.some((r) => r.set === run.set)) list.push(run);
      out.set(collection, list);
    }
  }
  return out;
}

let calls = 0; // a newer render replaces one still loading

export async function renderOverview(onPick) {
  const box = clear($("overview-body"));
  const call = ++calls;
  let collections;
  let runs = [];
  try {
    [collections, runs] = await Promise.all([getCollections(), getEvalRuns().catch(() => [])]);
  } catch (error) {
    if (call === calls) box.append(el("p", { class: "notice" }, `Couldn't load the collections: ${error.message}`));
    return;
  }
  if (call !== calls) return;
  if (!collections.length) {
    box.append(el("p", { class: "muted" }, "No collections yet. Make one above, then add a folder or upload files."));
    return;
  }
  const evals = latestEvals(runs);
  const rows = collections.map((c) => {
    const doubts = [c.n_suspect ? `${c.n_suspect} suspect` : "", c.n_failed ? `${c.n_failed} failed` : ""].filter(Boolean).join(", ");
    const folders = c.n_folders
      ? `${c.n_folders}${c.n_missing_folders ? ` (${c.n_missing_folders} missing)` : ""}`
      : "–";
    const runList = evals.get(c.collection_id) || [];
    return el(
      "tr",
      {},
      el(
        "td",
        {},
        el("button", { type: "button", class: "link-button", onclick: () => onPick(c.collection_id) }, c.collection_id),
        c.description ? el("div", { class: "muted small" }, c.description) : null,
      ),
      el("td", { class: "num" }, c.n_documents),
      el("td", { class: "num" }, c.n_chunks),
      el(
        "td",
        { class: doubts ? "warn" : "muted", title: "suspect: parsed, but a quality check raised a doubt (open the collection to see which)" },
        doubts || "none",
      ),
      el("td", { class: c.n_missing_folders ? "bad" : "" }, folders),
      el("td", {}, c.n_folders ? sqliteTime(c.last_synced) : "–"),
      el(
        "td",
        { class: "section" },
        runList.length
          ? runList.map((r) =>
              el(
                "div",
                {},
                `${r.set}: hit@5 ${ratio(r["hit@5"], r.n)}`,
                el("span", { class: "muted small" }, ` · ${r.retriever}${r.rerank !== "off" ? ` + ${r.rerank}` : ""}, ${runStamp(r.created)}`),
              ),
            )
          : el("span", { class: "muted" }, "no eval set"),
      ),
    );
  });
  box.append(
    el(
      "div",
      { class: "table-wrap" },
      el(
        "table",
        {},
        el(
          "thead",
          {},
          el(
            "tr",
            {},
            el("th", {}, "collection"),
            el("th", { class: "num" }, "documents"),
            el("th", { class: "num" }, "chunks"),
            el("th", {}, "parse doubts"),
            el("th", {}, "folders"),
            el("th", {}, "last synced"),
            el("th", {}, "latest retrieval eval"),
          ),
        ),
        el("tbody", {}, rows),
      ),
    ),
  );
}

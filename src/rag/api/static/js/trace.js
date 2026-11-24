// "How this answer was made": the step-20 log record of the request, drawn as a waterfall of
// stages, a few key facts, and every retrieved candidate with its rank in each list.

import { clear, el, seconds } from "./dom.js";

// Top-level stages in the order they run, each with its sub-stages (which run inside it, in order).
const STAGES = [
  { key: "llm_check_ms", label: "check the LLM is up", kind: "other" },
  {
    key: "retrieve_ms",
    label: "retrieve",
    kind: "retrieve",
    subs: [
      ["dense_embed_ms", "embed the question"],
      ["dense_search_ms", "dense search (FAISS)"],
      ["bm25_search_ms", "BM25 search"],
      ["fuse_ms", "fuse (RRF)"],
      ["rerank_ms", "rerank (cross-encoder)"],
      ["fetch_ms", "fetch chunks"],
    ],
  },
  {
    key: "search_ms",
    label: "search",
    kind: "retrieve",
    subs: [
      ["dense_embed_ms", "embed the query"],
      ["dense_search_ms", "dense search (FAISS)"],
      ["bm25_search_ms", "BM25 search"],
      ["fuse_ms", "fuse (RRF)"],
      ["rerank_ms", "rerank (cross-encoder)"],
      ["fetch_ms", "fetch chunks"],
    ],
  },
  { key: "assemble_ms", label: "build the prompt", kind: "other" },
  {
    key: "generate_ms",
    label: "generate (LLM)",
    kind: "llm",
    subs: [
      ["llm_load_ms", "load the model"],
      ["llm_prompt_ms", "read the prompt"],
      ["llm_write_ms", "write the answer"],
    ],
  },
  { key: "finish_ms", label: "guardrails + citation check", kind: "other" },
];

function stageRows(record) {
  const t = record.timings_ms;
  const total = t.total_ms || 1;
  const rows = [];
  let at = 0;
  for (const stage of STAGES) {
    if (t[stage.key] === undefined) continue;
    rows.push({ label: stage.label, ms: t[stage.key], start: at, kind: stage.kind });
    let subAt = at;
    for (const [key, label] of stage.subs || []) {
      if (t[key] === undefined) continue;
      rows.push({ label, ms: t[key], start: subAt, kind: stage.kind, sub: true });
      subAt += t[key];
    }
    at += t[stage.key];
  }
  if (t.other_ms > 0.5) rows.push({ label: "other (queues, network)", ms: t.other_ms, start: at, kind: "other" });
  return { rows, total };
}

function waterfall(record) {
  const { rows, total } = stageRows(record);
  const grid = el("div", { class: "stages", role: "table", "aria-label": "Time per stage" });
  for (const row of rows) {
    const bar = el("span", { class: `bar ${row.kind}` });
    bar.style.left = `${Math.min(100, (row.start / total) * 100)}%`;
    bar.style.width = `${Math.max(0.3, (row.ms / total) * 100)}%`;
    grid.append(
      el("span", { class: row.sub ? "sub" : "" }, row.label),
      el("span", { class: "bar-track" }, bar),
      el("span", { class: "ms" }, seconds(row.ms)),
    );
  }
  grid.append(el("strong", {}, "total"), el("span"), el("strong", { class: "ms" }, seconds(total)));
  return grid;
}

function facts(record) {
  const s = record.settings;
  const llm = record.llm;
  const t = record.timings_ms;
  const items = [
    ["request", record.request_id],
    ["retrieval", `${s.mode}${s.rerank && s.rerank !== "off" ? ` + rerank ${s.rerank}` : ""}, k ${s.k}`],
    ["collections", s.collections?.length ? s.collections.join(", ") : "all"],
  ];
  if (s.embedding_model) items.push(["embedder", `${s.embedding_model} (${s.embedding_device})`]);
  if (s.rerank_model) items.push(["reranker", `${s.rerank_model} (${s.rerank_device}), top ${s.rerank_candidates}`]);
  if (s.index_build) items.push(["index", s.index_build]);
  if (llm) {
    items.push(["model", `${llm.provider} ${llm.model}, prompt ${s.prompt}`]);
    if (llm.prompt_tokens_counted) {
      items.push(["prompt", `${llm.prompt_tokens_counted} tokens counted, ${llm.prompt_tokens ?? "?"} reported`]);
    }
    if (t.llm_prompt_ms && llm.prompt_tokens) {
      items.push(["prompt reading", `${(t.llm_prompt_ms / llm.prompt_tokens).toFixed(2)} ms per token`]);
    }
    if (t.llm_write_ms && llm.answer_tokens) {
      items.push(["writing", `${llm.answer_tokens} tokens, ${(t.llm_write_ms / llm.answer_tokens).toFixed(1)} ms per token`]);
    }
    if (t.request_first_token_ms) items.push(["first token", `after ${seconds(t.request_first_token_ms)}`]);
  }
  if (s.sanitize_sources !== undefined) {
    items.push(["sanitising", `${s.sanitize_sources ? "on" : "off"}${record.removed_instructions ? `, ${record.removed_instructions} sentences removed` : ""}`]);
  }
  if (record.fallback) items.push(["fallback", record.fallback]);
  if (record.error) items.push([record.status, record.error]);
  return el("dl", { class: "facts" }, items.map(([k, v]) => [el("dt", {}, k), el("dd", {}, v)]));
}

function candidates(record, titles) {
  const rows = record.candidates || [];
  const lists = ["dense", "bm25", "fused", "rerank", "final"].filter((name) =>
    rows.some((c) => c.ranks && name in c.ranks),
  );
  const asked = record.route === "ask";
  const head = el(
    "tr",
    {},
    el("th", { class: "num" }, "#"),
    el("th", {}, "document"),
    el("th", {}, "section"),
    lists.map((name) => el("th", { class: "num" }, name)),
    asked ? el("th", {}, "used as") : null,
  );
  const body = rows.map((c) => {
    const used = !asked ? null : c.source ? `source [${c.source}]` : c.dropped ? "dropped (token budget)" : "not used";
    return el(
      "tr",
      { class: asked && !c.source ? "unused" : "" },
      el("td", { class: "num" }, c.rank),
      el("td", {}, titles.get(c.doc_id) || c.doc_id),
      el("td", { class: "section" }, c.section || ""),
      lists.map((name) => el("td", { class: "num" }, c.ranks?.[name] ?? "–")),
      asked ? el("td", {}, used) : null,
    );
  });
  return el("div", { class: "table-wrap" }, el("table", {}, el("thead", {}, head), el("tbody", {}, body)));
}

export function renderTrace(container, record, titles = new Map()) {
  clear(container);
  container.append(
    el("h3", {}, "Time per stage"),
    waterfall(record),
    el("h3", {}, "Settings and tokens"),
    facts(record),
  );
  if (record.candidates?.length) {
    container.append(
      el("h3", {}, `Retrieved candidates (${record.candidates.length}) — rank in each list`),
      candidates(record, titles),
    );
  }
}

export function traceSummary(record) {
  const t = record.timings_ms;
  const parts = [seconds(t.total_ms)];
  if (t.rerank_ms) parts.push(`rerank ${seconds(t.rerank_ms)}`);
  if (t.llm_prompt_ms) parts.push(`prompt reading ${seconds(t.llm_prompt_ms)}`);
  return `· ${parts.join(" · ")}`;
}

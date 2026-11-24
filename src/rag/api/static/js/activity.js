// The Activity tab: `rag logs stats` as tables, and the recent requests from the step-20 log.
// A row opens the same trace view the Ask tab shows. Log records hold questions and answers
// people typed and models wrote: everything goes in as text nodes.

import { getLog, getLogs, getLogStats } from "./api.js";
import { $, clear, el, localTime, seconds } from "./dom.js";
import { renderTrace, traceSummary } from "./trace.js";

const PAGE = 50;
// statsCall / listCall: a newer load replaces an older one still in flight (opening the tab and
// changing a filter at once must not draw both answers).
const state = { offset: 0, loaded: false, statsCall: 0, listCall: 0 };

// The stages in the order they run; sub-stages are indented (they happen inside the line above).
const STAGES = [
  ["llm_check_ms", "check the LLM is up"],
  ["search_ms", "search"],
  ["retrieve_ms", "retrieve"],
  ["dense_embed_ms", "embed the query", true],
  ["dense_search_ms", "dense search (FAISS)", true],
  ["bm25_search_ms", "BM25 search", true],
  ["fuse_ms", "fuse (RRF)", true],
  ["rerank_ms", "rerank (cross-encoder)", true],
  ["fetch_ms", "fetch chunks", true],
  ["assemble_ms", "build the prompt"],
  ["generate_ms", "generate (LLM)"],
  ["llm_load_ms", "load the model", true],
  ["llm_prompt_ms", "read the prompt", true],
  ["first_token_ms", "first token (Ollama)", true],
  ["llm_write_ms", "write the answer", true],
  ["request_first_token_ms", "first token (request)"],
  ["finish_ms", "guardrails + citations"],
  ["other_ms", "other (queues, network)"],
  ["total_ms", "total"],
];

const RETRIEVAL = new Set(["search_ms", "retrieve_ms", "dense_embed_ms", "dense_search_ms", "bm25_search_ms", "fuse_ms", "rerank_ms", "fetch_ms"]);
const LLM = new Set(["generate_ms", "llm_load_ms", "llm_prompt_ms", "first_token_ms", "llm_write_ms", "request_first_token_ms"]);
const kindOf = (key) => (key === "total_ms" ? "total" : RETRIEVAL.has(key) ? "retrieve" : LLM.has(key) ? "llm" : "other");

// --- stats ---------------------------------------------------------------------------------------

function routeStats(route, s) {
  const counts = [
    `${s.requests} requests`,
    `${s.ok} ok`,
    `${s.errors} errors`,
    `${s.cancelled} cancelled`,
    `${s.fallbacks} reranker fallbacks`,
  ];
  if (route === "ask") counts.push(`${s.refusals} refusals`, `${s.flagged} flagged`);
  const scale = s.stages_ms.total_ms?.p95 || Math.max(1, ...Object.values(s.stages_ms).map((v) => v.p95));
  const known = new Set(STAGES.map(([key]) => key));
  const rows = [...STAGES, ...Object.keys(s.stages_ms).filter((k) => !known.has(k)).map((k) => [k, k.replace(/_ms$/, "")])]
    .filter(([key]) => s.stages_ms[key])
    .map(([key, label, sub]) => {
      const v = s.stages_ms[key];
      const spread = el("span", { class: "bar spread" });
      spread.style.width = `${Math.min(100, (v.p95 / scale) * 100)}%`;
      const median = el("span", { class: `bar ${kindOf(key)}` });
      median.style.width = `${Math.max(0.3, Math.min(100, (v.p50 / scale) * 100))}%`;
      return el(
        "tr",
        { class: key === "total_ms" ? "total-row" : "" },
        el("td", { class: sub ? "sub" : "", title: key }, label),
        el("td", { class: "num" }, v.n),
        el("td", { class: "num" }, seconds(v.p50)),
        el("td", { class: "num" }, seconds(v.p95)),
        el("td", { class: "num" }, seconds(v.max)),
        el("td", { class: "bar-cell" }, el("span", { class: "bar-track" }, spread, median)),
      );
    });
  return el(
    "div",
    { class: "route-stats" },
    el("h3", {}, `/${route}`, " ", el("span", { class: "muted" }, counts.join(" · "))),
    s.prompt_reading_ms_per_token_p50 !== undefined
      ? el("p", { class: "muted" }, `Ollama prompt reading: median ${s.prompt_reading_ms_per_token_p50} ms per token`)
      : null,
    el(
      "div",
      { class: "table-wrap" },
      el(
        "table",
        { class: "stats" },
        el(
          "thead",
          {},
          el(
            "tr",
            {},
            el("th", {}, "stage"),
            el("th", { class: "num" }, "n"),
            el("th", { class: "num" }, "p50"),
            el("th", { class: "num" }, "p95"),
            el("th", { class: "num" }, "max"),
            el("th", {}, "p50 (dark) and p95 (light), against the p95 total"),
          ),
        ),
        el("tbody", {}, rows),
      ),
    ),
  );
}

async function loadStats() {
  const call = ++state.statsCall;
  const body = clear($("stats-body"));
  const meta = $("stats-meta");
  meta.textContent = "reading the log…";
  let result;
  try {
    result = await getLogStats({ last: $("stats-window").value, source: $("stats-source").value });
  } catch (error) {
    if (call === state.statsCall) meta.textContent = `Couldn't read the log: ${error.message}`;
    return;
  }
  if (call !== state.statsCall) return;
  if (!result.n_records) {
    meta.textContent = "No requests logged yet.";
    return;
  }
  meta.textContent =
    `${result.n_records} requests from ${localTime(result.oldest)} to ${localTime(result.newest)}. ` +
    "Successful requests only in the stage times.";
  for (const [route, s] of Object.entries(result.routes)) body.append(routeStats(route, s));
}

// --- recent requests -----------------------------------------------------------------------------

function outcome(r) {
  if (r.status !== "ok") return el("span", { class: "bad", title: r.error || "" }, r.status);
  if (!r.answer) return el("span", { class: "muted" }, `${r.n_candidates} hits${r.fallback ? " · fallback" : ""}`);
  const parts = [r.answer.refusal ? "refused" : `cites ${r.answer.cited}`];
  if (r.answer.flags.length) parts.push(`${r.answer.flags.length} flag${r.answer.flags.length > 1 ? "s" : ""}`);
  return el(
    "span",
    { class: r.answer.flags.length ? "warn" : r.answer.refusal ? "muted" : "", title: r.answer.flags.join(", ") },
    parts.join(" · "),
  );
}

function requestRow(r) {
  const retrieval = [r.mode, r.rerank && r.rerank !== "off" ? `+ ${r.rerank}` : ""].filter(Boolean).join(" ");
  const collections = r.collections?.length ? r.collections.join(", ") : "all";
  const row = el(
    "tr",
    { class: "clickable", tabindex: "0", dataset: { id: r.request_id } },
    el("td", {}, localTime(r.ts)),
    el("td", {}, r.route),
    el("td", {}, r.source),
    el("td", { class: "query-cell" }, r.query ?? el("span", { class: "muted" }, "(text not logged)")),
    el("td", { title: `collections: ${collections}` }, retrieval),
    el("td", { class: "num" }, seconds(r.total_ms)),
    el("td", {}, outcome(r)),
  );
  const open = () => showRequest(r.request_id, row);
  row.addEventListener("click", open);
  row.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      open();
    }
  });
  return row;
}

function filters() {
  return {
    route: $("logs-route").value,
    status: $("logs-status").value,
    source: $("logs-source").value,
    flagged: $("logs-flagged").checked,
    limit: PAGE,
  };
}

async function loadRequests(more = false) {
  const call = ++state.listCall;
  if (!more) {
    state.offset = 0;
    clear($("log-rows"));
    $("activity-trace").hidden = true;
  }
  const meta = $("logs-meta");
  meta.textContent = "loading…";
  let result;
  try {
    result = await getLogs({ ...filters(), offset: state.offset });
  } catch (error) {
    if (call === state.listCall) meta.textContent = `Couldn't read the log: ${error.message}`;
    return;
  }
  if (call !== state.listCall) return;
  for (const r of result.records) $("log-rows").append(requestRow(r));
  state.offset += result.records.length;
  const shown = $("log-rows").children.length;
  meta.textContent = shown
    ? `${shown} shown, newest first${result.log_text ? "" : " · question and answer text is not logged (LOG_TEXT=false)"}`
    : "No logged requests match.";
  $("logs-more").hidden = !result.more;
}

async function showRequest(requestId, row) {
  for (const other of document.querySelectorAll("#log-rows tr.selected")) other.classList.remove("selected");
  row.classList.add("selected");
  const panel = $("activity-trace");
  const body = clear($("activity-trace-body"));
  $("activity-trace-title").textContent = `Request ${requestId}`;
  panel.hidden = false;
  try {
    const record = await getLog(requestId);
    $("activity-trace-sub").textContent = `${record.route} · ${localTime(record.ts)} ${traceSummary(record)}`;
    if (record.query) body.append(el("p", { class: "trace-query" }, record.query));
    if (record.answer?.text) {
      body.append(el("details", { class: "raw" }, el("summary", {}, "The answer"), el("pre", {}, record.answer.text)));
    }
    if (record.answer?.flags?.length) {
      body.append(el("ul", { class: "notes" }, record.answer.flags.map((f) => el("li", {}, `${f.kind}${f.detail ? `: ${f.detail}` : ""}`))));
    }
    const trace = el("div");
    renderTrace(trace, record);
    body.append(trace);
  } catch (error) {
    body.append(el("p", { class: "notice" }, `Couldn't load the record: ${error.message}`));
  }
  panel.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

// --- wiring --------------------------------------------------------------------------------------

export function showActivity() {
  if (state.loaded) return;
  state.loaded = true;
  loadStats();
  loadRequests();
}

export function setupActivity() {
  for (const id of ["stats-window", "stats-source"]) $(id).addEventListener("change", loadStats);
  for (const id of ["logs-route", "logs-status", "logs-source", "logs-flagged"]) {
    $(id).addEventListener("change", () => loadRequests());
  }
  $("activity-refresh").addEventListener("click", () => {
    loadStats();
    loadRequests();
  });
  $("logs-more").addEventListener("click", () => loadRequests(true));
  $("activity-trace-close").addEventListener("click", () => {
    $("activity-trace").hidden = true;
    for (const other of document.querySelectorAll("#log-rows tr.selected")) other.classList.remove("selected");
  });
}

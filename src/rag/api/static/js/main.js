// The page: status, collections, the Ask and Search tabs.

import { ApiError, askStream, getCollections, getHealth, getLog, search } from "./api.js";
import { renderFinal, renderStreaming } from "./answer.js";
import { $, clear, el, pages, seconds, store } from "./dom.js";
import { setupDocs, showDocs } from "./docs.js";
import { setupPassageDialog, showPassage } from "./passage.js";
import { renderTrace, traceSummary } from "./trace.js";

const state = { health: null, llmProblem: null, asking: null, defaultsApplied: false };

// --- status -------------------------------------------------------------------------------------

async function refreshHealth() {
  let health;
  try {
    health = await getHealth();
  } catch (error) {
    health = { status: "unavailable", problems: [`server not reachable: ${error.message}`] };
  }
  state.health = health;
  const button = $("status");
  button.dataset.state = health.status;
  $("status-text").textContent = { ok: "ready", degraded: "degraded", unavailable: "unavailable" }[health.status];
  const panel = clear($("status-panel"));
  const report = health.report || {};
  const llm = report.llm || {};
  panel.append(
    el("strong", {}, `Status: ${health.status}`),
    health.problems?.length ? el("ul", {}, health.problems.map((p) => el("li", {}, p))) : el("p", {}, "Everything is running."),
    llm.model ? el("p", { class: "muted" }, `LLM: ${llm.provider || ""} ${llm.model}`) : null,
    llm.provider === "ollama" && llm.state && llm.state !== "running"
      ? el("p", {}, "To start it: brew services start ollama — the page notices within 20 seconds.")
      : null,
  );
  state.llmProblem = llm.state && llm.state !== "running" ? llm.state : null; // e.g. "LLM qwen2.5:7b-instruct: not running"
  if (health.status === "unavailable") state.llmProblem = health.problems?.[0] || "the server can't answer";
  updateAskAvailability();
  applyDefaults(health);
  updateRerankCost(health);
}

function applyDefaults(health) {
  const d = health.service?.defaults;
  if (!d || state.defaultsApplied) return;
  state.defaultsApplied = true;
  $("ask-mode").value = d.mode;
  $("ask-rerank").value = d.ask_rerank;
  $("search-mode").value = d.mode;
  $("search-rerank").value = d.search_rerank;
}

function updateRerankCost(health) {
  const recent = health?.requests?.recent || {};
  const stages = recent.ask?.stages_ms?.rerank_ms || recent.search?.stages_ms?.rerank_ms;
  const rerank = $("ask-rerank").value;
  $("rerank-cost").textContent =
    rerank === "off"
      ? "no reranker: fastest, may rank the best passage lower"
      : stages
        ? `reranker adds ~${seconds(stages.p50)} (median of ${stages.n} recent requests)`
        : "the reranker adds about 2 s per question";
}

function updateAskAvailability() {
  const blocked = $("ask-blocked");
  const busy = state.asking !== null;
  if (state.llmProblem) {
    blocked.textContent = `Ask is unavailable: ${state.llmProblem}. Search still works.`;
    blocked.hidden = false;
  } else {
    blocked.hidden = true;
  }
  $("ask-go").disabled = busy || Boolean(state.llmProblem);
}

// --- collections ---------------------------------------------------------------------------------

// (Re)load the collection picker, keeping the selection (or selecting `pick`); called again after a
// collection is created or a sync changed document counts.
async function loadCollections(pick = null) {
  const select = $("collection");
  const keep = pick ?? select.value ?? store.get("rag.collection");
  try {
    const collections = await getCollections();
    clear(select).append(el("option", { value: "" }, "All collections"));
    for (const c of collections) {
      select.append(el("option", { value: c.collection_id }, `${c.collection_id} (${c.n_documents} documents)`));
    }
    const wanted = keep || store.get("rag.collection");
    if (wanted && [...select.options].some((o) => o.value === wanted)) select.value = wanted;
    if (pick) store.set("rag.collection", pick);
  } catch {
    /* the status panel already says why */
  }
}

const selectedCollections = () => ($("collection").value ? [$("collection").value] : null);

// --- tabs ----------------------------------------------------------------------------------------

function setupTabs() {
  const tabs = [
    ["tab-ask", "view-ask"],
    ["tab-search", "view-search"],
    ["tab-docs", "view-docs"],
  ];
  for (const [tabId] of tabs) {
    $(tabId).addEventListener("click", () => {
      for (const [otherTab, otherView] of tabs) {
        const active = otherTab === tabId;
        $(otherTab).setAttribute("aria-selected", String(active));
        $(otherView).hidden = !active;
      }
      if (tabId === "tab-docs") showDocs($("collection").value);
      else (tabId === "tab-ask" ? $("question") : $("query")).focus();
    });
  }
}

// --- ask -----------------------------------------------------------------------------------------

function sourceCard(source, onShow) {
  return el(
    "li",
    { class: "source", id: `source-${source.n}`, dataset: { n: source.n } },
    el(
      "div",
      { class: "source-head" },
      el("span", { class: "source-n" }, `[${source.n}]`),
      el("span", { class: "source-title" }, source.title || source.doc_id),
    ),
    el("div", { class: "source-meta" }, [source.section_path, source.pages !== "p.?" ? source.pages : ""].filter(Boolean).join(" · ")),
    el(
      "div",
      { class: "source-actions" },
      el("button", { type: "button", class: "secondary", onclick: () => onShow(source) }, "Show passage"),
    ),
  );
}

function focusSource(n) {
  const card = $(`source-${n}`);
  if (!card) return;
  for (const other of document.querySelectorAll(".source.focus")) other.classList.remove("focus");
  card.classList.add("focus");
  card.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

// Worth showing the model's own text only if the guardrails changed the words or which sources are
// cited, not when they just moved a citation inside the full stop.
function guardrailsChanged(raw, text) {
  const cited = (s) => [...s.matchAll(/\[(\d+)\]/g)].map((m) => m[1]).sort().join(",");
  const words = (s) => s.replace(/\[\d+(?:\s*[,–-]\s*\d+)*\]/g, "").replace(/[\s.,;:]+/g, " ").trim();
  return cited(raw) !== cited(text) || words(raw) !== words(text);
}

function note(text, kind = "") {
  $("answer-notes").append(el("li", { class: kind }, text));
}

async function ask(event) {
  event.preventDefault();
  const question = $("question").value.trim();
  if (!question || state.asking) return;
  const controller = new AbortController();
  state.asking = controller;
  updateAskAvailability();
  $("ask-stop").hidden = false;

  const answer = $("answer");
  clear(answer);
  clear($("answer-notes"));
  clear($("sources"));
  $("sources-count").textContent = "";
  $("raw-answer").hidden = true;
  $("trace").hidden = true;
  $("ask-result").hidden = false;
  const pill = $("answer-state");
  pill.dataset.kind = "";
  pill.textContent = "retrieving…";

  let sources = [];
  let raw = "";
  let requestId = null;
  const showSource = (s) => showPassage(s.chunk_ids, { title: s.title, page: s.page_start });

  try {
    await askStream(
      { question, collections: selectedCollections(), mode: $("ask-mode").value, rerank: $("ask-rerank").value },
      {
        sources(data) {
          requestId = data.request_id;
          sources = data.sources;
          $("sources-count").textContent = `(${sources.length} given to the model)`;
          for (const s of sources) $("sources").append(sourceCard(s, showSource));
          if (data.retrieval_fallback) note(`Reranker skipped: ${data.retrieval_fallback}`, "info");
          pill.textContent = "the model is reading the sources…";
          answer.classList.add("streaming");
          renderStreaming(answer, "");
        },
        token(data) {
          raw += data.text;
          pill.textContent = "writing…";
          renderStreaming(answer, raw);
        },
        done(data) {
          answer.classList.remove("streaming");
          renderFinal(answer, data.text, sources.length, focusSource);
          pill.textContent = data.refusal ? "the sources don't answer this" : `cites ${data.citations.cited.length} of ${sources.length} sources`;
          pill.dataset.kind = data.refusal ? "refusal" : "";
          for (const n of data.citations.cited) $(`source-${n}`)?.classList.add("cited");
          for (const flag of data.flags) note(`Check: ${flag.detail}${flag.sentence ? ` — “${flag.sentence}”` : ""}`);
          for (const warning of data.warnings) note(warning);
          if (guardrailsChanged(data.raw_text, data.text)) {
            $("raw-text").textContent = data.raw_text;
            $("raw-answer").hidden = false;
          }
        },
        error(data) {
          answer.classList.remove("streaming");
          pill.textContent = "failed";
          pill.dataset.kind = "error";
          note(`The model stopped with an error: ${data.detail}`, "error");
        },
      },
      controller.signal,
    );
  } catch (error) {
    answer.classList.remove("streaming");
    pill.dataset.kind = "error";
    if (error.name === "AbortError") {
      pill.textContent = "stopped";
      note("Stopped. The model was told to stop at its next token.", "info");
    } else {
      pill.textContent = "failed";
      note(error instanceof ApiError ? `${error.status}: ${error.message}` : error.message, "error");
      if (error.status === 503) refreshHealth();
    }
  } finally {
    state.asking = null;
    $("ask-stop").hidden = true;
    updateAskAvailability();
  }
  if (requestId) showTrace(requestId, sources);
}

async function showTrace(requestId, sources) {
  const titles = new Map(sources.map((s) => [s.doc_id, s.title]));
  // The record is written when the request ends; give a cancelled stream a moment to land.
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      const record = await getLog(requestId);
      renderTrace($("trace-body"), record, titles);
      $("trace-summary").textContent = traceSummary(record);
      $("trace").hidden = false;
      refreshHealth(); // recent timings, for the reranker hint
      return;
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 300));
    }
  }
}

// --- search --------------------------------------------------------------------------------------

function hitCard(hit) {
  const ranks = Object.entries(hit.ranks || {})
    .map(([name, rank]) => `${name} ${rank ?? "–"}`)
    .join(" · ");
  const text = el("p", { class: "hit-text clamped" }, hit.text);
  return el(
    "li",
    { class: "hit" },
    el(
      "div",
      { class: "hit-head" },
      el("span", { class: "hit-rank" }, `#${hit.rank}`),
      el("strong", {}, hit.doc_title || hit.doc_id),
      el("span", { class: "muted" }, [hit.section_path, pages(hit.page_start, hit.page_end)].filter(Boolean).join(" · ")),
    ),
    ranks ? el("div", { class: "hit-ranks" }, `score ${hit.score.toFixed(4)} · ${ranks}`) : null,
    text,
    el(
      "div",
      { class: "source-actions" },
      el("button", { type: "button", class: "secondary", onclick: () => text.classList.toggle("clamped") }, "More / less"),
      el(
        "button",
        { type: "button", class: "secondary", onclick: () => showPassage([hit.chunk_id], { title: hit.doc_title, page: hit.page_start }) },
        "Show in section",
      ),
    ),
  );
}

async function runSearch(event) {
  event.preventDefault();
  const query = $("query").value.trim();
  if (!query) return;
  const meta = $("search-meta");
  const hits = clear($("hits"));
  meta.textContent = "searching…";
  try {
    const result = await search({
      query,
      k: parseInt($("search-k").value, 10),
      collections: selectedCollections(),
      mode: $("search-mode").value,
      rerank: $("search-rerank").value,
    });
    const t = result.timings_ms;
    meta.textContent = `${result.hits.length} passages · ${result.mode}${result.rerank !== "off" ? ` + rerank ${result.rerank}` : ""} · ${seconds(t.total_ms)}${t.rerank_ms ? ` (rerank ${seconds(t.rerank_ms)})` : ""}${result.fallback ? ` · ${result.fallback}` : ""}`;
    for (const hit of result.hits) hits.append(hitCard(hit));
  } catch (error) {
    meta.textContent = `Search failed: ${error.message}`;
  }
}

// --- start ---------------------------------------------------------------------------------------

function start() {
  setupTabs();
  setupPassageDialog();
  $("status").addEventListener("click", () => {
    const panel = $("status-panel");
    panel.hidden = !panel.hidden;
    $("status").setAttribute("aria-expanded", String(!panel.hidden));
  });
  $("ask-form").addEventListener("submit", ask);
  $("ask-stop").addEventListener("click", () => state.asking?.abort());
  $("ask-rerank").addEventListener("change", () => updateRerankCost(state.health));
  $("question").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      $("ask-form").requestSubmit();
    }
  });
  $("search-form").addEventListener("submit", runSearch);
  setupDocs({ onCollectionsChanged: (pick) => loadCollections(pick) });
  $("collection").addEventListener("change", () => {
    store.set("rag.collection", $("collection").value);
    if (!$("view-docs").hidden) showDocs($("collection").value);
  });
  loadCollections();
  refreshHealth();
  setInterval(refreshHealth, 20000);
  $("question").focus();
}

start();

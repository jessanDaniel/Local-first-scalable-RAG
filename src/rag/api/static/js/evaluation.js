// The Evaluation tab: stored retrieval runs, one run's questions, A-vs-B comparison, answer runs,
// and the quality-gate baseline. Read-only: runs are made with `rag eval run` / `rag eval answers`,
// so every number here comes from a run with its configuration saved next to it.

import { compareRuns, getAnswerRuns, getEvalRun, getEvalRuns, getGate } from "./api.js";
import { $, clear, decimal, el, ratio, runStamp, seconds, store } from "./dom.js";

const state = { runs: [], a: null, b: null, loaded: false, detailCall: 0, compareCall: 0, answersCall: 0, gateCall: 0 };

const retrievalLabel = (r) => `${r.retriever}${r.rerank && r.rerank !== "off" ? ` + ${r.rerank}` : ""}`;

function rankCell(rank, maxK = 20) {
  const kind = rank === null || rank === undefined ? "miss" : rank === 1 ? "top1" : rank <= 5 ? "top5" : "low";
  return el("td", { class: "num" }, el("span", { class: `rank ${kind}` }, rank ?? `>${maxK}`));
}

function deltaCell(delta, digits = 0, unit = "") {
  if (delta === null || delta === undefined) return el("td", { class: "num" }, "–");
  const text = `${delta > 0 ? "+" : ""}${digits ? delta.toFixed(digits) : delta}${unit}`;
  return el("td", { class: `num ${delta > 0 ? "good" : delta < 0 ? "bad" : "muted"}` }, text);
}

const table = (head, rows, attrs = {}) =>
  el(
    "div",
    { class: "table-wrap" },
    el("table", attrs, el("thead", {}, el("tr", {}, head)), el("tbody", {}, rows)),
  );
const th = (text, num = false) => el("th", { class: num ? "num" : "" }, text);

// --- retrieval runs ------------------------------------------------------------------------------

async function loadRuns() {
  const meta = $("runs-meta");
  meta.textContent = "loading…";
  try {
    state.runs = await getEvalRuns();
  } catch (error) {
    meta.textContent = `Couldn't read the runs: ${error.message}`;
    return;
  }
  const sets = [...new Set(state.runs.map((r) => r.set))].sort();
  const select = clear($("runs-set"));
  for (const s of sets) select.append(el("option", { value: s }, s));
  const remembered = store.get("rag.evalSet");
  select.value = sets.includes(remembered) ? remembered : state.runs[0]?.set || "";
  renderRuns();
}

function renderRuns() {
  const set = $("runs-set").value;
  const runs = state.runs.filter((r) => r.set === set);
  state.a = state.b = null;
  updateCompareButton();
  $("runs-meta").textContent = runs.length
    ? `${runs.length} runs of the ${set} set, newest first. Pick A and B to compare, or a run's name to see its questions.`
    : "No runs yet: rag eval run --set NAME saves one.";
  const rows = runs.map((r) =>
    el(
      "tr",
      {},
      el("td", {}, el("input", { type: "radio", name: "run-a", "aria-label": `A: ${r.run_id}`, onchange: () => pick("a", r.run_id) })),
      el("td", {}, el("input", { type: "radio", name: "run-b", "aria-label": `B: ${r.run_id}`, onchange: () => pick("b", r.run_id) })),
      el(
        "td",
        {},
        el("button", { type: "button", class: "link-button", onclick: () => showRun(r.run_id) }, r.label || r.run_id),
        el("div", { class: "muted small" }, runStamp(r.created)),
      ),
      el("td", {}, retrievalLabel(r)),
      el("td", { class: "num" }, ratio(r["hit@1"], r.n)),
      el("td", { class: "num" }, ratio(r["hit@5"], r.n)),
      el("td", { class: "num" }, ratio(r["hit@10"], r.n)),
      el("td", { class: "num" }, decimal(r.mrr)),
      el("td", { class: "num" }, r.latency_ms ? seconds(r.latency_ms.p50) : "–"),
      el("td", { class: "num" }, r.n_chunks ?? "–"),
      el("td", { class: "muted" }, r.git_commit || ""),
    ),
  );
  const body = clear($("runs-table"));
  body.append(
    table(
      [th("A"), th("B"), th("run"), th("retrieval"), th("hit@1", true), th("hit@5", true), th("hit@10", true), th("MRR", true), th("p50", true), th("chunks", true), th("commit")],
      rows,
    ),
  );
}

function pick(which, runId) {
  state[which] = runId;
  updateCompareButton();
}

function updateCompareButton() {
  const button = $("runs-compare");
  button.disabled = !(state.a && state.b && state.a !== state.b);
  button.title = button.disabled ? "Pick two different runs, one as A and one as B" : "";
}

// --- one run -------------------------------------------------------------------------------------

async function showRun(runId) {
  const panel = $("eval-detail");
  const body = clear($("eval-detail-body"));
  $("eval-detail-title").textContent = runId;
  panel.hidden = false;
  const call = ++state.detailCall;
  let run;
  try {
    run = await getEvalRun(runId);
  } catch (error) {
    if (call === state.detailCall) body.append(el("p", { class: "notice" }, error.message));
    return;
  }
  if (call !== state.detailCall) return; // another run was picked meanwhile
  const c = run.config;
  const facts = [
    ["retrieval", retrievalLabel(run)],
    ["created", `${runStamp(run.created)}, commit ${run.git_commit || "?"}`],
    ["index", `${c.index_build || "?"} (${c.index_type || "?"}, ${c.n_chunks ?? "?"} chunks)`],
    ["chunker", c.chunker_version],
    ["embedder", c.embedding_model],
    ["collection filter", c.collection_filter ? "on (each question searches its own collection)" : "off"],
  ];
  if (run.rerank !== "off") facts.push(["reranker", `${c.rerank_model} on ${c.rerank_device}`]);
  if (run.latency_ms) {
    facts.push(["search time", `p50 ${seconds(run.latency_ms.p50)}, p95 ${seconds(run.latency_ms.p95)}, max ${seconds(run.latency_ms.max)}`]);
  }
  if (run.fallbacks) facts.push(["fallbacks", `reranker fell back on ${run.fallbacks} questions`]);
  body.append(el("dl", { class: "facts" }, facts.filter(([, v]) => v).map(([k, v]) => [el("dt", {}, k), el("dd", {}, v)])));

  const byType = Object.entries(run.summary.by_type).map(([name, b]) =>
    el(
      "tr",
      {},
      el("td", {}, name),
      el("td", { class: "num" }, b.n),
      el("td", { class: "num" }, ratio(b["hit@1"], b.n)),
      el("td", { class: "num" }, ratio(b["hit@5"], b.n)),
      el("td", { class: "num" }, ratio(b["hit@10"], b.n)),
      el("td", { class: "num" }, decimal(b.mrr)),
    ),
  );
  const o = run.summary.overall;
  byType.push(
    el(
      "tr",
      { class: "total-row" },
      el("td", {}, "all answerable"),
      el("td", { class: "num" }, o.n),
      el("td", { class: "num" }, ratio(o["hit@1"], o.n)),
      el("td", { class: "num" }, ratio(o["hit@5"], o.n)),
      el("td", { class: "num" }, ratio(o["hit@10"], o.n)),
      el("td", { class: "num" }, decimal(o.mrr)),
    ),
  );
  body.append(
    el("h3", {}, "By question type"),
    table([th("type"), th("n", true), th("hit@1", true), th("hit@5", true), th("hit@10", true), th("MRR", true)], byType),
  );

  const only = el("input", { type: "checkbox", id: "detail-misses" });
  const rowsBox = el("div");
  const answerable = run.questions.filter((q) => q.answerable);
  const draw = () => {
    const shown = answerable
      .filter((q) => !only.checked || q.first_rank === null || q.first_rank > 5)
      .sort((x, y) => (x.first_rank ?? 99) - (y.first_rank ?? 99));
    clear(rowsBox).append(
      table(
        [th("first gold rank", true), th("question"), th("type"), th("word overlap", true), th("top-1 came from")],
        shown.map((q) =>
          el(
            "tr",
            {},
            rankCell(q.first_rank),
            el("td", { class: "section" }, el("strong", {}, q.id), q.question ? el("div", { class: "muted small" }, q.question) : null),
            el("td", {}, q.type),
            el("td", { class: "num" }, decimal(q.overlap, 2)),
            el("td", { class: (q.first_rank ?? 99) > 1 ? "warn" : "muted" }, q.top1_doc || "–"),
          ),
        ),
      ),
    );
  };
  only.addEventListener("change", draw);
  body.append(
    el("div", { class: "panel-head" }, el("h3", {}, `Questions (${answerable.length} answerable)`), el("label", { class: "check" }, only, " only gold outside the top 5")),
    run.question_text_from ? el("p", { class: "muted small" }, `Question wording from ${run.question_text_from}.`) : null,
    rowsBox,
  );
  draw();
  const unanswerable = run.questions.filter((q) => !q.answerable);
  if (unanswerable.length) {
    body.append(
      el("h3", {}, `Unanswerable questions (${unanswerable.length}): what came first anyway`),
      table(
        [th("question"), th("top-1 score", true), th("top-1 came from")],
        unanswerable.map((q) =>
          el(
            "tr",
            {},
            el("td", { class: "section" }, el("strong", {}, q.id), q.question ? el("div", { class: "muted small" }, q.question) : null),
            el("td", { class: "num" }, decimal(q.top1_score, 4)),
            el("td", { class: "muted" }, q.top1_doc || "–"),
          ),
        ),
      ),
    );
  }
  panel.scrollIntoView({ block: "start" });
}

// --- A vs B --------------------------------------------------------------------------------------

async function showCompare() {
  const panel = $("eval-compare");
  const body = clear($("eval-compare-body"));
  panel.hidden = false;
  const call = ++state.compareCall;
  let d;
  try {
    d = await compareRuns(state.a, state.b);
  } catch (error) {
    if (call === state.compareCall) body.append(el("p", { class: "notice" }, error.message));
    return;
  }
  if (call !== state.compareCall) return;
  body.append(el("dl", { class: "facts" }, el("dt", {}, "A"), el("dd", {}, d.a), el("dt", {}, "B"), el("dd", {}, d.b)));
  if (d.config_changes.length) {
    body.append(
      el("h3", {}, "What changed in the configuration"),
      table(
        [th("setting"), th("A"), th("B")],
        d.config_changes.map((c) =>
          el("tr", {}, el("td", {}, c.key), el("td", { class: "section" }, JSON.stringify(c.a)), el("td", { class: "section" }, JSON.stringify(c.b))),
        ),
      ),
    );
  } else {
    body.append(el("p", { class: "muted" }, "Same configuration (only the commit may differ)."));
  }
  const metricRows = d.metrics.map((m) => {
    const hit = m.metric.startsWith("hit@");
    return el(
      "tr",
      {},
      el("td", {}, m.metric),
      el("td", { class: "num" }, hit ? ratio(m.a, d.n.a) : decimal(m.a)),
      el("td", { class: "num" }, hit ? ratio(m.b, d.n.b) : decimal(m.b)),
      deltaCell(m.delta, hit ? 0 : 3),
    );
  });
  for (const l of d.latency_ms) {
    // Slower is worse: show the delta with the sign flipped in colour, not in value.
    const cell = deltaCell(l.delta, 0, " ms");
    cell.className = `num ${l.delta > 0 ? "bad" : l.delta < 0 ? "good" : "muted"}`;
    metricRows.push(el("tr", {}, el("td", {}, `latency ${l.key}`), el("td", { class: "num" }, seconds(l.a)), el("td", { class: "num" }, seconds(l.b)), cell));
  }
  body.append(el("h3", {}, "Metrics"), table([th("metric"), th("A", true), th("B", true), th("B − A", true)], metricRows));
  if (d.flips.length) {
    body.append(
      el("h3", {}, `Questions whose first gold rank moved (${d.flips.length}; + = better in B)`),
      table(
        [th("change", true), th("question"), th("A", true), th("B", true), th("crosses")],
        d.flips.map((f) =>
          el(
            "tr",
            {},
            deltaCell(f.change),
            el("td", { class: "section" }, el("strong", {}, f.id), f.question ? el("div", { class: "muted small" }, f.question) : null),
            rankCell(f.a, d.max_k),
            rankCell(f.b, d.max_k),
            el("td", {}, f.crosses.length ? `top ${f.crosses.join(", ")}` : ""),
          ),
        ),
      ),
    );
  } else {
    body.append(el("p", { class: "muted" }, "No question changed rank."));
  }
  if (d.only_in_a.length || d.only_in_b.length) {
    body.append(
      el("p", { class: "muted small" }, `Only in A: ${d.only_in_a.join(", ") || "none"}. Only in B: ${d.only_in_b.join(", ") || "none"}.`),
    );
  }
  panel.scrollIntoView({ block: "start" });
}

// --- answer runs ---------------------------------------------------------------------------------

async function loadAnswerRuns() {
  const box = clear($("answers-body"));
  const call = ++state.answersCall;
  let runs;
  try {
    runs = await getAnswerRuns();
  } catch (error) {
    if (call === state.answersCall) box.append(el("p", { class: "notice" }, error.message));
    return;
  }
  if (call !== state.answersCall) return;
  if (!runs.length) {
    box.append(el("p", { class: "muted" }, "No answer runs yet: rag eval answers --set NAME makes one (needs the LLM)."));
    return;
  }
  const rows = runs.map((r) => {
    const s = r.summary;
    const a = s.answerable || {};
    const u = s.unanswerable || {};
    const flags = Object.entries(s.flags || {}).map(([k, v]) => `${k} ${v}`).join(", ");
    const inj = s.injections?.attempts ? ` · injections followed ${s.injections.followed}/${s.injections.attempts}` : "";
    return el(
      "tr",
      {},
      el("td", {}, el("strong", {}, r.run_id), el("div", { class: "muted small" }, `${runStamp(r.created)} · ${r.git_commit || ""}`)),
      el("td", {}, `${r.llm || "?"}`, el("div", { class: "muted small" }, `prompt ${r.prompt}, guardrails ${r.guardrails ? "on" : "off"}${r.sanitize_sources ? ", sanitising" : ""}`)),
      el("td", {}, `${r.retriever} + ${r.rerank}`),
      el("td", { class: "num" }, `${s.attempts} (${r.runs}×)`),
      el("td", { class: "num" }, a.key_match ?? "–"),
      el("td", { class: "num" }, a.gold_cited ?? "–"),
      el("td", { class: "num" }, u.of ? `${u.refused}/${u.of}` : "–"),
      el("td", { class: "num" }, u.refused_with_citations ?? "–"),
      el("td", { class: "section small" }, (flags || "none") + inj),
      el("td", { class: "num" }, s.seconds?.total_median !== undefined ? `${s.seconds.total_median} s` : "–"),
    );
  });
  box.append(
    table(
      [th("run"), th("model"), th("retrieval"), th("answers", true), th("key facts", true), th("gold cited", true), th("refused (unanswerable)", true), th("refusals citing", true), th("guardrail flags"), th("median", true)],
      rows,
    ),
    el(
      "p",
      { class: "muted small" },
      "Automatic checks only: “key facts” means the expected answer's key words appear in the answer, not that it is right. Human labels live in eval/labels/.",
    ),
  );
}

// --- quality gate --------------------------------------------------------------------------------

async function loadGate() {
  const box = clear($("gate-body"));
  const call = ++state.gateCall;
  let g;
  try {
    g = await getGate();
  } catch (error) {
    if (call === state.gateCall) box.append(el("p", { class: "notice" }, error.message));
    return;
  }
  if (call !== state.gateCall) return;
  if (!g || g.detail) {
    box.append(el("p", { class: "muted" }, g?.detail || "No baseline."));
    return;
  }
  box.append(
    el(
      "dl",
      { class: "facts" },
      el("dt", {}, "updated"),
      el("dd", {}, `${g.updated} (commit ${g.commit || "?"})`),
      el("dt", {}, "reason"),
      el("dd", {}, g.reason || "–"),
      el("dt", {}, "retrieval"),
      el("dd", {}, g.config ? `${g.config.retriever} + rerank ${g.config.rerank}, ${g.config.n_chunks} chunks` : "–"),
      el("dt", {}, "answers replayed"),
      el("dd", {}, `${g.answers_replayed} stored answers through the guardrails; ${g.answers_flagged} carry a flag (any change fails the gate)`),
    ),
  );
  const rows = Object.entries(g.sets).map(([name, s]) =>
    el(
      "tr",
      {},
      el("td", {}, name),
      el("td", { class: "num" }, s.n),
      el("td", { class: "num" }, ratio(s.metrics?.["hit@1"], s.n)),
      el("td", { class: "num" }, ratio(s.metrics?.["hit@5"], s.n)),
      el("td", { class: "num" }, ratio(s.metrics?.["hit@10"], s.n)),
      el("td", { class: "num" }, decimal(s.metrics?.mrr)),
      el("td", { class: "section small" }, s.misses_top5.length ? s.misses_top5.map((m) => `${m.id} (${m.rank ?? ">20"})`).join(", ") : "none"),
    ),
  );
  box.append(
    el("h3", {}, "What a commit must keep"),
    table([th("set"), th("n", true), th("hit@1", true), th("hit@5", true), th("hit@10", true), th("MRR", true), th("gold outside the top 5 (allowed to stay there)")], rows),
    el(
      "p",
      { class: "muted small" },
      "scripts/check.sh fails on any metric drop, any question leaving the top 5, or any change in guardrail flags. An intended change: rag eval gate --update \"reason\".",
    ),
  );
  if (g.history.length) {
    box.append(
      el("h3", {}, `History (${g.history.length} updates, newest first)`),
      el(
        "ol",
        { class: "history" },
        g.history.map((h) =>
          el(
            "li",
            {},
            el("strong", {}, h.when),
            ` · ${h.commit || "?"}`,
            el("p", {}, h.reason),
            el(
              "p",
              { class: "muted small" },
              Object.entries(h.metrics || {})
                .map(([set, m]) => `${set}: hit@1 ${m["hit@1"]}, hit@5 ${m["hit@5"]}, MRR ${decimal(m.mrr)}`)
                .join(" · "),
            ),
          ),
        ),
      ),
    );
  }
}

// --- wiring --------------------------------------------------------------------------------------

export function showEvaluation() {
  if (state.loaded) return;
  state.loaded = true;
  loadRuns();
  loadAnswerRuns();
  loadGate();
}

export function setupEvaluation() {
  $("runs-set").addEventListener("change", () => {
    store.set("rag.evalSet", $("runs-set").value);
    renderRuns();
  });
  $("runs-compare").addEventListener("click", showCompare);
  $("eval-refresh").addEventListener("click", () => {
    loadRuns();
    loadAnswerRuns();
    loadGate();
  });
  $("eval-detail-close").addEventListener("click", () => ($("eval-detail").hidden = true));
  $("eval-compare-close").addEventListener("click", () => ($("eval-compare").hidden = true));
}

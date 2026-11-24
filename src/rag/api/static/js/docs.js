// The Documents tab: a collection's folders, uploads, documents, and the progress of the sync job
// that every change starts. Everything shown here comes from the API as data: text nodes only.

import {
  addFolder,
  cancelJob,
  createCollection,
  deleteCollection,
  deleteDocument,
  getCollection,
  getJob,
  removeFolder,
  syncCollection,
  uploadFile,
} from "./api.js";
import { $, clear, el } from "./dom.js";

const SUPPORTED = [".pdf", ".docx", ".html", ".htm", ".txt", ".md", ".markdown", ".text"];
const TERMINAL = new Set(["done", "failed", "cancelled"]);
const state = { collection: null, job: null, seen: 0, timer: null, onChanged: () => {}, detail: null };

const suffix = (name) => (name.includes(".") ? name.slice(name.lastIndexOf(".")).toLowerCase() : "");

function message(text, kind = "") {
  const box = $("docs-message");
  box.textContent = text;
  box.className = `notice ${kind}`;
  box.hidden = !text;
}

// --- showing a collection ----------------------------------------------------------------------

export async function showDocs(collection) {
  state.collection = collection || null;
  $("docs-body").hidden = !state.collection;
  $("delete-collection").hidden = !state.collection;
  $("docs-pick").textContent = state.collection
    ? `Managing “${state.collection}”. Pick another collection at the top, or make a new one.`
    : "Pick a collection at the top, or make a new one.";
  if (!state.collection) return;
  let detail;
  try {
    detail = await getCollection(state.collection);
  } catch (error) {
    message(`Couldn't load ${state.collection}: ${error.message}`);
    return;
  }
  state.detail = detail;
  renderFolders(detail.folders);
  renderDocuments(detail.documents);
  const running = detail.jobs.find((j) => !TERMINAL.has(j.status));
  if (running && (!state.job || state.job.id !== running.id)) follow(running);
}

function renderFolders(folders) {
  const list = clear($("folders"));
  if (!folders.length) list.append(el("li", { class: "muted" }, "No folders yet."));
  for (const folder of folders) {
    const synced = folder.last_synced ? `synced ${folder.last_synced} UTC` : "not synced yet";
    list.append(
      el(
        "li",
        { class: folder.exists ? "" : "failed" },
        el("span", {}, folder.path, " ", el("span", { class: "muted" }, folder.exists ? `· ${synced}` : "· not found now (its documents are kept)")),
        el("button", { type: "button", class: "secondary", onclick: () => forgetFolder(folder.path) }, "Remove"),
      ),
    );
  }
}

function renderDocuments(documents) {
  $("doc-count").textContent = `(${documents.length})`;
  const rows = clear($("doc-rows"));
  for (const d of documents) {
    const removable = d.origin !== "folder";
    rows.append(
      el(
        "tr",
        { class: d.status === "failed" ? "failed" : "" },
        el("td", { class: "section" }, d.title || d.doc_id),
        el("td", {}, d.file || d.doc_id),
        el("td", {}, d.origin === "folder" ? "folder" : d.origin === "upload" ? "upload" : d.origin),
        el("td", { class: "num" }, d.n_chunks),
        el("td", { title: d.issues || "" }, d.status),
        el(
          "td",
          {},
          removable
            ? el("button", { type: "button", class: "secondary", onclick: () => forgetDocument(d) }, "Remove")
            : el("span", { class: "muted", title: "Delete or move the file in its folder; the next rescan removes it" }, "—"),
        ),
      ),
    );
  }
}

// --- actions -----------------------------------------------------------------------------------

async function act(promise) {
  message("");
  try {
    const result = await promise;
    if (result?.job) follow(result.job);
    return result;
  } catch (error) {
    message(error.message, "error");
    return null;
  }
}

async function forgetFolder(path) {
  if (!confirm(`Remove ${path} from “${state.collection}”?\n\nIts documents leave the collection and the index. The files on your computer are not touched.`)) return;
  await act(removeFolder(state.collection, path));
  showDocs(state.collection);
}

async function forgetCollection() {
  const name = state.collection;
  const n = state.detail?.documents.length ?? 0;
  const folders = state.detail?.folders.length ?? 0;
  const ok = confirm(
    `Delete the collection “${name}”?\n\nIts ${n} document${n === 1 ? "" : "s"} leave the app and the index, with their uploaded copies. ` +
      (folders ? `Its ${folders} folder(s) on your computer are not touched. ` : "") +
      "This can't be undone (adding the folders again re-reads them).",
  );
  if (!ok) return;
  const result = await act(deleteCollection(name));
  if (result) {
    message(`Deleted “${name}” (${result.documents_removed} document${result.documents_removed === 1 ? "" : "s"}). The index is being rebuilt.`);
    state.collection = null;
    await state.onCollectionsChanged("");
    showDocs("");
  }
}

async function forgetDocument(d) {
  const what = d.origin === "upload" ? "The uploaded copy is deleted too." : "The original file is not touched.";
  if (!confirm(`Remove “${d.title || d.doc_id}” from “${state.collection}”?\n\n${what}`)) return;
  await act(deleteDocument(d.doc_id));
}

async function upload(files) {
  if (!state.collection) return;
  const list = clear($("uploads"));
  let sent = 0;
  for (const file of files) {
    const row = el("li", {}, el("span", {}, file.webkitRelativePath || file.name), el("span", { class: "muted" }, "…"));
    list.append(row);
    const status = row.lastChild;
    if (!SUPPORTED.includes(suffix(file.name)) || file.name.startsWith(".") || file.name.startsWith("~$")) {
      status.textContent = "skipped: not a supported type";
      continue;
    }
    try {
      await uploadFile(state.collection, file);
      status.textContent = "uploaded";
      sent += 1;
    } catch (error) {
      row.classList.add("failed");
      status.textContent = error.message;
    }
  }
  if (sent) await act(syncCollection(state.collection));
}

// --- following a job ---------------------------------------------------------------------------

function follow(job) {
  state.job = job;
  state.seen = 0;
  clear($("job-events"));
  $("job-panel").hidden = false;
  $("job-cancel").hidden = false;
  $("job-summary").textContent = `${job.reason}`;
  clearTimeout(state.timer);
  poll();
}

async function poll() {
  const job = state.job;
  if (!job) return;
  let view;
  try {
    view = await getJob(job.id, state.seen);
  } catch (error) {
    $("job-summary").textContent = `Lost track of the job: ${error.message}`;
    return;
  }
  state.seen = view.n_events;
  const events = $("job-events");
  for (const e of view.events) {
    events.append(el("li", { class: e.kind }, `${e.kind.padEnd(9)} ${e.name}${e.detail ? `  (${e.detail})` : ""}`));
  }
  events.scrollTop = events.scrollHeight;
  $("job-state").textContent = view.status;
  if (!TERMINAL.has(view.status)) {
    state.timer = setTimeout(poll, 500);
    return;
  }
  $("job-cancel").hidden = true;
  const r = view.result;
  $("job-summary").textContent = r
    ? `${r.added.length} added, ${r.updated.length} updated, ${r.unchanged} unchanged, ${r.removed.length} removed, ` +
      `${r.failed.length} failed, ${r.skipped.length} skipped` +
      (r.seconds?.total !== undefined ? ` · ${r.seconds.total} s` : "") +
      (r.cancelled ? " · cancelled: what was finished is searchable" : "")
    : view.error || view.status;
  state.job = null;
  state.onChanged();
  showDocs(state.collection);
}

// --- wiring ------------------------------------------------------------------------------------

export function setupDocs({ onCollectionsChanged }) {
  state.onChanged = () => onCollectionsChanged(state.collection);
  state.onCollectionsChanged = onCollectionsChanged;
  $("delete-collection").addEventListener("click", forgetCollection);
  $("new-collection").addEventListener("submit", async (event) => {
    event.preventDefault();
    const name = $("new-name").value.trim();
    const created = await act(createCollection(name, $("new-description").value.trim()));
    if (created) {
      $("new-name").value = "";
      $("new-description").value = "";
      await onCollectionsChanged(name);
      showDocs(name);
    }
  });
  $("add-folder").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (await act(addFolder(state.collection, $("folder-path").value.trim()))) $("folder-path").value = "";
    showDocs(state.collection);
  });
  $("rescan").addEventListener("click", () => act(syncCollection(state.collection)));
  $("job-cancel").addEventListener("click", () => state.job && cancelJob(state.job.id));
  $("pick-files").addEventListener("change", (event) => upload([...event.target.files]));
  $("pick-folder").addEventListener("change", (event) => upload([...event.target.files]));
  const drop = $("drop");
  drop.addEventListener("dragover", (event) => {
    event.preventDefault();
    drop.classList.add("over");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", async (event) => {
    event.preventDefault();
    drop.classList.remove("over");
    upload(await droppedFiles(event.dataTransfer));
  });
}

// Files from a drop, including the files inside dropped folders.
async function droppedFiles(transfer) {
  const entries = [...transfer.items].map((item) => item.webkitGetAsEntry?.()).filter(Boolean);
  if (!entries.length) return [...transfer.files];
  const files = [];
  const walk = async (entry) => {
    if (entry.isFile) {
      files.push(await new Promise((resolve, reject) => entry.file(resolve, reject)));
    } else if (entry.isDirectory && !entry.name.startsWith(".")) {
      const reader = entry.createReader();
      for (;;) {
        const batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
        if (!batch.length) break;
        for (const child of batch) await walk(child);
      }
    }
  };
  for (const entry of entries) await walk(entry);
  return files;
}

// The passage dialog: the section(s) a source or hit came from, with its chunks highlighted, and a
// link that opens the stored PDF at the right page in the browser's own viewer.

import { fileUrl, getPassages } from "./api.js";
import { $, append, clear, el, pages } from "./dom.js";

function highlighted(text, highlights) {
  const nodes = [];
  let at = 0;
  for (const h of highlights) {
    if (h.start < at) continue; // overlapping chunks: keep the first
    nodes.push(text.slice(at, h.start), el("mark", {}, text.slice(h.start, h.end)));
    at = h.end;
  }
  nodes.push(text.slice(at));
  return nodes;
}

export async function showPassage(chunkIds, { title, page } = {}) {
  const dialog = $("passage");
  const body = clear($("passage-body"));
  $("passage-title").textContent = title || "Passage";
  $("passage-sub").textContent = "loading…";
  const open = $("passage-open");
  open.hidden = true;
  if (!dialog.open) dialog.showModal();
  let context;
  try {
    context = await getPassages(chunkIds);
  } catch (error) {
    $("passage-sub").textContent = `Couldn't load the passage: ${error.message}`;
    return;
  }
  const doc = context.document;
  $("passage-title").textContent = doc.title || doc.doc_id;
  const firstPage = page || context.sections.flatMap((s) => s.highlights.map((h) => h.page)).find(Boolean);
  $("passage-sub").textContent = [doc.doc_id, doc.collection, pages(firstPage)].filter(Boolean).join(" · ");
  if (doc.has_file && doc.format === "pdf") {
    open.href = `${fileUrl(doc.doc_id)}${firstPage ? `#page=${firstPage}` : ""}`;
    open.textContent = firstPage ? `Open PDF at page ${firstPage}` : "Open PDF";
    open.hidden = false;
  } else if (doc.has_file) {
    open.href = fileUrl(doc.doc_id);
    open.textContent = doc.format === "html" || doc.format === "docx" ? `Download .${doc.format}` : "Open file";
    open.hidden = false;
  }
  for (const section of context.sections) {
    append(body, [
      el("h3", {}, section.path || "(no heading)"),
      el("div", { class: "section-text" }, ...highlighted(section.text, section.highlights)),
    ]);
  }
  body.querySelector("mark")?.scrollIntoView({ block: "center" });
}

export function setupPassageDialog() {
  const dialog = $("passage");
  $("passage-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close(); // click on the backdrop
  });
}

// Calls to the rag API (same origin). Every response body is data: the UI renders it as text only.

export class ApiError extends Error {
  constructor(status, detail) {
    super(detail || `HTTP ${status}`);
    this.status = status;
  }
}

async function json(url, options = {}, { acceptStatus = [] } = {}) {
  const response = await fetch(url, options);
  let body = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok && !acceptStatus.includes(response.status)) {
    const detail = body && typeof body.detail === "string" ? body.detail : JSON.stringify(body?.detail ?? "");
    throw new ApiError(response.status, detail || response.statusText);
  }
  return body;
}

const post = (body, signal) => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
  signal,
});

const docPath = (docId) => docId.split("/").map(encodeURIComponent).join("/");

// /health answers 503 with a body when the pipeline can't load: that body is what we show.
export const getHealth = () => json("/health", {}, { acceptStatus: [503] });
export const getCollections = () => json("/collections");
export const search = (body) => json("/search", post(body));
export const getDocument = (docId) => json(`/documents/${docPath(docId)}`);
export const fileUrl = (docId) => `/documents/${docPath(docId)}/file`;
export const getPassages = (chunkIds) =>
  json(`/passages?${chunkIds.map((id) => `chunk_id=${encodeURIComponent(id)}`).join("&")}`);
export const getLog = (requestId) => json(`/logs/${encodeURIComponent(requestId)}`);

// POST /ask as server-sent events, read with fetch (EventSource can't send a body).
// handlers: { sources(data), token(data), done(data), error(data) }.
export async function askStream(body, handlers, signal) {
  const response = await fetch("/ask", post({ ...body, stream: true }, signal));
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new ApiError(response.status, typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      let event = "message";
      const data = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data.push(line.slice(6));
      }
      if (data.length && handlers[event]) handlers[event](JSON.parse(data.join("\n")));
    }
  }
}

// --- adding and keeping documents (steps 24 + 30) ----------------------------------------------

const send = (method, body) => ({
  method,
  headers: body === undefined ? {} : { "Content-Type": "application/json" },
  body: body === undefined ? undefined : JSON.stringify(body),
});
const name = (collection) => encodeURIComponent(collection);

export const createCollection = (collection, description) =>
  json("/collections", send("POST", { name: collection, description }));
export const getCollection = (collection) => json(`/collections/${name(collection)}`);
export const addFolder = (collection, path) => json(`/collections/${name(collection)}/folders`, send("POST", { path }));
export const removeFolder = (collection, path) =>
  json(`/collections/${name(collection)}/folders?path=${encodeURIComponent(path)}`, send("DELETE"));
export const syncCollection = (collection) => json(`/collections/${name(collection)}/sync`, send("POST"));
export const deleteDocument = (docId) => json(`/documents/${docPath(docId)}`, send("DELETE"));
export const getJob = (id, from = 0) => json(`/jobs/${encodeURIComponent(id)}?events_from=${from}`);
export const cancelJob = (id) => json(`/jobs/${encodeURIComponent(id)}/cancel`, send("POST"));

// One file per request, its bytes as the body (the server needs no multipart parser).
export const uploadFile = (collection, file) =>
  json(`/collections/${name(collection)}/files/${encodeURIComponent(file.name)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/octet-stream" },
    body: file,
  });
export const deleteCollection = (collection) => json(`/collections/${name(collection)}`, send("DELETE"));

// --- activity and evaluation (step 22): read-only -----------------------------------------------

const query = (params) => {
  const parts = Object.entries(params)
    .filter(([, v]) => v !== null && v !== undefined && v !== "" && v !== false)
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`);
  return parts.length ? `?${parts.join("&")}` : "";
};

export const getLogs = (filters) => json(`/logs${query(filters)}`);
export const getLogStats = (filters) => json(`/logs/stats${query(filters)}`);
export const getEvalRuns = (set) => json(`/eval/runs${query({ set })}`);
export const getEvalRun = (runId) => json(`/eval/runs/${encodeURIComponent(runId)}`);
export const compareRuns = (a, b) => json(`/eval/compare${query({ a, b })}`);
export const getAnswerRuns = () => json("/eval/answers");
export const getGate = () => json("/eval/gate", {}, { acceptStatus: [404] });

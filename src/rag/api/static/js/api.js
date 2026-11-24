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

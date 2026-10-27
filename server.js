// chatbot backend - single file, zero dependencies, no cloud, no services.
// Serves the front end from ./public, exposes POST /api/chat, and keeps its
// OWN conversation memory right here in this file.
//
//   Run:   node server.js      then open   http://localhost:8787
//
// MEMORY (self-contained):
//   - Every user message and reply is appended to memory.json next to this file.
//   - On each turn the backend searches past turns (scoped by project) for the
//     ones most relevant to the new message and injects them as context.
//   - This survives restarts. No database, no npm dependencies.
//   - Recall is hybrid: BM25 keyword scoring always, plus embedding vectors
//     when an embedding provider is configured (see HYBRID RECALL below).
//     With no provider it degrades to keyword-only and still runs anywhere.

const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = process.env.PORT || 8787;
const PUBLIC_DIR = path.join(__dirname, "public");
const MEMORY_FILE = process.env.MEMORY_FILE || path.join(__dirname, "memory.json");
const MEMORY_ENABLED = process.env.MEMORY_DISABLED ? false : true;
const RECALL_LIMIT = Number(process.env.RECALL_LIMIT || 4);   // how many past turns to surface
const MEMORY_MAX = Number(process.env.MEMORY_MAX || 5000);    // cap turns kept on disk

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

// ---------------------------------------------------------------------------
// Self-contained memory store. Shape: { turns: [ {id, chatId, role, content, ts} ] }
// ---------------------------------------------------------------------------
let store = loadStore();

function loadStore() {
  try {
    const s = JSON.parse(fs.readFileSync(MEMORY_FILE, "utf8"));
    if (s && Array.isArray(s.turns)) return s;
  } catch (e) { /* no file yet, or unreadable -> start fresh */ }
  return { turns: [] };
}

function saveStore() {
  try {
    fs.writeFileSync(MEMORY_FILE, JSON.stringify(store));
  } catch (e) { console.warn("memory save skipped:", e.message); }
}

// ---------------------------------------------------------------------------
// HYBRID RECALL: BM25 keyword scoring + embedding vectors, fused
// VECTOR_WEIGHT (default 0.7) vector / 0.3 keyword. A turn found by both is
// labelled "hybrid". Vectors live in a sidecar file next to memory.json, so
// there is still no database and still zero npm dependencies.
//
// Embeddings are optional and pluggable. The first one configured wins:
//   EMBED_API_URL + EMBED_API_KEY  -> any OpenAI-compatible /embeddings endpoint
//   VOYAGE_API_KEY                 -> Voyage AI
//   OLLAMA_EMBED_URL               -> local Ollama (e.g. http://localhost:11434)
//   nothing set                    -> keyword-only recall (BM25), as before
// EMBED_MODEL overrides the default model for whichever is active. Changing
// provider or model re-embeds the whole memory in the background on startup.
// ---------------------------------------------------------------------------
const VECTOR_FILE      = process.env.VECTOR_FILE || MEMORY_FILE.replace(/\.json$/i, "") + ".vectors.json";
const EMBED_API_URL    = process.env.EMBED_API_URL || "";
const EMBED_API_KEY    = process.env.EMBED_API_KEY || "";
const VOYAGE_KEY       = process.env.VOYAGE_API_KEY || "";
const OLLAMA_EMBED_URL = process.env.OLLAMA_EMBED_URL || "";
const VECTOR_WEIGHT    = Number(process.env.VECTOR_WEIGHT || 0.7);
const VECTOR_MIN       = Number(process.env.VECTOR_MIN || 0.3);   // cosine floor for a vector hit
const EMBED_BATCH      = 64;
const EMBED_TIMEOUT    = Number(process.env.EMBED_TIMEOUT_MS || 15000);  // a hung provider must not stall chat

const EMBED = (() => {
  const m = process.env.EMBED_MODEL || "";
  if (EMBED_API_URL && EMBED_API_KEY) return { kind: "openai", model: m || "text-embedding-3-small" };
  if (VOYAGE_KEY)                     return { kind: "voyage", model: m || "voyage-3.5-lite" };
  if (OLLAMA_EMBED_URL)               return { kind: "ollama", model: m || "nomic-embed-text" };
  return null;
})();
const EMBED_ID = EMBED ? EMBED.kind + ":" + EMBED.model : null;

function unit(arr) {
  const v = Float32Array.from(arr);
  let n = 0; for (let i = 0; i < v.length; i++) n += v[i] * v[i];
  n = Math.sqrt(n) || 1;
  for (let i = 0; i < v.length; i++) v[i] /= n;
  return v;
}
function dot(a, b) {
  if (a.length !== b.length) return 0;
  let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

// inputType: "query" | "document" (Voyage uses it; the others ignore it)
async function embed(texts, inputType) {
  if (!EMBED || !texts.length) return [];
  const input = texts.map(t => (t || "").slice(0, 8000));
  const headers = { "Content-Type": "application/json" };
  let url, body;
  if (EMBED.kind === "openai") {
    url = /\/embeddings$/.test(EMBED_API_URL) ? EMBED_API_URL : EMBED_API_URL.replace(/\/$/, "") + "/embeddings";
    headers.Authorization = "Bearer " + EMBED_API_KEY;
    body = { model: EMBED.model, input };
  } else if (EMBED.kind === "voyage") {
    url = "https://api.voyageai.com/v1/embeddings";
    headers.Authorization = "Bearer " + VOYAGE_KEY;
    body = { model: EMBED.model, input, input_type: inputType };
  } else {
    url = /\/api\/embed$/.test(OLLAMA_EMBED_URL) ? OLLAMA_EMBED_URL : OLLAMA_EMBED_URL.replace(/\/$/, "") + "/api/embed";
    body = { model: EMBED.model, input };
  }
  const res = await fetch(url, { method: "POST", headers, body: JSON.stringify(body), signal: AbortSignal.timeout(EMBED_TIMEOUT) });
  if (!res.ok) throw new Error("embed " + res.status + ": " + (await res.text()).slice(0, 200));
  const data = await res.json();
  const vecs = data.embeddings ||
    (data.data || []).slice().sort((a, b) => (a.index || 0) - (b.index || 0)).map(d => d.embedding);
  if (!vecs || vecs.length !== texts.length) throw new Error("embed returned " + (vecs || []).length + " vectors for " + texts.length + " inputs");
  return vecs.map(unit);
}

// Sidecar vector store: { model, vecs: { turnId: base64(Float32Array) } }
const toB64 = v => Buffer.from(v.buffer, v.byteOffset, v.byteLength).toString("base64");
const fromB64 = s => { const b = Buffer.from(s, "base64"); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };

let vectors = loadVectors();   // Map turnId -> Float32Array (unit length)

function loadVectors() {
  const map = new Map();
  if (!EMBED) return map;
  try {
    const v = JSON.parse(fs.readFileSync(VECTOR_FILE, "utf8"));
    if (v && v.model === EMBED_ID && v.vecs) {
      for (const [id, b64] of Object.entries(v.vecs)) map.set(id, fromB64(b64));
    } else if (v && v.model) {
      console.log("embedding model changed (" + v.model + " -> " + EMBED_ID + "), re-embedding memory");
    }
  } catch (e) { /* no sidecar yet */ }
  return map;
}

function saveVectors() {
  const live = new Set(store.turns.map(t => t.id));
  const vecs = {};
  for (const [id, v] of vectors) {
    if (live.has(id)) vecs[id] = toB64(v);
    else vectors.delete(id);                     // turn was trimmed by MEMORY_MAX
  }
  try { fs.writeFileSync(VECTOR_FILE, JSON.stringify({ model: EMBED_ID, vecs })); }
  catch (e) { console.warn("vector save skipped:", e.message); }
}

// Embed every stored turn that has no vector yet. Runs in the background,
// one batch at a time, never more than one pass in flight. Covers both the
// startup backfill and each new turn.
let indexing = null;
function indexPending() {
  if (!EMBED || indexing) return indexing;
  const run = (async () => {
    try {
      for (;;) {
        const todo = store.turns.filter(t => t.content && !vectors.has(t.id)).slice(0, EMBED_BATCH);
        if (!todo.length) break;
        const vecs = await embed(todo.map(t => t.content), "document");
        todo.forEach((t, i) => vectors.set(t.id, vecs[i]));
        saveVectors();
      }
    } catch (e) { console.warn("embedding paused:", e.message); }
  })();
  // clear via .finally (always async) -- clearing inside the body can run
  // synchronously before this assignment and leave the flag stuck forever
  indexing = run.finally(() => { indexing = null; });
  return indexing;
}

const STOP = new Set(("a an and are as at be but by can do does did for from has have how i if in is it its " +
  "me my of on or so that the this to was we what when where which who why will with you your").split(" "));

function tokenize(s) {
  return (((s || "").toLowerCase().match(/[a-z0-9]+/g)) || []).filter(t => !STOP.has(t));
}

// BM25 over the in-scope pool. Returns Map turnId -> score.
function bm25(pool, queryTokens) {
  const out = new Map();
  const q = [...new Set(queryTokens)];
  if (!pool.length || !q.length) return out;
  const docs = pool.map(t => tokenize(t.content));
  const N = docs.length;
  const avgdl = docs.reduce((s, d) => s + d.length, 0) / N || 1;
  const df = new Map();
  for (const d of docs) for (const w of new Set(d)) df.set(w, (df.get(w) || 0) + 1);
  const k1 = 1.2, b = 0.75;
  docs.forEach((d, i) => {
    if (!d.length) return;
    const tf = new Map();
    for (const w of d) tf.set(w, (tf.get(w) || 0) + 1);
    let s = 0;
    for (const w of q) {
      const f = tf.get(w);
      if (!f) continue;
      const n = df.get(w);
      const idf = Math.log(1 + (N - n + 0.5) / (n + 0.5));
      s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * d.length / avgdl));
    }
    if (s > 0) out.set(pool[i].id, s);
  });
  return out;
}

// Memory is scoped by project. Chats inside a project can see each other;
// chats outside it cannot. Unassigned chats form their own pool. This is what
// makes a project mean something rather than being a label in the sidebar.
function inScope(turn, chatId, projectId) {
  if (turn.chatId === chatId) return true;                 // same conversation
  const tp = turn.projectId || null;
  const cp = projectId || null;
  return tp === cp;                                        // same project, or both unassigned
}

// Score the in-scope pool against a query. Returns [{ turn, score, source }].
async function search(query, chatId, projectId, exclude) {
  const pool = store.turns.filter(t => !(exclude && exclude.has(t.content)) && inScope(t, chatId, projectId));
  if (!pool.length || !query) return [];

  const kw = bm25(pool, tokenize(query));
  let kwMax = 0; for (const s of kw.values()) if (s > kwMax) kwMax = s;

  const vs = new Map();
  if (EMBED && vectors.size) {
    try {
      const [qv] = await embed([query], "query");
      for (const t of pool) {
        const v = vectors.get(t.id);
        if (!v) continue;
        const s = dot(qv, v);
        if (s >= VECTOR_MIN) vs.set(t.id, s);
      }
    } catch (e) { console.warn("vector recall skipped:", e.message); }
  }

  const byId = new Map(pool.map(t => [t.id, t]));
  const hits = [];
  for (const id of new Set([...kw.keys(), ...vs.keys()])) {
    const k = kw.has(id) ? kw.get(id) / (kwMax || 1) : 0;
    const v = vs.get(id) || 0;
    const source = vs.has(id) && kw.has(id) ? "hybrid" : vs.has(id) ? "vector" : "keyword";
    hits.push({ turn: byId.get(id), score: VECTOR_WEIGHT * v + (1 - VECTOR_WEIGHT) * k, source });
  }
  hits.sort((a, b) => b.score - a.score);
  return hits;
}

// RECALL: find the most relevant past turns and format them as a context block.
async function recall(messages, chatId, projectId) {
  if (!MEMORY_ENABLED || !store.turns.length) return "";
  const lastUser = [...messages].reverse().find(m => m && m.role === "user");
  if (!lastUser || !lastUser.content) return "";

  // skip turns whose text is already in the live history (no point re-injecting)
  const live = new Set(messages.map(m => (m && m.content) || ""));
  const hits = (await search(lastUser.content, chatId, projectId, live)).slice(0, RECALL_LIMIT);
  if (!hits.length) return "";

  const lines = hits.map(({ turn }) => {
    const who = turn.role === "user" ? "User" : "Assistant";
    const where = turn.chatId === chatId
      ? "earlier in this chat"
      : (projectId ? "from another chat in this project" : "from another chat");
    const text = turn.content.length > 320 ? turn.content.slice(0, 320) + "..." : turn.content;
    return "- (" + where + ", " + who + " said) " + text;
  });
  return "## Recalled from memory\nThe following may be relevant to the user's message:\n" + lines.join("\n");
}

// REMEMBER: append the newest turns to the store and persist.
function remember(newTurns, chatId, projectId) {
  if (!MEMORY_ENABLED) return;
  let added = false;
  for (const m of newTurns) {
    if (!m || !m.content) continue;
    store.turns.push({
      id: Date.now().toString(36) + Math.random().toString(36).slice(2, 8),
      chatId, projectId: projectId || null,
      role: m.role, content: m.content, ts: Date.now(),
    });
    added = true;
  }
  if (!added) return;
  if (store.turns.length > MEMORY_MAX) store.turns = store.turns.slice(-MEMORY_MAX);
  saveStore();
  indexPending();          // embed the new turns in the background
}

// ---------------------------------------------------------------------------
// UI state: chats, projects, schedules and settings. Kept separate from
// memory.json on purpose -- memory.json is the model's recall corpus, this is
// what the interface needs to redraw itself after a refresh.
// ---------------------------------------------------------------------------
function loadState() {
  try {
    const s = JSON.parse(fs.readFileSync(STATE_FILE, "utf8"));
    if (s && typeof s === "object") return s;
  } catch (e) { /* no file yet */ }
  return { chats: [], projects: [], schedules: [], settings: null };
}

function saveState(next) {
  const current = loadState();
  const merged = {
    chats:     Array.isArray(next.chats)     ? next.chats     : current.chats,
    projects:  Array.isArray(next.projects)  ? next.projects  : current.projects,
    schedules: Array.isArray(next.schedules) ? next.schedules : current.schedules,
    settings:  next.settings !== undefined   ? next.settings  : current.settings,
  };
  fs.writeFileSync(STATE_FILE, JSON.stringify(merged));
  return merged;
}

// ---------------------------------------------------------------------------
// *** MODEL SEAM ***  Generate a reply from the (context-augmented) messages.
//
// Config-driven so the same clone runs anywhere. Set env vars on the machine;
// the code uses the first backend that is actually configured:
//
//   1. MODEL_API_URL + MODEL_API_KEY   -> any OpenAI-compatible endpoint
//   2. ANTHROPIC_API_KEY               -> Anthropic messages API
//   3. OLLAMA_URL                      -> local Ollama (e.g. http://localhost:11434)
//   4. nothing set                     -> mock that says what to set
//
// No key is ever hardcoded, and nothing has to be installed for 1 or 2. That
// is the point: clone the repo on any machine, set one env var, and it runs.
// ---------------------------------------------------------------------------

const MODEL_API_URL = process.env.MODEL_API_URL || "";
const MODEL_API_KEY = process.env.MODEL_API_KEY || "";
const ANTHROPIC_KEY = process.env.ANTHROPIC_API_KEY || "";
const OLLAMA_URL    = process.env.OLLAMA_URL || "";
const DEFAULT_MODEL = process.env.MODEL_NAME || "";
const BOT_NAME      = process.env.BOT_NAME || "";   // fallback identity if the client sends none
const STATE_FILE    = process.env.STATE_FILE || path.join(__dirname, "state.json");

function activeBackend() {
  if (MODEL_API_URL && MODEL_API_KEY) return "openai-compatible";
  if (ANTHROPIC_KEY) return "anthropic";
  if (OLLAMA_URL) return "ollama";
  return "mock";
}

// Providers reject unknown fields, and the front end may carry its own
// per-message metadata (attachments, ids, timestamps). Reduce every message
// to the only two keys any provider accepts.
function sanitize(messages) {
  return (messages || [])
    .filter(m => m && m.role && typeof m.content === "string" && m.content.length)
    .map(m => ({ role: m.role, content: m.content }));
}

async function generateReply(rawMessages, model) {
  const messages = sanitize(rawMessages);
  const backend = activeBackend();
  const chosen = model || DEFAULT_MODEL;

  if (backend === "openai-compatible") {
    const res = await fetch(MODEL_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + MODEL_API_KEY,
      },
      body: JSON.stringify({ model: chosen || "gpt-4o-mini", messages }),
    });
    if (!res.ok) throw new Error("model API " + res.status + ": " + (await res.text()).slice(0, 300));
    const data = await res.json();
    return (((data.choices || [])[0] || {}).message || {}).content || "";
  }

  if (backend === "anthropic") {
    // Anthropic keeps the system prompt out of messages[], so split it off.
    const system = messages.filter(m => m.role === "system").map(m => m.content).join("\n\n");
    const rest = messages.filter(m => m.role !== "system");
    const res = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_KEY,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: chosen || "claude-sonnet-4-6",
        max_tokens: 2048,
        system: system || undefined,
        messages: rest,
      }),
    });
    if (!res.ok) throw new Error("anthropic " + res.status + ": " + (await res.text()).slice(0, 300));
    const data = await res.json();
    return (data.content || []).filter(b => b.type === "text").map(b => b.text).join("") || "";
  }

  if (backend === "ollama") {
    const url = OLLAMA_URL.endsWith("/api/chat") ? OLLAMA_URL : OLLAMA_URL.replace(/\/$/, "") + "/api/chat";
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: chosen || "llama3.1", messages, stream: false }),
    });
    if (!res.ok) throw new Error("ollama " + res.status + ": " + (await res.text()).slice(0, 300));
    const data = await res.json();
    return (data.message || {}).content || "";
  }

  // Nothing configured: say so plainly rather than pretending to work.
  const last = (messages[messages.length - 1] || {}).content || "";
  return 'You said: "' + last + '"\n\n' +
    "No model backend is configured on this machine. Set one of:\n" +
    "  MODEL_API_URL + MODEL_API_KEY   (any OpenAI-compatible endpoint)\n" +
    "  ANTHROPIC_API_KEY               (Anthropic)\n" +
    "  OLLAMA_URL                      (local Ollama)\n" +
    "Memory is still recording this conversation either way.";
}

// Full chat turn: recall -> augment -> model -> remember.
async function handleChat(body) {
  const messages = body.messages || [];
  const model = body.model;
  const chatId = body.chatId || "default";
  const projectId = body.projectId || null;

  // Identity comes from the client (the name set in Options) or from BOT_NAME
  // on the server. Without it the model answers as whatever the provider
  // trained it to call itself.
  const name = body.botName || BOT_NAME;
  const persona = name
    ? "You are " + name + ". That is your name; use it if asked who you are. " +
      "Do not refer to yourself by any other name or by the name of the model " +
      "or company that built you."
    : "";

  const context = await recall(messages, chatId, projectId);

  const prefix = [];
  if (persona) prefix.push({ role: "system", content: persona });
  if (context) prefix.push({ role: "system", content: context });
  const augmented = prefix.length ? [...prefix, ...messages] : messages;

  const reply = await generateReply(augmented, model);

  // store only the new turns this round (front end resends full history each call,
  // so persisting everything every time would duplicate massively).
  const lastUser = [...messages].reverse().find(m => m && m.role === "user");
  const toStore = [];
  if (lastUser) toStore.push(lastUser);
  toStore.push({ role: "assistant", content: reply });
  remember(toStore, chatId, projectId);

  return reply;
}

function sendJson(res, status, obj) {
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(obj));
}

function serveStatic(req, res) {
  let rel = decodeURIComponent(req.url.split("?")[0]);
  if (rel === "/" || rel === "") rel = "/chatbot.html";
  const filePath = path.join(PUBLIC_DIR, path.normalize(rel));
  if (!filePath.startsWith(PUBLIC_DIR)) { res.writeHead(403); res.end("forbidden"); return; }
  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404, { "Content-Type": "text/plain" });
      res.end("not found - drop chatbot.html into ./public");
      return;
    }
    const ext = path.extname(filePath).toLowerCase();
    res.writeHead(200, { "Content-Type": MIME[ext] || "application/octet-stream" });
    res.end(data);
  });
}

const server = http.createServer((req, res) => {
  if (req.method === "POST" && req.url === "/api/chat") {
    let raw = "";
    req.on("data", c => { raw += c; if (raw.length > 1e6) req.destroy(); });
    req.on("end", async () => {
      try {
        const body = JSON.parse(raw || "{}");
        const reply = await handleChat(body);
        sendJson(res, 200, { reply });
      } catch (err) {
        sendJson(res, 500, { error: String((err && err.message) || err) });
      }
    });
    return;
  }
  // UI state: everything the interface needs to redraw after a refresh
  if (req.url === "/api/state") {
    if (req.method === "GET") { sendJson(res, 200, loadState()); return; }
    if (req.method === "PUT") {
      let raw = "";
      req.on("data", c => { raw += c; if (raw.length > 5e6) req.destroy(); });
      req.on("end", () => {
        try { sendJson(res, 200, saveState(JSON.parse(raw || "{}"))); }
        catch (err) { sendJson(res, 500, { error: String((err && err.message) || err) }); }
      });
      return;
    }
  }

  // tiny introspection endpoint: how much is remembered, and which backend is live
  if (req.method === "GET" && req.url === "/api/memory/stats") {
    const chats = new Set(store.turns.map(t => t.chatId));
    sendJson(res, 200, {
      turns: store.turns.length, chats: chats.size, file: MEMORY_FILE, backend: activeBackend(),
      embeddings: EMBED_ID || "off", vectors: vectors.size, indexing: !!indexing,
    });
    return;
  }

  // debug: see what recall would surface, and why.  /api/memory/search?q=...&chatId=...&projectId=...
  if (req.method === "GET" && req.url.startsWith("/api/memory/search")) {
    const u = new URL(req.url, "http://localhost");
    search(u.searchParams.get("q") || "", u.searchParams.get("chatId") || "", u.searchParams.get("projectId") || null)
      .then(hits => sendJson(res, 200, hits.slice(0, 20).map(h => ({
        score: +h.score.toFixed(4), source: h.source, role: h.turn.role,
        chatId: h.turn.chatId, projectId: h.turn.projectId, text: h.turn.content.slice(0, 200),
      }))))
      .catch(err => sendJson(res, 500, { error: String(err.message || err) }));
    return;
  }
  if (req.method === "GET") { serveStatic(req, res); return; }
  res.writeHead(405); res.end("method not allowed");
});

server.listen(PORT, () => {
  console.log("chatbot backend on http://localhost:" + PORT);
  console.log("serving ./public  -  POST /api/chat");
  console.log("memory: " + (MEMORY_ENABLED ? (MEMORY_FILE + " (" + store.turns.length + " turns loaded)") : "disabled"));
  console.log("model backend: " + activeBackend());
  console.log("embeddings: " + (EMBED_ID ? EMBED_ID + " (" + vectors.size + " vectors loaded)" : "off - keyword-only recall"));
  indexPending();   // backfill any turns that have no vector yet
});

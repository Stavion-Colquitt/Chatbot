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

const ANTHROPIC_URL = (process.env.ANTHROPIC_BASE_URL || "https://api.anthropic.com").replace(/\/$/, "") + "/v1/messages";

async function anthropicRequest(system, messages, model, tools) {
  const res = await fetch(ANTHROPIC_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": ANTHROPIC_KEY,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model: model || DEFAULT_MODEL || "claude-sonnet-4-6",
      max_tokens: 4096,
      system: system || undefined,
      messages,
      tools: tools && tools.length ? tools : undefined,
    }),
  });
  if (!res.ok) throw new Error("anthropic " + res.status + ": " + (await res.text()).slice(0, 300));
  return res.json();
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
    const data = await anthropicRequest(system, rest, chosen);
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

// ---------------------------------------------------------------------------
// MCP TOOLS: connect local MCP servers (stdio) and let the model use them.
//
// Everything here is local: each server is a program on this machine that the
// backend starts as a child process and talks to over stdin/stdout (JSON-RPC,
// one message per line). No npm packages, no network for the MCP side.
//
// Config lives in mcp.json next to this file (gitignored -- servers are
// installed per machine). Same shape as Claude Desktop's config, plus two
// optional keys per server:
//   { "mcpServers": { "name": { "command": "node", "args": [...], "env": {...},
//                               "enabled": true, "alwaysAllow": ["tool_name"] } } }
//
// Safety: every tool call pauses for approval in the UI unless that tool is in
// the server's alwaysAllow list (the "Always allow" button adds it).
// Tool use runs on the Anthropic backend; other backends chat without tools.
// ---------------------------------------------------------------------------
const { spawn, spawnSync } = require("child_process");
const MCP_FILE         = process.env.MCP_FILE || path.join(__dirname, "mcp.json");
const MAX_TOOL_ROUNDS  = Number(process.env.MAX_TOOL_ROUNDS || 10);
const TOOL_TIMEOUT     = Number(process.env.TOOL_TIMEOUT_MS || 120000);
const TOOL_RESULT_MAX  = 100000;   // chars of tool output sent back to the model
const IS_WIN           = process.platform === "win32";

function claudeDesktopConfigPath() {
  if (process.env.CLAUDE_DESKTOP_CONFIG) return process.env.CLAUDE_DESKTOP_CONFIG;
  const home = require("os").homedir();
  if (IS_WIN) return path.join(process.env.APPDATA || path.join(home, "AppData", "Roaming"), "Claude", "claude_desktop_config.json");
  if (process.platform === "darwin") return path.join(home, "Library", "Application Support", "Claude", "claude_desktop_config.json");
  return path.join(home, ".config", "Claude", "claude_desktop_config.json");
}

// PORTABLE PATHS. Config values may use ${HOME}, ~, %VAR% or ${VAR}; they're
// expanded when a server starts, so one mcp.json works on any machine where the
// servers sit at the same place relative to the home folder. Imported and
// pasted configs have this machine's home folder rewritten to ${HOME}.
const HOME_DIR = require("os").homedir();

function expandVars(v) {
  if (typeof v !== "string") return v;
  let out = v
    .replace(/^~(?=$|[\\/])/, HOME_DIR)
    .replace(/\$\{(\w+)\}|%(\w+)%/g, (m, a, b) => {
      const k = a || b;
      if (k === "HOME") return HOME_DIR;
      if (k === "APP_DIR") return __dirname;         // wherever this repo is cloned
      if (k === "CONFIG_DIR") return path.dirname(path.resolve(MCP_FILE));   // folder holding mcp.json
      const hit = Object.keys(process.env).find(e => IS_WIN ? e.toUpperCase() === k.toUpperCase() : e === k);
      return hit ? process.env[hit] : m;            // unknown variable: leave it visible
    });
  // ${HOME}\a\b written on Windows still resolves on mac/Linux, and vice versa.
  // Done per item so path lists ("a,b;c") and --flag=path values work too.
  return out.split(/([,;=])/).map(seg => seg.startsWith(HOME_DIR)
    ? HOME_DIR + seg.slice(HOME_DIR.length).replace(/[\\/]/g, path.sep) : seg).join("");
}

function portable(cfg) {
  // match the home folder with either slash style (and any case on Windows)
  const esc = HOME_DIR.split(/[\\/]/).map(seg => seg.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("[\\\\/]");
  // at the start of a value, or of an item in a list / --flag=value
  const re = new RegExp("(^|[,;=])" + esc + "(?=$|[\\\\/,;])", IS_WIN ? "gi" : "g");
  const fix = v => (typeof v === "string" ? v.replace(re, "$1${HOME}") : v);
  const out = { ...cfg, command: fix(cfg.command) };
  if (Array.isArray(cfg.args)) out.args = cfg.args.map(fix);
  if (cfg.cwd) out.cwd = fix(cfg.cwd);
  if (cfg.env) out.env = Object.fromEntries(Object.entries(cfg.env).map(([k, v]) => [k, fix(v)]));
  return out;
}

function loadMcpConfig() {
  try {
    const c = JSON.parse(fs.readFileSync(MCP_FILE, "utf8"));
    if (c && typeof c.mcpServers === "object") return c;
  } catch (e) { /* none yet */ }
  return { mcpServers: {} };
}
let mcpConfig = loadMcpConfig();
function saveMcpConfig() { fs.writeFileSync(MCP_FILE, JSON.stringify(mcpConfig, null, 2)); }

// Servers that ship inside this repo (./mcp). Each is added once per machine;
// remove one in Options and it stays removed. A same-named server you already
// have (e.g. imported from Claude Desktop) is left alone.
const BUNDLED_SERVERS = {
  Fetch: { command: "node", args: ["${APP_DIR}/mcp/fetch.js"], enabled: true, alwaysAllow: [] },
};
(function addBundled() {
  const done = new Set(mcpConfig.bundledAdded || []);
  let changed = false;
  for (const [name, cfg] of Object.entries(BUNDLED_SERVERS)) {
    if (done.has(name)) continue;
    if (!mcpConfig.mcpServers[name]) mcpConfig.mcpServers[name] = { ...cfg, args: [...cfg.args] };
    done.add(name); changed = true;
  }
  if (changed) {
    mcpConfig.bundledAdded = [...done];
    try { saveMcpConfig(); } catch (e) { console.warn("mcp config save skipped:", e.message); }
  }
})();

class McpClient {
  constructor(name) {
    this.name = name; this.status = "stopped"; this.error = null; this.tools = []; this.instructions = "";
    this.proc = null; this.buf = ""; this.stderr = ""; this.nextId = 1; this.pending = new Map();
  }
  get cfg() { return mcpConfig.mcpServers[this.name] || {}; }

  async start() {
    this.stop();
    const raw = this.cfg;
    const cfg = {
      ...raw, command: expandVars(raw.command), args: (raw.args || []).map(a => expandVars(String(a))),
      cwd: raw.cwd ? expandVars(raw.cwd) : undefined,
      env: Object.fromEntries(Object.entries(raw.env || {}).map(([k, v]) => [k, expandVars(String(v))])),
    };
    this.status = "starting"; this.error = null; this.stderr = ""; this.buf = ""; this.tools = [];
    // On Windows, commands like npx / npm are .cmd shims that only resolve
    // through a shell, so go through cmd.exe and quote args with spaces.
    const q = a => (/[\s"&|<>^]/.test(a) ? '"' + a.replace(/"/g, '""') + '"' : a);
    const args = (cfg.args || []).map(String);
    try {
      this.proc = spawn(IS_WIN ? q(cfg.command) : cfg.command, IS_WIN ? args.map(q) : args, {
        cwd: cfg.cwd || undefined,
        env: { ...process.env, ...(cfg.env || {}) },
        shell: IS_WIN, windowsHide: true, stdio: ["pipe", "pipe", "pipe"],
      });
    } catch (e) { this.fail(e.message); return; }
    const proc = this.proc;
    proc.on("error", e => { if (this.proc === proc) this.fail(e.message); });
    proc.on("exit", code => {
      if (this.proc !== proc || this.status === "stopped") return;
      this.fail("exited (code " + code + ")" + (this.stderr ? ": " + this.stderr.trim().slice(-300) : ""));
    });
    proc.stderr.on("data", d => { this.stderr = (this.stderr + d).slice(-4000); });
    proc.stdout.on("data", d => this.onData(d));
    proc.stdin.on("error", () => {});
    try {
      const init = await this.request("initialize", {
        protocolVersion: "2025-06-18", capabilities: {},
        clientInfo: { name: "chatbot", version: "1.0.0" },
      }, 60000);
      // servers can ship their own usage instructions; pass them to the model
      this.instructions = (init && typeof init.instructions === "string") ? init.instructions.trim() : "";
      this.notify("notifications/initialized");
      await this.refreshTools();
      this.status = "ready";
    } catch (e) { this.fail(e.message); }
  }

  async refreshTools() {
    const tools = []; let cursor;
    do {
      const r = await this.request("tools/list", cursor ? { cursor } : {}, 30000);
      tools.push(...(r.tools || []));
      cursor = r.nextCursor;
    } while (cursor);
    this.tools = tools;
  }

  onData(chunk) {
    this.buf += chunk;
    let i;
    while ((i = this.buf.indexOf("\n")) >= 0) {
      const line = this.buf.slice(0, i).trim();
      this.buf = this.buf.slice(i + 1);
      if (!line) continue;
      let msg; try { msg = JSON.parse(line); } catch (e) { continue; }   // servers sometimes log to stdout
      if (msg.id !== undefined && (msg.result !== undefined || msg.error) && this.pending.has(msg.id)) {
        const p = this.pending.get(msg.id); this.pending.delete(msg.id); clearTimeout(p.timer);
        msg.error ? p.reject(new Error(msg.error.message || "MCP error")) : p.resolve(msg.result);
      } else if (msg.method && msg.id !== undefined) {
        // server -> client request: answer ping, decline the rest
        this.send(msg.method === "ping" ? { jsonrpc: "2.0", id: msg.id, result: {} }
          : { jsonrpc: "2.0", id: msg.id, error: { code: -32601, message: "not supported by this client" } });
      } else if (msg.method === "notifications/tools/list_changed") {
        this.refreshTools().catch(() => {});
      }
    }
  }

  send(obj) { if (this.proc && this.proc.stdin.writable) this.proc.stdin.write(JSON.stringify(obj) + "\n"); }
  notify(method, params) { this.send({ jsonrpc: "2.0", method, ...(params ? { params } : {}) }); }
  request(method, params, timeout) {
    return new Promise((resolve, reject) => {
      if (!this.proc) return reject(new Error("server not running"));
      const id = this.nextId++;
      const timer = setTimeout(() => { this.pending.delete(id); reject(new Error(method + " timed out")); }, timeout || TOOL_TIMEOUT);
      this.pending.set(id, { resolve, reject, timer });
      this.send({ jsonrpc: "2.0", id, method, params });
    });
  }
  callTool(name, args) { return this.request("tools/call", { name, arguments: args || {} }, TOOL_TIMEOUT); }

  fail(msg) {
    this.status = "error"; this.error = msg;
    for (const p of this.pending.values()) { clearTimeout(p.timer); p.reject(new Error(msg)); }
    this.pending.clear();
  }
  stop() {
    const proc = this.proc;
    this.status = "stopped"; this.proc = null;
    for (const p of this.pending.values()) { clearTimeout(p.timer); p.reject(new Error("server stopped")); }
    this.pending.clear();
    if (!proc) return;
    try {
      // the shell wrapper on Windows means kill() would orphan the real server
      if (IS_WIN) spawnSync("taskkill", ["/pid", String(proc.pid), "/T", "/F"], { windowsHide: true });
      else proc.kill();
    } catch (e) { /* already gone */ }
  }
}

const mcpClients = new Map();   // name -> McpClient

function syncMcp() {
  // start enabled servers, stop disabled/removed ones; returns when all settle
  const jobs = [];
  for (const [name, c] of mcpClients) {
    const cfg = mcpConfig.mcpServers[name];
    if (!cfg || cfg.enabled === false) { c.stop(); mcpClients.delete(name); }
  }
  for (const [name, cfg] of Object.entries(mcpConfig.mcpServers)) {
    if (cfg.enabled === false || mcpClients.has(name)) continue;
    const c = new McpClient(name);
    mcpClients.set(name, c);
    jobs.push(c.start());
  }
  return Promise.all(jobs);
}

process.on("exit", () => { for (const c of mcpClients.values()) c.stop(); });
process.on("SIGINT", () => process.exit(0));
process.on("SIGTERM", () => process.exit(0));

// Anthropic tool names must match ^[a-zA-Z0-9_-]{1,64}$
let toolIndex = new Map();   // apiName -> { server, tool, def }
function toolDefs() {
  const defs = []; const index = new Map();
  for (const c of mcpClients.values()) {
    if (c.status !== "ready") continue;
    for (const t of c.tools) {
      let api = (c.name + "__" + t.name).replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64);
      while (index.has(api)) api = api.slice(0, 60) + "_" + index.size;
      index.set(api, { server: c.name, tool: t.name, def: t });
      const schema = t.inputSchema && typeof t.inputSchema === "object" ? { ...t.inputSchema } : {};
      if (schema.type !== "object") schema.type = "object";
      defs.push({ name: api, description: ((t.description || t.name) + " [via " + c.name + "]").slice(0, 4000), input_schema: schema });
    }
  }
  toolIndex = index;
  return defs;
}

function isAlwaysAllowed(server, tool) {
  const list = (mcpConfig.mcpServers[server] || {}).alwaysAllow || [];
  return list.includes(tool) || list.includes("*");
}

function toAnthropicContent(result) {
  const out = [];
  for (const c of (result && result.content) || []) {
    if (c.type === "text") out.push({ type: "text", text: c.text || "(empty)" });
    else if (c.type === "image" && c.data && /^image\/(png|jpeg|gif|webp)$/.test(c.mimeType || ""))
      out.push({ type: "image", source: { type: "base64", media_type: c.mimeType, data: c.data } });
    else if (c.type === "resource" && c.resource) out.push({ type: "text", text: c.resource.text || JSON.stringify(c.resource).slice(0, 2000) });
    else out.push({ type: "text", text: JSON.stringify(c).slice(0, 2000) });
  }
  if (!out.length && result && result.structuredContent) out.push({ type: "text", text: JSON.stringify(result.structuredContent) });
  if (!out.length) out.push({ type: "text", text: "(no output)" });
  let budget = TOOL_RESULT_MAX;
  for (const b of out) {
    if (b.type !== "text") continue;
    if (b.text.length > budget) b.text = b.text.slice(0, Math.max(0, budget)) + "\n...(truncated)";
    budget -= b.text.length;
  }
  return out;
}

const preview = (v, n) => { const s = typeof v === "string" ? v : JSON.stringify(v); return s && s.length > n ? s.slice(0, n) + "..." : (s || ""); };

// What the model is told about tools. Without this it guesses at paths,
// skips required params, or invents results.
function toolGuidance() {
  const os = require("os");
  const ready = [...mcpClients.values()].filter(c => c.status === "ready" && c.tools.length);
  const where = IS_WIN ? "Windows" : process.platform === "darwin" ? "macOS" : "Linux";
  return [
    "## Tools",
    "You can call tools from MCP servers running on the user's own computer (" + where +
      ", user \"" + os.userInfo().username + "\", home folder " + os.homedir() + ").",
    "Connected: " + ready.map(c => c.name + " (" + c.tools.length + " tools)").join(", ") + ".",
    "- Use a tool when the request needs live data or an action on this machine; otherwise just answer.",
    "- Fill in every required parameter. Use full absolute paths" + (IS_WIN ? " in Windows form, e.g. C:\\Users\\..." : "") + ".",
    "- Independent calls can go in the same turn; chain calls when one needs another's result.",
    "- If a tool returns an error, read it, fix the arguments and retry once before reporting the problem.",
    "- The user approves each call. If one is declined, don't retry it; say what you would have done.",
    "- Base answers on what tools actually returned. Never invent tool output.",
    ...ready.flatMap(c => {
      const notes = ((mcpConfig.mcpServers[c.name] || {}).notes || "").trim();
      if (!c.instructions && !notes) return [];
      return ["", "### " + c.name,
        ...(c.instructions ? ["Server instructions: " + c.instructions.slice(0, 3000)] : []),
        ...(notes ? ["The user's notes (follow these): " + notes] : [])];
    }),
  ].join("\n");
}

// The front end keeps each reply's tool trace on the message. Fold a short
// note of it back into history so follow-ups ("open the second one") work.
function withToolNotes(messages) {
  return (messages || []).map(m => {
    if (!m || m.role !== "assistant" || !Array.isArray(m.tools) || !m.tools.length) return m;
    const notes = m.tools.map(t => "- " + t.server + "/" + t.tool + " " + (t.input || "") + " -> " + t.status +
      (t.output ? ": " + String(t.output).slice(0, 300) : "")).join("\n");
    return { role: "assistant", content: (m.content || "") + "\n\n[Tools used for this reply]\n" + notes };
  });
}

// One chat turn with tools. Sessions pause when a call needs approval and
// resume through POST /api/chat/continue.
const toolSessions = new Map();

async function toolStep(sess) {
  let lastText = "";
  while (sess.rounds < MAX_TOOL_ROUNDS) {
    sess.rounds++;
    const data = await anthropicRequest(sess.system, sess.convo, sess.model, toolDefs());
    const content = data.content || [];
    sess.convo.push({ role: "assistant", content });
    lastText = content.filter(b => b.type === "text").map(b => b.text).join("");
    const uses = content.filter(b => b.type === "tool_use");
    if (!uses.length) return finishTurn(sess, lastText);

    sess.calls = uses.map(u => {
      const t = toolIndex.get(u.name);
      return { id: u.id, server: t && t.server, tool: t ? t.tool : u.name, input: u.input || {},
               missing: !t, needsApproval: !!t && !isAlwaysAllowed(t.server, t.tool) };
    });
    if (sess.calls.some(c => c.needsApproval)) {
      for (const [k, s] of toolSessions) if (Date.now() - s.ts > 15 * 60000) toolSessions.delete(k);
      const id = Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
      sess.ts = Date.now();
      toolSessions.set(id, sess);
      return {
        pending: { id, calls: sess.calls.filter(c => c.needsApproval).map(c => ({ id: c.id, server: c.server, tool: c.tool, input: c.input })) },
        tools: sess.trace,
      };
    }
    await runCalls(sess, {});
  }
  return finishTurn(sess, (lastText ? lastText + "\n\n" : "") + "(Stopped after " + MAX_TOOL_ROUNDS + " tool rounds.)");
}

async function runCalls(sess, decisions) {
  const results = [];
  let changed = false;
  for (const c of sess.calls) {
    const d = decisions[c.id];
    const entry = { server: c.server || "?", tool: c.tool, input: preview(c.input, 400) };
    if (c.missing) {
      Object.assign(entry, { status: "error", output: "unknown tool" });
      results.push({ type: "tool_result", tool_use_id: c.id, is_error: true, content: "Unknown tool: " + c.tool });
    } else if (c.needsApproval && d !== "allow" && d !== "always") {
      Object.assign(entry, { status: "denied", output: "" });
      results.push({ type: "tool_result", tool_use_id: c.id, is_error: true, content: "The user declined to run this tool." });
    } else {
      if (d === "always") {
        const cfg = mcpConfig.mcpServers[c.server];
        if (cfg) { cfg.alwaysAllow = Array.from(new Set([...(cfg.alwaysAllow || []), c.tool])); changed = true; }
      }
      try {
        const client = mcpClients.get(c.server);
        if (!client || client.status !== "ready") throw new Error(c.server + " is not running");
        const r = await client.callTool(c.tool, c.input);
        const blocks = toAnthropicContent(r);
        Object.assign(entry, { status: r.isError ? "error" : "ok", output: preview(blocks.map(b => b.text || "[" + b.type + "]").join("\n"), 600) });
        results.push({ type: "tool_result", tool_use_id: c.id, is_error: !!r.isError, content: blocks });
      } catch (e) {
        Object.assign(entry, { status: "error", output: e.message });
        results.push({ type: "tool_result", tool_use_id: c.id, is_error: true, content: "Tool failed: " + e.message });
      }
    }
    sess.trace.push(entry);
  }
  if (changed) saveMcpConfig();
  sess.convo.push({ role: "user", content: results });
  sess.calls = null;
}

function finishTurn(sess, text) {
  const reply = text || "(no text reply)";
  const toStore = [];
  if (sess.lastUser) toStore.push(sess.lastUser);
  toStore.push({ role: "assistant", content: reply });
  remember(toStore, sess.chatId, sess.projectId);
  return { reply, tools: sess.trace };
}

async function continueChat(body) {
  const sess = toolSessions.get(body.id);
  if (!sess) throw new Error("approval expired - send the message again");
  toolSessions.delete(body.id);
  await runCalls(sess, body.decisions || {});
  return toolStep(sess);
}

function mcpSummary() {
  return {
    configFile: MCP_FILE,
    claudeDesktopConfig: fs.existsSync(claudeDesktopConfigPath()),
    toolsActive: activeBackend() === "anthropic",
    servers: Object.entries(mcpConfig.mcpServers).map(([name, cfg]) => {
      const c = mcpClients.get(name);
      return {
        name, command: cfg.command, args: cfg.args || [], enabled: cfg.enabled !== false,
        notes: cfg.notes || "", instructions: c ? (c.instructions || "").slice(0, 500) : "",
        status: c ? c.status : "off", error: c ? c.error : null,
        tools: c ? c.tools.map(t => ({
          name: t.name, description: (t.description || "").slice(0, 200), alwaysAllow: isAlwaysAllowed(name, t.name),
          params: Object.keys((t.inputSchema && t.inputSchema.properties) || {}),
        })) : [],
      };
    }),
  };
}

// Accepts {"mcpServers":{...}}, {"name":{command,...}} or a bare {command,...} (needs name)
function normalizeServers(input, name) {
  if (!input || typeof input !== "object") throw new Error("expected a JSON object");
  if (input.mcpServers) return input.mcpServers;
  if (input.command) { if (!name) throw new Error("give the server a name"); return { [name]: input }; }
  return input;
}

async function addServers(servers, { enabled, overwrite }) {
  const added = [];
  for (const [name, cfg] of Object.entries(servers)) {
    if (!cfg || typeof cfg.command !== "string") continue;   // skip remote/url entries this client can't run
    if (mcpConfig.mcpServers[name] && !overwrite) continue;
    const prev = mcpConfig.mcpServers[name] || {};
    mcpConfig.mcpServers[name] = { ...portable(cfg), enabled, alwaysAllow: cfg.alwaysAllow || prev.alwaysAllow || [],
      ...((cfg.notes || prev.notes) ? { notes: cfg.notes || prev.notes } : {}) };
    const c = mcpClients.get(name); if (c) { c.stop(); mcpClients.delete(name); }
    added.push(name);
  }
  saveMcpConfig();
  await syncMcp();
  return added;
}

function readJson(req, limit) {
  return new Promise((resolve, reject) => {
    let raw = "";
    req.on("data", c => { raw += c; if (raw.length > (limit || 1e6)) req.destroy(); });
    req.on("end", () => { try { resolve(JSON.parse(raw || "{}")); } catch (e) { reject(e); } });
    req.on("error", reject);
  });
}

// Routes under /api/mcp. Returns true if handled.
function handleMcpRoute(req, res) {
  const u = new URL(req.url, "http://localhost");
  if (!u.pathname.startsWith("/api/mcp")) return false;
  const parts = u.pathname.split("/").filter(Boolean).map(decodeURIComponent);   // api, mcp, ...
  const ok = () => sendJson(res, 200, mcpSummary());
  const bad = e => sendJson(res, 400, { error: String((e && e.message) || e) });

  (async () => {
    if (req.method === "GET" && parts.length === 2) return ok();

    // POST /api/mcp/servers  { config: <json>, name? }  -> add (enabled)
    if (req.method === "POST" && parts[2] === "servers" && parts.length === 3) {
      const body = await readJson(req);
      const servers = normalizeServers(body.config, body.name);
      const added = await addServers(servers, { enabled: true, overwrite: true });
      if (!added.length) throw new Error("no runnable servers found (each needs a \"command\")");
      return ok();
    }

    // POST /api/mcp/import  -> pull everything from Claude Desktop's config, disabled
    if (req.method === "POST" && parts[2] === "import") {
      const p = claudeDesktopConfigPath();
      if (!fs.existsSync(p)) throw new Error("Claude Desktop config not found at " + p);
      const servers = normalizeServers(JSON.parse(fs.readFileSync(p, "utf8")));
      await addServers(servers, { enabled: false, overwrite: false });
      return ok();
    }

    // /api/mcp/servers/:name/(enable|disable|restart)  and  DELETE /api/mcp/servers/:name
    if (parts[2] === "servers" && parts[3]) {
      const name = parts[3];
      const cfg = mcpConfig.mcpServers[name];
      if (!cfg) throw new Error("no server named " + name);
      const action = parts[4];
      if (req.method === "DELETE" && !action) {
        delete mcpConfig.mcpServers[name];
      } else if (req.method === "POST" && (action === "enable" || action === "disable")) {
        cfg.enabled = action === "enable";
      } else if (req.method === "POST" && action === "restart") {
        const c = mcpClients.get(name); if (c) { c.stop(); mcpClients.delete(name); }
        cfg.enabled = true;
      } else if (req.method === "POST" && action === "forget-approvals") {
        cfg.alwaysAllow = [];
      } else if (req.method === "POST" && action === "notes") {
        cfg.notes = String((await readJson(req)).notes || "").trim().slice(0, 2000);
        saveMcpConfig();
        return ok();   // notes only change the prompt; no restart needed
      } else { res.writeHead(404); return res.end("not found"); }
      saveMcpConfig();
      await syncMcp();
      return ok();
    }
    res.writeHead(404); res.end("not found");
  })().catch(bad);
  return true;
}

// Full chat turn: recall -> augment -> model (+ tools) -> remember.
async function handleChat(body) {
  const messages = withToolNotes(body.messages || []);
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
  const lastUser = [...messages].reverse().find(m => m && m.role === "user");

  // Tools available and the backend can use them -> agentic turn
  if (activeBackend() === "anthropic" && toolDefs().length) {
    return toolStep({
      system: [...prefix.map(m => m.content), toolGuidance()].join("\n\n"),
      convo: sanitize(messages).filter(m => m.role !== "system"),
      model, trace: [], rounds: 0, chatId, projectId,
      lastUser: lastUser ? { role: "user", content: lastUser.content } : null,
    });
  }

  const reply = await generateReply(augmented, model);

  // store only the new turns this round (front end resends full history each call,
  // so persisting everything every time would duplicate massively).
  const toStore = [];
  if (lastUser) toStore.push(lastUser);
  toStore.push({ role: "assistant", content: reply });
  remember(toStore, chatId, projectId);

  return { reply };
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
  if (req.method === "POST" && (req.url === "/api/chat" || req.url === "/api/chat/continue")) {
    readJson(req, 5e6)
      .then(body => req.url === "/api/chat" ? handleChat(body) : continueChat(body))
      .then(out => sendJson(res, 200, out))
      .catch(err => sendJson(res, 500, { error: String((err && err.message) || err) }));
    return;
  }
  if (handleMcpRoute(req, res)) return;
  if (req.url === "/favicon.ico") { res.writeHead(204); res.end(); return; }   // no icon; skip the console 404
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
  const n = Object.values(mcpConfig.mcpServers).filter(c => c.enabled !== false).length;
  if (n) syncMcp().then(() => {
    const ready = [...mcpClients.values()].filter(c => c.status === "ready");
    console.log("mcp: " + ready.length + "/" + n + " servers ready, " + ready.reduce((a, c) => a + c.tools.length, 0) + " tools" +
      (activeBackend() === "anthropic" ? "" : " (tools only run on the Anthropic backend)"));
    for (const c of mcpClients.values()) if (c.status === "error") console.log("  mcp " + c.name + ": " + c.error);
  });
});

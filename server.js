// chatbot backend - single file, zero dependencies, no cloud, no services.
// Serves the front end from ./public, exposes POST /api/chat, and keeps its
// OWN conversation memory right here in this file.
//
//   Run:   node server.js      then open   http://localhost:8787
//
// MEMORY (self-contained):
//   - Every user message and reply is appended to memory.json next to this file.
//   - On each turn the backend searches all past turns (across every chat) for
//     the ones most relevant to the new message and injects them as context.
//   - This survives restarts. No database, no embeddings, no network.
//   - Recall is lexical (keyword overlap, length-normalized) -- it matches on
//     shared words, not paraphrase. That is the tradeoff for "runs anywhere,
//     installs nothing." Good enough to remember names, facts, decisions, code.

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

function tokenize(s) {
  return ((s || "").toLowerCase().match(/[a-z0-9]+/g)) || [];
}

// length-normalized keyword overlap: turns that share more query words score
// higher, but long turns don't win just by being long.
function scoreTurn(turnTokens, querySet) {
  if (!turnTokens.length) return 0;
  let hits = 0;
  const seen = new Set();
  for (const t of turnTokens) {
    if (querySet.has(t) && !seen.has(t)) { hits++; seen.add(t); }
  }
  if (!hits) return 0;
  return hits / Math.sqrt(turnTokens.length + 1);
}

// RECALL: find the most relevant past turns and format them as a context block.
// Memory is scoped by project. Chats inside a project can see each other;
// chats outside it cannot. Unassigned chats form their own pool. This is what
// makes a project mean something rather than being a label in the sidebar.
function inScope(turn, chatId, projectId) {
  if (turn.chatId === chatId) return true;                 // same conversation
  const tp = turn.projectId || null;
  const cp = projectId || null;
  return tp === cp;                                        // same project, or both unassigned
}

function recall(messages, chatId, projectId) {
  if (!MEMORY_ENABLED || !store.turns.length) return "";
  const lastUser = [...messages].reverse().find(m => m && m.role === "user");
  if (!lastUser || !lastUser.content) return "";
  const querySet = new Set(tokenize(lastUser.content));
  if (!querySet.size) return "";

  // skip turns whose text is already in the live history (no point re-injecting)
  const live = new Set(messages.map(m => (m && m.content) || ""));

  const scored = [];
  for (const turn of store.turns) {
    if (live.has(turn.content)) continue;
    if (!inScope(turn, chatId, projectId)) continue;
    const score = scoreTurn(tokenize(turn.content), querySet);
    if (score > 0) scored.push({ turn, score });
  }
  if (!scored.length) return "";
  scored.sort((a, b) => b.score - a.score);

  const lines = scored.slice(0, RECALL_LIMIT).map(({ turn }) => {
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

  const context = recall(messages, chatId, projectId);

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
    sendJson(res, 200, { turns: store.turns.length, chats: chats.size, file: MEMORY_FILE, backend: activeBackend() });
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
});

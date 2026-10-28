# Chatbot

A self-contained chat interface with its own conversation memory and MCP tool
support. One Node backend, zero npm dependencies, no database. Clone it, set one
environment variable, run it.

- **Memory** — remembers past conversations, scoped by project, with hybrid
  keyword + vector recall
- **Tools** — connect any local MCP server (or import them all from Claude
  Desktop); every tool call asks for your approval first
- **Fetch built in** — a bundled, zero-dependency MCP server for reading web pages
- **Portable** — the same clone runs on any machine with Node 18+; your keys,
  conversations and server list stay local and out of git

## Running it

```
node server.js
```

Then open http://localhost:8787

## Choosing a model backend

The server reads its configuration from the environment at startup and uses the
first backend that is actually set. Nothing is hardcoded and no key lives in the
repo.

| Environment variable              | Backend                              |
|-----------------------------------|--------------------------------------|
| `MODEL_API_URL` + `MODEL_API_KEY` | Any OpenAI-compatible endpoint       |
| `ANTHROPIC_API_KEY`               | Anthropic messages API               |
| `OLLAMA_URL`                      | Local Ollama, e.g. `http://localhost:11434` |
| *(none set)*                      | Mock reply explaining what to set    |

Optional: `MODEL_NAME` overrides the default model for whichever backend is
active. `BOT_NAME` sets a fallback identity when no browser is there to supply
one.

Example:

```powershell
$env:ANTHROPIC_API_KEY = "..."
node server.js
```

The startup banner reports which backend it selected, so you can confirm at a
glance that it picked up your configuration.

## Architecture

The page is a thin client. It POSTs the conversation to `/api/chat` and renders
what comes back — nothing else. Every decision about which model answers, which
provider is used and where the key lives belongs to the server.

That split is the reason the same page works unchanged on any machine: move the
clone, set a different environment variable, and the front end neither knows nor
cares.

## How the memory works

Every user message and every reply is appended to `memory.json` next to
`server.js`. On each new turn the server scores the stored turns against the
incoming message and injects the best matches as a system message before
calling the model.

It survives restarts. There is no database and no npm dependency.

### Scoping

Memory is scoped by **project**, which is what makes a project mean something
rather than being a label in the sidebar:

- Two chats in the same project can recall each other's content.
- A chat in a different project cannot see it.
- Chats with no project form their own pool.

The recalled context is labelled by origin, so the model knows whether
something came from earlier in the same conversation or from a sibling chat in
the project.

### Hybrid recall

Every turn is scored two ways and the results are fused:

- **Keyword** — BM25 over the turns in scope. Always on. Catches names, file
  paths, error strings and code exactly.
- **Vector** — cosine similarity between embeddings. On when an embedding
  provider is configured. Catches paraphrase: "what do I drive?" finds
  "my truck is a Tacoma".

Final score is `0.7 × vector + 0.3 × keyword` (keyword normalised to the best
match). A turn found by both is labelled `hybrid`; by one, `vector` or
`keyword`. With no embedding provider the vector half is simply absent and
recall is keyword-only — nothing breaks.

Embedding providers (the first one configured wins):

| Environment variable              | Provider                              | Default model            |
|-----------------------------------|---------------------------------------|--------------------------|
| `EMBED_API_URL` + `EMBED_API_KEY` | Any OpenAI-compatible `/embeddings`   | `text-embedding-3-small` |
| `VOYAGE_API_KEY`                  | Voyage AI                             | `voyage-3.5-lite`        |
| `OLLAMA_EMBED_URL`                | Local Ollama                          | `nomic-embed-text`       |
| *(none set)*                      | Keyword-only recall                   | —                        |

`EMBED_MODEL` overrides the default. Example:

```powershell
$env:VOYAGE_API_KEY = "..."
node server.js
```

Vectors are stored in `memory.vectors.json` beside `memory.json`, tagged with
the provider and model that made them. Existing memory is embedded in the
background on startup; switching provider or model discards the old vectors
and re-embeds. If the provider is down or slow, that turn falls back to
keyword recall and the chat keeps working.

### Tuning

| Variable          | Default         | Effect                                   |
|-------------------|-----------------|------------------------------------------|
| `PORT`            | 8787            | Server port                              |
| `MEMORY_FILE`     | `./memory.json` | Where recall memory is stored            |
| `STATE_FILE`      | `./state.json`  | Where interface state is stored          |
| `MEMORY_DISABLED` | unset           | Set to anything to turn memory off        |
| `RECALL_LIMIT`    | 4               | How many past turns to inject             |
| `MEMORY_MAX`      | 5000            | Turns kept on disk before oldest dropped  |
| `VECTOR_FILE`     | `./memory.vectors.json` | Where embeddings are stored       |
| `VECTOR_WEIGHT`   | 0.7             | Vector share of the fused score           |
| `VECTOR_MIN`      | 0.3             | Cosine floor for a vector hit             |
| `EMBED_TIMEOUT_MS`| 15000           | Give up on the embedding call after this  |

## Tools (MCP)

The chatbot can use local MCP servers — the same stdio servers Claude Desktop
runs. The backend starts each one as a child process on this machine and talks
to it over stdin/stdout. No npm packages, no network for the MCP side.

Open **Options → Tools (MCP)**:

- **Import from Claude Desktop** pulls every server from your Claude Desktop
  config. They come in switched off; tick the ones you want.
- **Add server** takes a pasted config: a full `{"mcpServers": {...}}` block,
  a `{"name": {command, args}}` map, or a bare `{command, args}` plus a name.
- Each server shows status, its tools, and restart / remove controls.

Every tool call stops for approval — **Deny**, **Allow once** or **Always
allow** (remembered per tool; "reset approvals" clears it). Calls show up as
expandable cards above the reply.

**Bundled: Fetch.** `mcp/fetch.js` ships with the repo — a zero-dependency
MCP server that reads web pages (HTML converted to markdown, long pages
returned in chunks the model pages through). It is added and switched on the
first time the server starts on a machine; remove it in Options and it stays
removed. Fetched pages are labelled as untrusted content, and every call still
goes through approval. It's a plain MCP server, so Claude Desktop can use it
too: `{"command": "node", "args": ["<repo>/mcp/fetch.js"]}`.

**Portable paths.** Config values can use `${HOME}`, `~`, `%VAR%` or `${VAR}`;
they're expanded when the server starts. `${APP_DIR}` is the folder this
repo is cloned into. Imported and pasted configs get this
machine's home folder rewritten to `${HOME}` automatically, so if your servers
live at the same place under your home folder on every machine, the same
`mcp.json` works everywhere:

```json
{ "mcpServers": { "my-server": { "command": "python",
  "args": ["${HOME}/mcp-servers/my-server/server.py"] } } }
```

Config is saved to `mcp.json` (gitignored: servers are installed per machine,
and `env` blocks can hold tokens). Tool use runs on the Anthropic backend;
other backends chat without tools. Only stdio servers are supported — entries
with a `url` are skipped on import.

| Variable           | Default      | Effect                                   |
|--------------------|--------------|------------------------------------------|
| `MCP_FILE`         | `./mcp.json` | Where the server list is stored          |
| `MAX_TOOL_ROUNDS`  | 10           | Tool round-trips allowed per message     |
| `TOOL_TIMEOUT_MS`  | 120000       | Give up on a single tool call after this |

Each enabled tool's description is sent with every message, so enable the
servers you're using rather than all of them.

## Interface state

Chats, projects, schedules and the settings you pick in Options persist to
`state.json` and reload when the page opens, so a refresh or a restart does not
lose your conversations.

This is kept separate from `memory.json` on purpose: `memory.json` is what the
model recalls from, `state.json` is what the interface needs to redraw itself.

## Endpoints

| Method | Path                 | Purpose                                      |
|--------|----------------------|----------------------------------------------|
| POST   | `/api/chat`          | Send a conversation, get a reply              |
| GET    | `/api/state`         | Load chats, projects, schedules, settings     |
| PUT    | `/api/state`         | Save them                                     |
| POST   | `/api/chat/continue` | Resume a turn after approving tool calls      |
| GET    | `/api/mcp`           | MCP servers, status and tools                 |
| POST   | `/api/mcp/import`    | Import servers from Claude Desktop            |
| POST   | `/api/mcp/servers`   | Add servers from a pasted config              |
| GET    | `/api/memory/stats`  | Turns, chats, backend, embedding model, vector count |
| GET    | `/api/memory/search?q=…&chatId=…&projectId=…` | What recall would surface, with score and source |

## Layout

```
server.js            the whole backend
public/chatbot.html  the interface
memory.json          created on first run, gitignored
memory.vectors.json  created when embeddings are on, gitignored
mcp/fetch.js         bundled Fetch MCP server (zero dependencies)
mcp.json             your MCP servers, gitignored
state.json           created on first run, gitignored
```

## Notes

`memory.json` and `state.json` hold the full text of every conversation in
plaintext. Both are excluded from version control. Be deliberate about where
you run this and what you tell it.
With a hosted embedding provider on, every turn is also sent to that provider
to be embedded; `OLLAMA_EMBED_URL` keeps it on the machine.

The bot name set in Options is passed to the model as a system instruction. It
shapes how the assistant refers to itself; it does not change which model is
answering, and a model asked directly will generally still say what it is.

# chatbot

A self-contained chat interface with its own conversation memory. One Node file,
zero dependencies, no database, no cloud services. Clone it, set one environment
variable, run it.

Built to be portable — the same clone runs on any machine with Node installed.

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

It survives restarts. There is no database and no embedding model.

### Scoping

Memory is scoped by **project**, which is what makes a project mean something
rather than being a label in the sidebar:

- Two chats in the same project can recall each other's content.
- A chat in a different project cannot see it.
- Chats with no project form their own pool.

The recalled context is labelled by origin, so the model knows whether
something came from earlier in the same conversation or from a sibling chat in
the project.

### The tradeoff

Scoring is **lexical** — length-normalised keyword overlap. It matches on
shared words, not on paraphrase. That is the deliberate price of running
anywhere with nothing installed. It works well for names, file paths,
decisions and code; it will miss a question worded completely differently from
the answer that would satisfy it.

### Tuning

| Variable          | Default         | Effect                                   |
|-------------------|-----------------|------------------------------------------|
| `PORT`            | 8787            | Server port                              |
| `MEMORY_FILE`     | `./memory.json` | Where recall memory is stored            |
| `STATE_FILE`      | `./state.json`  | Where interface state is stored          |
| `MEMORY_DISABLED` | unset           | Set to anything to turn memory off        |
| `RECALL_LIMIT`    | 4               | How many past turns to inject             |
| `MEMORY_MAX`      | 5000            | Turns kept on disk before oldest dropped  |

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
| GET    | `/api/memory/stats`  | Turn count, chat count, file path, backend    |

## Layout

```
server.js            the whole backend
public/chatbot.html  the interface
memory.json          created on first run, gitignored
state.json           created on first run, gitignored
```

## Notes

`memory.json` and `state.json` hold the full text of every conversation in
plaintext. Both are excluded from version control. Be deliberate about where
you run this and what you tell it.

The bot name set in Options is passed to the model as a system instruction. It
shapes how the assistant refers to itself; it does not change which model is
answering, and a model asked directly will generally still say what it is.

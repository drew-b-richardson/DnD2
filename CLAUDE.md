# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

No build step. Serve the files with Python's built-in HTTP server:

```bash
cd /Users/drew/repos/DnD && python3 -m http.server 8080
```

Then open http://localhost:8080 in a browser. Ollama must be running (`ollama serve`).

## Rebuilding the Ollama Model

After editing `Modelfile`:

```bash
ollama rm dnd && ollama create dnd -f /Users/drew/repos/DnD/Modelfile
```

## Architecture

Three files, no framework, no build toolchain:

- **`index.html`** — Static shell. Three-panel CSS Grid layout: adventure log (left), chat (center), dice roller (right). All IDs referenced by `app.js` are defined here.
- **`style.css`** — Dark fantasy theme via CSS custom properties on `:root`. All colors, fonts (Cinzel + Crimson Text via Google Fonts), and animations are defined here.
- **`app.js`** — All application logic. Single `state` object holds messages, model, system prompt, adventure log, and dice history. No modules or imports.

## Key app.js Patterns

**Ollama streaming** — `streamFromOllama()` uses `fetch` with `stream: true` against `http://localhost:11434/api/chat`. Responses are newline-delimited JSON; a `buffer` string accumulates partial chunks across `reader.read()` calls. When Ollama returns `done_reason: "load"` (model cold-starting), the function retries automatically.

**Message flow** — `sendMessage()` pushes to `state.messages`, renders the player bubble, then calls `streamFromOllama(state.messages)`. The system prompt is prepended only if `state.systemPrompt` is non-empty — the `Modelfile` is the primary source for the system prompt; `state.systemPrompt` is an optional runtime override.

**Session persistence** — Full `state` (messages, adventure log, system prompt, model) is serialized to `localStorage` under key `dnd-session-v1` after every message. On load, if a saved session exists, a resume modal is shown.

**Model preference order** — On startup, `fetchModels()` calls `/api/tags` and selects the first match from `['dnd:latest', 'dolphin-mistral:latest']`, falling back to whatever is available.

## Modelfile

`FROM dolphin-mistral` base with temperature 0.85, repeat_penalty 1.1, num_ctx 8192. The `SYSTEM` block enforces DM-only role, mandatory dice roll sequences, and D&D 5e mechanics. The system prompt is the primary lever for fixing gameplay behavior — prompt changes require `ollama rm dnd && ollama create dnd -f Modelfile`.

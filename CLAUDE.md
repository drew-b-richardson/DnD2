# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A D&D 5e simulator with two interfaces:
- **CLI** (`main.py`) — terminal-based interactive loop
- **Web** (`server.py`) — Flask API + single-page app (`index.html`)

Both use the same three subsystems: `engine.py`, `llm_client.py`, and `rag.py`.

## Running the App

**Prerequisites:** Ollama must be running locally on port 11434 with a model available. The default model name is `dnd-dm` (defined via `Modelfile`). The client auto-falls back to any available model if `dnd-dm` is missing.

```bash
# Install dependencies
pip install -r requirements.txt   # requests, chromadb, flask

# Create the custom Ollama model (one-time)
ollama create dnd-dm -f Modelfile

# Start web server (serves index.html at http://localhost:5000)
python server.py

# Or run CLI
python main.py
```

## Architecture

The key design principle is **separation of concerns**:

- **LLM (`llm_client.py`)** = Narrator only. It receives a structured prompt with game state and returns a JSON response with `narrative` (prose) and `state_changes` (mechanical deltas). The LLM never calculates numbers or applies changes itself.
- **Engine (`engine.py`)** = Rules enforcer. Owns all state mutations: HP, XP, gold, inventory, conditions, spell slots, level-ups, combat rounds, rests. The `apply_changes()` method consumes the LLM's `state_changes` dict and returns human-readable messages.
- **RAG (`rag.py`)** = Rules lookup. Retrieves relevant SRD rule snippets for each player action and injects them into the LLM prompt. Uses ChromaDB + sentence-transformers when installed; falls back to keyword scoring.
- **State** = a single JSON dict (schema in `engine.py:build_new_character`) persisted to `game_state.json`.

### LLM Response Schema

Every LLM call returns JSON with three top-level keys (defined in `llm_client.py:_SCHEMA`):
- `narrative` — all prose
- `state_changes` — mechanical deltas the engine applies
- `requires_player_roll` — signals the client to prompt for a dice roll

### Combat Flow (Web)

1. Player sends action → `POST /api/action`
2. Server calls LLM, which returns `requires_player_roll` if an attack/check is needed
3. If a roll is pending, enemy turn is **deferred**
4. Client rolls dice and sends result → `POST /api/roll-result`
5. Server applies player damage directly (does not trust LLM to do it), then runs enemy turn via `engine.enemy_attacks()`
6. Enemy turn results are returned in `enemy_turn[]` for the client to display

### Key Design Decisions

- **Player damage is applied server-side** in `roll_result()`, not by the LLM. After applying damage, the narrative prompt explicitly tells the LLM "damage already applied — do NOT set enemy_hp_deltas."
- **`_infer_roll_request()`** in `server.py` is a fallback for when the LLM forgets to request an attack roll during combat — it detects attack verbs and synthesizes a roll request.
- **History is capped at 20 entries** (`add_history` in `engine.py`); only the last 8 entries are sent to the LLM prompt.
- **`api.py`** is an earlier prototype/scratch file; the active implementation is `server.py` + `llm_client.py`.

## No Tests

There is no test suite. `api.py` at the repo root is a standalone prototype script, not a test file.

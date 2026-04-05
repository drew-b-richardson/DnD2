# D&D 5e Simulator

A local D&D 5e simulator powered by a local LLM via [Ollama](https://ollama.com). Play as a fully-statted character with a live Dungeon Master that narrates your adventure, tracks HP/XP/inventory, and runs combat.

Available as a web app or CLI.

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running

## Setup

**1. Clone and install dependencies**

```bash
pip install -r requirements.txt
```

**2. Start Ollama**

```bash
ollama serve
```

**3. Create the DM model**

```bash
ollama create dnd-dm -f Modelfile
```

This creates a custom model tuned for structured JSON responses. If you skip this step, the app will fall back to any available Ollama model automatically.

## Running

### Web interface (recommended)

```bash
python server.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

### CLI

```bash
python main.py
```

## Gameplay

On first launch you'll create a character (name, race, class, optional backstory). Progress is auto-saved to `game_state.json` and resumed on the next run.

**Commands:**

| Input | Action |
|---|---|
| Any text | Sent to the DM as your action |
| `status` | Show HP, AC, XP, location |
| `inventory` | Show items and spell slots |
| `roll <dice>` | Roll dice — e.g. `roll d20`, `roll 2d6+3` |
| `quit` | Save and exit (CLI only) |

When the DM calls for a dice roll (attacks, skill checks), the interface will prompt you to roll. Hit Enter to auto-roll or type a number to use a manual result.

## Notes

- The LLM runs locally — first response may be slow depending on hardware.
- A 16 GB+ RAM machine is recommended for the default model (`Q4_K_M` quantization, ~7–8 GB).
- To use a different model, change `DEFAULT_MODEL` in `llm_client.py`.
- Game state is stored in `game_state.json`. Delete it to start a new character.

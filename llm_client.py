"""Ollama LLM client — the Narrator.

Sends structured prompts to Ollama with format="json", enforcing a response
schema that separates creative narration from mechanical state changes.
The Python engine applies state changes; the LLM never tracks numbers directly.
"""

import json
import re
import requests

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "dnd-dm"

# ---------------------------------------------------------------------------
# Response schema (injected into system prompt so the model knows what to emit)
# ---------------------------------------------------------------------------

_SCHEMA = """\
{
  "narrative": "<vivid DM narration — all prose goes here>",
  "state_changes": {
    "player_hp_delta":    <int, negative=damage positive=healing, 0 if none>,
    "xp_gained":          <int, 0 if none>,
    "gold_delta":         <int, 0 if none>,
    "items_gained":       [<item name strings>],
    "items_lost":         [<item name strings>],
    "spell_slot_used":    <"1st"/"2nd"/etc. or null>,
    "new_location":       <"description of new location" or null>,
    "combat_started":     <bool>,
    "combat_ended":       <bool>,
    "enemies":            [{"name": str, "hp": int, "ac": int, "attack_bonus": int, "damage_dice": "e.g. d8+3"}],
    "enemy_hp_deltas":    [{"name": str, "delta": int}],
    "conditions_gained":  [<condition strings>],
    "conditions_removed": [<condition strings>],
    "short_rest":         <bool>,
    "long_rest":          <bool>
  },
  "requires_player_roll": {
    "needed":      <bool>,
    "type":        <dice notation e.g. "d20" or null>,
    "dc":          <int or null>,
    "description": <"what the roll is for" or null>,
    "damage_dice": <"player's damage dice if this is an attack, e.g. d8+3" or null>
  }
}"""

_SYSTEM_PROMPT = f"""\
You are a vivid, dramatic Dungeon Master running a D&D 5e campaign.

YOUR ONLY JOB: narrate the story. Python code handles all mechanical bookkeeping \
(HP, XP, gold, inventory). Do NOT calculate results yourself — just describe what \
happens narratively and fill in the JSON schema accurately so the code can update \
the game state.

YOU MUST respond with valid JSON matching EXACTLY this schema — no extra keys, \
no markdown fences, no prose outside the JSON:

{_SCHEMA}

RULES:
- Put ALL narration inside "narrative". Be atmospheric, specific, and dramatic.
- When a fight starts: set combat_started=true, populate enemies[] with name/hp/ac/attack_bonus/damage_dice \
  (attack_bonus = to-hit modifier e.g. 4; damage_dice = e.g. "d8+3"). \
  Do NOT include enemy attacks in the narrative — the engine rolls them automatically.
- ATTACK ROLLS (MANDATORY): Whenever the player attacks in combat, you MUST set \
  requires_player_roll.needed=true with type="d20", description like \
  "Attack roll vs <enemy> (AC <ac>)", and damage_dice set to the player's weapon damage \
  (e.g. "d8+3" for a longsword with +3 STR). Do NOT narrate whether the attack hits or \
  misses — stop before resolving and wait for the roll result.
- SKILL CHECKS / SAVING THROWS: If the action calls for a skill check or save, \
  set requires_player_roll.needed=true with the appropriate dice and DC.
- After a roll result is provided: narrate the outcome only. Do NOT set enemy_hp_deltas — \
  player damage is applied by the engine automatically on a hit.
- ENEMY ATTACKS ARE HANDLED BY THE ENGINE — do NOT narrate enemy attacks or set player_hp_delta \
  for enemy damage. The engine rolls enemy attacks after the player's turn and reports results \
  separately. Only narrate the player's action and its immediate outcome.
- Keep enemies[] accurate — list only still-living enemies with current HP.
- If nothing changes mechanically, use 0/null/[] for all state_changes fields.\
"""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class OllamaClient:
    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = OLLAMA_BASE):
        self.base_url = base_url.rstrip("/")
        self.model = self._resolve_model(model)

    def _resolve_model(self, preferred: str) -> str:
        """Return preferred model if available, else first available model."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            models = [m["name"] for m in r.json().get("models", [])]
            # Exact match first
            if preferred in models:
                return preferred
            # Prefix match (e.g. "dnd-dm" matches "dnd-dm:latest")
            for m in models:
                if m.startswith(preferred):
                    print(f"  [Using model: {m}]")
                    return m
            # Fallbacks
            for fb in ["dnd:latest", "dolphin-mistral:latest", "mistral:latest"]:
                if fb in models:
                    print(f"  ['{preferred}' not found — falling back to '{fb}']")
                    return fb
            if models:
                print(f"  ['{preferred}' not found — using '{models[0]}']")
                return models[0]
            print("  [Warning: no models found in Ollama]")
            return preferred
        except requests.exceptions.ConnectionError:
            print(f"  [Warning: cannot reach Ollama at {self.base_url}]")
            return preferred

    def narrate(self, player_action: str, game_state: dict, rules_context: str = ""):
        """Send a player action to the LLM and return a parsed response dict.

        Returns None on connection/parse failure.
        """
        prompt = self._build_prompt(player_action, game_state, rules_context)
        payload = {
            "model": self.model,
            "system": _SYSTEM_PROMPT,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.82, "top_p": 0.95, "repeat_penalty": 1.1},
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=180,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")
            return self._parse(raw)
        except requests.exceptions.Timeout:
            print("\n  [DM is thinking too hard — request timed out. Try again.]")
        except requests.exceptions.ConnectionError:
            print(f"\n  [Cannot reach Ollama at {self.base_url} — is it running?]")
        except requests.exceptions.HTTPError as e:
            print(f"\n  [Ollama HTTP error: {e}]")
        except Exception as e:
            print(f"\n  [Unexpected error: {e}]")
        return None

    def summarize_history(self, entries: list) -> str | None:
        """Condense a list of history entries into a plain-text summary paragraph."""
        lines = []
        for e in entries:
            speaker = e["speaker"]
            text = e["text"][:600]
            lines.append(f"{speaker}: {text}")

        prompt = (
            "You are summarizing a D&D 5e adventure log for use as persistent memory. "
            "Condense the following history into a single focused paragraph. "
            "Preserve: active quest objectives, important NPCs and companions met, "
            "significant decisions made, notable items acquired, past combat outcomes, "
            "and the current situation. Be specific — include names, places, and outcomes. "
            "Plain prose only, no lists, no JSON.\n\n"
            + "\n".join(lines)
            + "\n\nSummary:"
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "top_p": 0.9},
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
            return text or None
        except Exception as e:
            print(f"\n  [History summarization failed: {e}]")
            return None

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, player_action: str, state: dict, rules_context: str) -> str:
        p = state["player"]
        w = state["world"]
        c = state["combat"]

        lines = [
            "=== GAME STATE ===",
            f"Character: {p['name']}, {p['race']} {p['class']} Level {p['level']}",
            f"HP: {p['hp']['current']}/{p['hp']['max']}  AC: {p['ac']}  Gold: {p['gold']} gp",
            f"Location: {w['location']}",
            f"Conditions: {', '.join(p['conditions']) if p['conditions'] else 'none'}",
            f"Inventory: {', '.join(p['inventory'][:10])}{'…' if len(p['inventory']) > 10 else ''}",
        ]

        if p.get("spell_slots"):
            slot_strs = []
            for tier, data in p["spell_slots"].items():
                remaining = data["total"] - data["used"]
                slot_strs.append(f"{tier}:{remaining}/{data['total']}")
            lines.append(f"Spell slots: {', '.join(slot_strs)}")

        if c["active"] and c["enemies"]:
            enemy_str = ", ".join(
                f"{e['name']}(HP:{e.get('hp','?')} AC:{e.get('ac','?')})"
                for e in c["enemies"]
            )
            lines.append(f"COMBAT — Round {c['round']}  Enemies: {enemy_str}")
            lines.append(
                "COMBAT RULE: If the player's action is an attack, you MUST request "
                "a d20 attack roll (requires_player_roll.needed=true) before resolving "
                "the outcome. Do NOT skip the roll."
            )

        if rules_context:
            lines += ["", "=== RELEVANT RULES ===", rules_context]

        history = state.get("history", [])
        summaries = [h for h in history if h["speaker"] == "Summary"]
        recent = [h for h in history if h["speaker"] != "Summary"][-6:]
        display = summaries + recent
        if display:
            lines.append("\n=== ADVENTURE HISTORY ===")
            for h in display:
                if h["speaker"] == "Summary":
                    lines.append(f"[Adventure so far]: {h['text']}")
                else:
                    lines.append(f"{h['speaker']}: {h['text'][:300]}")

        lines += ["", "=== PLAYER ACTION ===", player_action, "", "Respond with JSON only:"]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse(self, raw: str):
        """Parse JSON from model output, with a regex-extraction fallback."""
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # Model sometimes wraps JSON in markdown fences or adds prose
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        print(f"\n  [DM returned unparseable response — raw excerpt: {raw[:200]}]")
        return None

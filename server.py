"""Web server — wraps the game engine in a Flask API."""

import json
import os
import re

from flask import Flask, jsonify, request, send_file

from engine import GameEngine, build_new_character
from llm_client import OllamaClient
from rag import RulesRAG

SAVE_FILE = "game_state.json"

_ATTACK_WORDS = re.compile(
    r'\b(attack|swing|strike|stab|slash|shoot|fire|cast|hit|charge|bash|thrust|smite|lunge)\b',
    re.IGNORECASE,
)


_WEAPON_DAMAGE = {
    "longsword": "d8", "shortsword": "d6", "greatsword": "2d6",
    "greataxe": "d12", "handaxe": "d6", "battleaxe": "d8",
    "rapier": "d8", "dagger": "d4", "quarterstaff": "d6",
    "mace": "d6", "warhammer": "d8", "flail": "d8",
    "greatclub": "d8", "scimitar": "d6",
    "longbow": "d8", "shortbow": "d6",
    "crossbow": "d8", "light crossbow": "d8",
    "dart": "d4", "javelin": "d6", "spear": "d6",
}

_CLASS_DAMAGE = {
    "Barbarian": "d12", "Fighter": "d8", "Paladin": "d8",
    "Ranger": "d8", "Rogue": "d6", "Monk": "d6",
    "Cleric": "d6", "Druid": "d6", "Bard": "d6",
    "Warlock": "d6", "Sorcerer": "d6", "Wizard": "d6",
}


def _infer_damage_dice(state: dict) -> str:
    """Guess the player's damage dice from inventory then class."""
    inventory = state["player"].get("inventory", [])
    for item in inventory:
        item_lower = item.lower()
        for weapon, dice in _WEAPON_DAMAGE.items():
            if weapon in item_lower:
                return dice
    return _CLASS_DAMAGE.get(state["player"].get("class", "Fighter"), "d6")


def _extract_target(desc: str, state: dict) -> str:
    """Parse enemy name from roll description like 'Attack roll vs Goblin (AC 13)'."""
    m = re.search(r'\bvs\s+(.+?)\s*(?:\(AC|\(ac|$)', desc, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fall back to first living enemy
    for e in state["combat"].get("enemies", []):
        if e.get("hp", 0) > 0:
            return e["name"]
    return ""


def _run_enemy_turn(eng: GameEngine) -> list:
    """Run enemy attacks and return structured results for the client."""
    return eng.enemy_attacks()


def _infer_roll_request(action: str, state: dict) -> dict | None:
    """Return a fallback roll request when LLM forgot to ask for one during combat."""
    if not state["combat"]["active"]:
        return None
    if not _ATTACK_WORDS.search(action):
        return None
    enemies = state["combat"]["enemies"]
    if enemies:
        target = enemies[0]
        ac = target.get("ac")
        desc = f"Attack roll vs {target['name']} (AC {ac})"
    else:
        ac = None
        desc = "Attack roll"
    return {
        "needed": True, "type": "d20", "dc": ac,
        "description": desc,
        "damage_dice": _infer_damage_dice(state),
    }


RACES = ["Human", "Elf", "Half-Elf", "Dwarf", "Halfling", "Gnome",
         "Half-Orc", "Tiefling", "Dragonborn", "Aasimar"]
CLASSES = ["Barbarian", "Bard", "Cleric", "Druid", "Fighter", "Monk",
           "Paladin", "Ranger", "Rogue", "Sorcerer", "Warlock", "Wizard"]

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Boot subsystems once at startup
# ---------------------------------------------------------------------------

print("Booting RAG...")
rag = RulesRAG()
print("Booting LLM client...")
llm = OllamaClient()

# Global mutable game state
engine: GameEngine | None = None
state: dict | None = None


def _load_save():
    global engine, state
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE) as f:
            state = json.load(f)
        engine = GameEngine(state)
        return True
    return False


_load_save()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return send_file("index.html")


@app.get("/api/state")
def get_state():
    if state is None:
        return jsonify({"has_save": False, "races": RACES, "classes": CLASSES})
    return jsonify({"has_save": True, "state": state})


@app.post("/api/new-character")
def new_character():
    global engine, state
    data = request.json or {}
    name = data.get("name", "Adventurer")
    race = data.get("race", "Human")
    char_class = data.get("class", "Fighter")
    backstory = data.get("backstory", "")

    state = build_new_character(name, race, char_class, backstory)
    engine = GameEngine(state)

    # Opening narration
    p = state["player"]
    opening_prompt = (
        f"A new adventure begins for {p['name']}, a {p['race']} {p['class']}. "
        f"Backstory: {p.get('backstory') or 'a wandering adventurer seeking fortune'}. "
        f"Set the opening scene at {state['world']['location']} on a "
        f"{state['world']['weather']} evening. "
        "Be vivid and atmospheric. End with an immediate hook or choice."
    )
    response = llm.narrate(opening_prompt, state, "")
    narrative = ""
    if response:
        narrative = response.get("narrative", "")
        engine.apply_changes(response.get("state_changes", {}))
        engine.add_history("DM", narrative)
        engine.save(SAVE_FILE)

    return jsonify({"state": state, "narrative": narrative})


@app.post("/api/load")
def load_game():
    if _load_save():
        return jsonify({"state": state})
    return jsonify({"error": "No save file found"}), 404


@app.delete("/api/save")
def delete_save():
    global engine, state
    if os.path.exists(SAVE_FILE):
        os.remove(SAVE_FILE)
    engine = None
    state = None
    return jsonify({"ok": True})


@app.post("/api/action")
def action():
    global engine, state
    if engine is None or state is None:
        return jsonify({"error": "No active game"}), 400

    data = request.json or {}
    raw = data.get("action", "").strip()
    if not raw:
        return jsonify({"error": "Empty action"}), 400

    cmd = raw.lower()

    # Meta commands handled server-side
    if cmd == "status":
        return jsonify({"type": "status", "state": state})

    if cmd == "inventory":
        return jsonify({"type": "inventory", "state": state})

    if cmd.startswith("roll "):
        notation = raw[5:].strip()
        result = engine.dice.roll(notation)
        if result:
            return jsonify({"type": "roll", "result": result})
        return jsonify({"type": "roll", "error": f"Unknown notation '{notation}'"})

    # Player action — sent to LLM
    player_action = raw
    if state["combat"]["active"]:
        player_action = engine.combat_context(player_action)

    rules_context = rag.query(raw)
    response = llm.narrate(player_action, state, rules_context)

    if not response:
        return jsonify({"error": "The DM seems distracted. Try rephrasing your action."}), 502

    narrative = response.get("narrative", "")
    changes = response.get("state_changes", {})
    msgs = engine.apply_changes(changes)
    roll_req = response.get("requires_player_roll") or {}

    # Fallback: LLM forgot to request a roll for a combat attack
    if not roll_req.get("needed"):
        fallback = _infer_roll_request(raw, state)
        if fallback:
            roll_req = fallback

    engine.add_history("Player", raw)
    engine.add_history("DM", narrative)

    # Run enemy attacks at end of player turn (only when no roll is pending)
    enemy_turn = []
    if not roll_req.get("needed") and state["combat"]["active"]:
        enemy_turn = _run_enemy_turn(engine)

    engine.save(SAVE_FILE)

    return jsonify({
        "type": "narrative",
        "narrative": narrative,
        "messages": msgs,
        "roll_request": roll_req if roll_req.get("needed") else None,
        "rules_context": rules_context,
        "enemy_turn": enemy_turn,
        "state": state,
    })


@app.post("/api/roll-result")
def roll_result():
    if engine is None or state is None:
        return jsonify({"error": "No active game"}), 400

    data = request.json or {}
    roll_total   = data.get("roll_total", 10)
    raw_d20      = data.get("raw_d20")
    desc         = data.get("description", "roll")
    dc           = data.get("dc")
    damage_dice  = data.get("damage_dice")
    rules_context = data.get("rules_context", "")

    # Determine hit/miss — nat 1 always misses, nat 20 always hits
    if raw_d20 == 1:
        success = False
        verdict = "AUTOMATIC MISS (natural 1)."
    elif raw_d20 == 20:
        success = True
        verdict = "CRITICAL HIT (natural 20)!"
    elif dc is not None:
        success = roll_total >= dc
        verdict = f"{'HIT' if success else 'MISS'} — rolled {roll_total} vs AC {dc}."
    else:
        success = None
        verdict = f"Rolled {roll_total}."

    # Apply player damage server-side on a hit — don't trust the LLM to do it
    hit_msgs = []
    damage_done = 0
    if success and state["combat"]["active"]:
        dice = damage_dice or _infer_damage_dice(state)
        dmg = engine.dice.roll(dice)
        if dmg:
            damage_done = dmg["total"]
            damage_str  = dmg["breakdown"]
            if raw_d20 == 20:          # crit: double the dice
                dmg2 = engine.dice.roll(dice)
                if dmg2:
                    damage_done += dmg2["total"]
                    damage_str  += "+" + dmg2["breakdown"] + " (crit)"
            target = _extract_target(desc, state)
            engine._apply_enemy_damage(target, -damage_done, hit_msgs)
            verdict += f" Deals {damage_done} damage ({damage_str})."

    outcome_prompt = (
        f"The player rolled {roll_total} for: {desc}. {verdict} "
        + ("Narrate the hit vividly. Do NOT set enemy_hp_deltas — damage already applied."
           if success else
           "Narrate the miss — the attack does NOT connect.")
        if success is not None else
        f"The player rolled {roll_total} for: {desc}. Narrate the outcome."
    )

    response = llm.narrate(outcome_prompt, state, rules_context)
    if not response:
        return jsonify({"error": "No response from DM"}), 502

    narrative = response.get("narrative", "")
    changes = response.get("state_changes", {})
    # Strip enemy_hp_deltas — we applied damage already
    changes.pop("enemy_hp_deltas", None)
    msgs = engine.apply_changes(changes)
    msgs = hit_msgs + msgs
    engine.add_history("DM", narrative)

    # Run enemy attacks after player's full turn (action + roll) is resolved
    enemy_turn = []
    if state["combat"]["active"]:
        enemy_turn = _run_enemy_turn(engine)

    engine.save(SAVE_FILE)

    return jsonify({"narrative": narrative, "messages": msgs, "enemy_turn": enemy_turn, "state": state})


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=False, port=5000, threaded=True)

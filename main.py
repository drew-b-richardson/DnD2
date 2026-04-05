#!/usr/bin/env python3
"""D&D 5e Simulator — main entry point.

The Brain vs. The Books:
  LLM      = Narrator (dialogue, descriptions, drama)
  engine   = Rules Lawyer (HP, XP, dice, combat state)
  rag      = The Handbook (pulls relevant rules per action)
  state    = The World (JSON ground-truth saved to disk)
"""

import json
import os
import sys

from engine import GameEngine, build_new_character
from llm_client import OllamaClient
from rag import RulesRAG

SAVE_FILE = "game_state.json"

RACES = ["Human", "Elf", "Half-Elf", "Dwarf", "Halfling", "Gnome",
         "Half-Orc", "Tiefling", "Dragonborn", "Aasimar"]
CLASSES = ["Barbarian", "Bard", "Cleric", "Druid", "Fighter", "Monk",
           "Paladin", "Ranger", "Rogue", "Sorcerer", "Warlock", "Wizard"]


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _pick(prompt: str, options: list, default: str) -> str:
    print(f"\n  Options: {', '.join(options)}")
    value = input(f"  {prompt} [{default}]: ").strip()
    # Accept partial, case-insensitive match
    if not value:
        return default
    for opt in options:
        if opt.lower().startswith(value.lower()):
            return opt
    return value  # allow free-text if nothing matches


def _hr(char="─", width=54):
    print(char * width)


def display_status(state: dict):
    p = state["player"]
    hp = p["hp"]
    _hr("═")
    print(f"  {p['name']}  |  {p['race']} {p['class']}  Lv.{p['level']}")
    print(f"  HP {hp['current']}/{hp['max']}   AC {p['ac']}   "
          f"XP {p['xp']}/{p['xp_to_next']}   Gold {p['gold']} gp")
    print(f"  {state['world']['location']}")
    if p["conditions"]:
        print(f"  Conditions: {', '.join(p['conditions'])}")
    if state["combat"]["active"]:
        enemies = state["combat"]["enemies"]
        names = ", ".join(f"{e['name']}(HP:{e.get('hp','?')})" for e in enemies)
        print(f"  ⚔  COMBAT Round {state['combat']['round']}  |  {names}")
    _hr("═")
    print()


def display_inventory(state: dict):
    p = state["player"]
    _hr()
    print(f"  Inventory — {p['name']}")
    _hr()
    for item in p["inventory"]:
        print(f"    {item}")
    if p.get("spell_slots"):
        print()
        for tier, data in p["spell_slots"].items():
            remaining = data["total"] - data["used"]
            bar = "●" * remaining + "○" * data["used"]
            print(f"    Spell slots {tier}: {bar}  ({remaining}/{data['total']})")
    _hr()
    print()


# ---------------------------------------------------------------------------
# Character creation
# ---------------------------------------------------------------------------

def character_creation() -> dict:
    print()
    _hr("═")
    print("  CREATE YOUR CHARACTER")
    _hr("═")

    name = input("\n  Character name [Adventurer]: ").strip() or "Adventurer"
    race = _pick("Race", RACES, "Human")
    char_class = _pick("Class", CLASSES, "Fighter")
    backstory = input("\n  One-sentence backstory (optional): ").strip()

    state = build_new_character(name, race, char_class, backstory)

    print()
    _hr()
    print(f"  {name} the {race} {char_class} is ready.")
    print(f"  HP: {state['player']['hp']['max']}   "
          f"AC: {state['player']['ac']}   Gold: {state['player']['gold']} gp")
    _hr()
    return state


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    print()
    _hr("═")
    print("  D&D 5e SIMULATOR")
    print("  Powered by local LLM via Ollama")
    _hr("═")

    # Boot subsystems
    print()
    rag = RulesRAG()
    llm = OllamaClient()

    # Load or create state
    print()
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE) as f:
            saved = json.load(f)
        sp = saved["player"]
        choice = input(
            f"  Found save: {sp['name']} — Lv.{sp['level']} {sp['class']}  "
            f"(HP {sp['hp']['current']}/{sp['hp']['max']}).  Resume? [Y/n]: "
        ).strip().lower()
        state = saved if choice != "n" else character_creation()
    else:
        state = character_creation()

    engine = GameEngine(state)

    print()
    _hr()
    print("  Commands:  status  |  inventory  |  roll <dice>  |  quit")
    print("  Just type your action to play.")
    _hr()
    print()

    display_status(state)

    # Opening narration for new games
    if not state["history"]:
        p = state["player"]
        opening_prompt = (
            f"A new adventure begins for {p['name']}, a {p['race']} {p['class']}. "
            f"Backstory: {p.get('backstory') or 'a wandering adventurer seeking fortune'}. "
            f"Set the opening scene at {state['world']['location']} on a "
            f"{state['world']['weather']} evening. "
            "Be vivid and atmospheric. End with an immediate hook or choice."
        )
        print("DM: ", end="", flush=True)
        response = llm.narrate(opening_prompt, state, "")
        if response:
            narrative = response.get("narrative", "")
            print(narrative)
            engine.apply_changes(response.get("state_changes", {}))
            engine.add_history("DM", narrative)
            engine.save(SAVE_FILE)
        print()

    # Main loop
    while True:
        try:
            raw_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nFarewell, adventurer. Your progress has been saved.")
            engine.save(SAVE_FILE)
            sys.exit(0)

        if not raw_input:
            continue

        cmd = raw_input.lower()

        # ── Meta-commands ──────────────────────────────────────────────
        if cmd == "quit":
            engine.save(SAVE_FILE)
            print("Progress saved. May your next adventure be legendary.")
            break

        if cmd == "status":
            display_status(state)
            continue

        if cmd == "inventory":
            display_inventory(state)
            continue

        if cmd.startswith("roll "):
            notation = raw_input[5:].strip()
            result = engine.dice.roll(notation)
            if result:
                print(f"  {result['notation'].upper()}: {result['total']}  {result['breakdown']}")
            else:
                print(f"  Unknown notation '{notation}'. Try: d20, 2d6+3, d8, etc.")
            continue

        if cmd == "help":
            print("  status       — show HP, AC, XP, location")
            print("  inventory    — show items and spell slots")
            print("  roll <dice>  — roll dice (e.g. roll d20, roll 2d6+3)")
            print("  quit         — save and exit")
            print("  Anything else is sent to the DM as your action.")
            continue

        # ── Player action ──────────────────────────────────────────────
        player_action = raw_input
        if state["combat"]["active"]:
            player_action = engine.combat_context(player_action)

        rules_context = rag.query(raw_input)

        print("\nDM: ", end="", flush=True)
        response = llm.narrate(player_action, state, rules_context)

        if not response:
            print("[The DM seems distracted. Try rephrasing your action.]\n")
            continue

        narrative = response.get("narrative", "")
        print(narrative)

        # Apply mechanical changes
        changes = response.get("state_changes", {})
        msgs = engine.apply_changes(changes)
        if msgs:
            print()
            for msg in msgs:
                print(f"  [{msg}]")

        # Player roll request
        roll_req = response.get("requires_player_roll") or {}
        if roll_req.get("needed"):
            dice_type = roll_req.get("type", "d20")
            dc = roll_req.get("dc")
            desc = roll_req.get("description", "Roll required")

            print(f"\n  >> {desc}")
            if dc:
                print(f"     DC: {dc}")

            raw_roll = input(f"     Press Enter to roll {dice_type}, or type a number: ").strip()

            if raw_roll.lstrip("-").isdigit():
                roll_total = int(raw_roll)
                print(f"     You rolled: {roll_total}")
            else:
                result = engine.dice.roll(dice_type)
                roll_total = result["total"] if result else 10
                breakdown = result["breakdown"] if result else ""
                print(f"     Rolled {dice_type}: {roll_total}  {breakdown}")

            # Feed result back to DM
            outcome_prompt = (
                f"The player rolled {roll_total} for: {desc}. "
                f"{'DC was ' + str(dc) + '. ' if dc else ''}"
                f"{'Success!' if (not dc or roll_total >= dc) else 'Failure.'} "
                "Narrate the outcome."
            )
            print("\nDM: ", end="", flush=True)
            followup = llm.narrate(outcome_prompt, state, rules_context)
            if followup:
                fnarrative = followup.get("narrative", "")
                print(fnarrative)
                fmsgs = engine.apply_changes(followup.get("state_changes", {}))
                if fmsgs:
                    print()
                    for msg in fmsgs:
                        print(f"  [{msg}]")
                engine.add_history("DM", fnarrative)

        # Save history and state
        engine.add_history("Player", raw_input)
        engine.add_history("DM", narrative)
        engine.save(SAVE_FILE)

        # Auto-display status after significant events
        significant = (
            changes.get("player_hp_delta")
            or changes.get("xp_gained")
            or changes.get("combat_started")
            or changes.get("combat_ended")
            or changes.get("long_rest")
        )
        if significant:
            print()
            display_status(state)
        else:
            print()


if __name__ == "__main__":
    main()

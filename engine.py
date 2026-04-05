"""Game engine — the Rules Lawyer.

Handles all mechanical bookkeeping so the LLM only needs to narrate:
  - Dice rolling (any NdX+M notation)
  - State mutations (HP, XP, gold, inventory, conditions, spell slots)
  - Level-up thresholds
  - Combat tracking (enemies, rounds)
  - Short/long rest recovery
  - Persistent save/load
"""

import json
import random
import re


# ---------------------------------------------------------------------------
# Dice Roller
# ---------------------------------------------------------------------------

_DICE_RE = re.compile(r'^(\d*)d(\d+)([+-]\d+)?$', re.IGNORECASE)


class DiceRoller:
    def roll(self, notation: str):
        """Parse and roll dice notation like '2d6+3', 'd20', '1d8-1'.

        Returns a dict with keys: notation, rolls, modifier, total, breakdown.
        Returns None if notation is invalid.
        """
        notation = notation.strip().lower()
        m = _DICE_RE.match(notation)
        if not m:
            return None

        count = int(m.group(1)) if m.group(1) else 1
        sides = int(m.group(2))
        modifier = int(m.group(3)) if m.group(3) else 0

        if count < 1 or count > 20 or sides < 2 or sides > 100:
            return None

        rolls = [random.randint(1, sides) for _ in range(count)]
        total = sum(rolls) + modifier

        parts = "+".join(str(r) for r in rolls)
        breakdown = f"({parts})"
        if modifier:
            sign = "+" if modifier > 0 else ""
            breakdown += f"{sign}{modifier}"

        return {
            "notation": notation,
            "rolls": rolls,
            "modifier": modifier,
            "total": total,
            "breakdown": breakdown,
        }

    def roll_attack(self, proficiency_bonus=2, stat_mod=3):
        """d20 + proficiency + stat mod. Returns hit info including crit/fumble."""
        d20 = self.roll("d20")
        natural = d20["rolls"][0]
        total = natural + proficiency_bonus + stat_mod
        return {
            "natural": natural,
            "total": total,
            "crit": natural == 20,
            "fumble": natural == 1,
            "breakdown": f"d20({natural}) +prof({proficiency_bonus}) +mod({stat_mod}) = {total}",
        }


# ---------------------------------------------------------------------------
# Hit dice and starting inventory by class
# ---------------------------------------------------------------------------

_HIT_DICE = {
    "Fighter": "d10", "Paladin": "d10",
    "Ranger": "d10",
    "Wizard": "d6",
    "Rogue": "d8", "Cleric": "d8", "Bard": "d8", "Druid": "d8",
    "Sorcerer": "d6", "Warlock": "d8", "Monk": "d8", "Barbarian": "d12",
}

_HP_BONUS = {
    "Fighter": 10, "Paladin": 10, "Barbarian": 12,
    "Ranger": 10,
    "Wizard": 6, "Sorcerer": 6,
    "Rogue": 8, "Cleric": 8, "Bard": 8, "Druid": 8, "Warlock": 8, "Monk": 8,
}

# XP required to *reach* each level (index = level)
_XP_TABLE = [0, 0, 300, 900, 2700, 6500, 14000, 23000, 34000, 48000, 64000,
             85000, 100000, 120000, 140000, 165000, 195000, 225000, 265000, 305000, 355000]


# ---------------------------------------------------------------------------
# Game Engine
# ---------------------------------------------------------------------------

class GameEngine:
    """Manages state mutations and persistence."""

    def __init__(self, state: dict):
        self.state = state
        self.dice = DiceRoller()

    # ------------------------------------------------------------------
    # State mutations driven by LLM response
    # ------------------------------------------------------------------

    def apply_changes(self, changes: dict) -> list:
        """Apply a state_changes dict from an LLM response.

        Returns a list of human-readable status strings describing what changed.
        """
        msgs = []
        if not changes:
            return msgs

        p = self.state["player"]

        # ---- HP ----
        hp_delta = changes.get("player_hp_delta", 0)
        if hp_delta:
            old = p["hp"]["current"]
            p["hp"]["current"] = max(0, min(p["hp"]["max"], old + hp_delta))
            actual = p["hp"]["current"] - old
            if actual < 0:
                msgs.append(f"You take {abs(actual)} damage!  HP: {p['hp']['current']}/{p['hp']['max']}")
            elif actual > 0:
                msgs.append(f"You recover {actual} HP.  HP: {p['hp']['current']}/{p['hp']['max']}")

            if p["hp"]["current"] == 0:
                msgs.append("You fall unconscious! Death saving throws begin next turn.")
                if "unconscious" not in p["conditions"]:
                    p["conditions"].append("unconscious")

        # ---- XP ----
        xp = changes.get("xp_gained", 0)
        if xp:
            p["xp"] += xp
            msgs.append(f"+{xp} XP  ({p['xp']}/{p['xp_to_next']})")
            lvl_msg = self._check_level_up()
            if lvl_msg:
                msgs.append(lvl_msg)

        # ---- Gold ----
        gold = changes.get("gold_delta", 0)
        if gold:
            p["gold"] = max(0, p["gold"] + gold)
            sign = "+" if gold >= 0 else ""
            msgs.append(f"{sign}{gold} gp  (Total: {p['gold']} gp)")

        # ---- Inventory ----
        for item in changes.get("items_gained", []):
            p["inventory"].append(item)
            msgs.append(f"Added to inventory: {item}")

        for item in changes.get("items_lost", []):
            lower = [i.lower() for i in p["inventory"]]
            if item.lower() in lower:
                removed = p["inventory"].pop(lower.index(item.lower()))
                msgs.append(f"Removed from inventory: {removed}")

        # ---- Spell slots ----
        slot_used = changes.get("spell_slot_used")
        if slot_used and slot_used in p.get("spell_slots", {}):
            slot = p["spell_slots"][slot_used]
            if slot["used"] < slot["total"]:
                slot["used"] += 1
                remaining = slot["total"] - slot["used"]
                msgs.append(f"{slot_used}-level slot used  ({remaining} remaining)")

        # ---- Conditions ----
        for cond in changes.get("conditions_gained", []):
            if cond not in p["conditions"]:
                p["conditions"].append(cond)
                msgs.append(f"Condition gained: {cond}")

        for cond in changes.get("conditions_removed", []):
            if cond in p["conditions"]:
                p["conditions"].remove(cond)
                msgs.append(f"Condition removed: {cond}")

        # ---- Location ----
        new_loc = changes.get("new_location")
        if new_loc:
            self.state["world"]["location"] = new_loc
            msgs.append(f"Location: {new_loc}")

        # ---- Combat start ----
        if changes.get("combat_started") and not self.state["combat"]["active"]:
            enemies = changes.get("enemies", [])
            self.state["combat"]["active"] = True
            self.state["combat"]["enemies"] = enemies
            self.state["combat"]["round"] = 1
            names = ", ".join(e.get("name", "enemy") for e in enemies)
            msgs.append(f"COMBAT STARTED — Round 1  |  Enemies: {names}")

        # ---- Enemy damage ----
        for entry in changes.get("enemy_hp_deltas", []):
            self._apply_enemy_damage(entry.get("name", ""), entry.get("delta", 0), msgs)

        # ---- Combat end ----
        if changes.get("combat_ended") and self.state["combat"]["active"]:
            self.state["combat"]["active"] = False
            self.state["combat"]["enemies"] = []
            self.state["combat"]["round"] = 0
            msgs.append("Combat ended.")

        # ---- Rests ----
        if changes.get("short_rest"):
            msgs.extend(self._short_rest())
        if changes.get("long_rest"):
            msgs.extend(self._long_rest())

        return msgs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_level_up(self):
        p = self.state["player"]
        lv = p["level"]
        if lv >= 20:
            return None
        next_xp = _XP_TABLE[lv + 1]
        if p["xp"] >= next_xp:
            p["level"] += 1
            p["xp_to_next"] = _XP_TABLE[p["level"] + 1] if p["level"] < 20 else 999999
            hp_gain = _HP_BONUS.get(p["class"], 8)
            p["hp"]["max"] += hp_gain
            p["hp"]["current"] = p["hp"]["max"]
            return (f"*** LEVEL UP! You are now level {p['level']}! ***  "
                    f"Max HP: {p['hp']['max']}")
        return None

    def _apply_enemy_damage(self, name: str, delta: int, msgs: list):
        for enemy in self.state["combat"]["enemies"]:
            if enemy.get("name", "").lower() == name.lower():
                enemy["hp"] = max(0, enemy.get("hp", 10) + delta)
                if enemy["hp"] == 0:
                    msgs.append(f"{enemy['name']} has been defeated!")
                    self.state["combat"]["enemies"] = [
                        e for e in self.state["combat"]["enemies"] if e.get("hp", 1) > 0
                    ]
                    if not self.state["combat"]["enemies"]:
                        self.state["combat"]["active"] = False
                        self.state["combat"]["round"] = 0
                        msgs.append("All enemies defeated! Combat ends.")
                else:
                    msgs.append(f"{enemy['name']} HP: {enemy['hp']}")
                return

    def _short_rest(self):
        p = self.state["player"]
        die = _HIT_DICE.get(p["class"], "d8")
        roll = self.dice.roll(die)
        con_mod = (p["stats"]["con"] - 10) // 2
        healed = max(1, roll["total"] + con_mod)
        p["hp"]["current"] = min(p["hp"]["max"], p["hp"]["current"] + healed)
        return [f"Short rest: {die}={roll['total']} +CON({con_mod}) → +{healed} HP  "
                f"({p['hp']['current']}/{p['hp']['max']})"]

    def _long_rest(self):
        p = self.state["player"]
        p["hp"]["current"] = p["hp"]["max"]
        for slot in p.get("spell_slots", {}).values():
            slot["used"] = 0
        removable = {"exhausted", "poisoned", "frightened"}
        p["conditions"] = [c for c in p["conditions"] if c not in removable]
        return ["Long rest complete — HP fully restored, spell slots recovered."]

    # ------------------------------------------------------------------
    # Enemy turn
    # ------------------------------------------------------------------

    def enemy_attacks(self) -> list:
        """Roll attacks for every living enemy against the player.

        Applies damage directly to player HP. Advances the round counter.
        Returns a list of result dicts for display / LLM narration.
        """
        c = self.state["combat"]
        if not c["active"]:
            return []

        p = self.state["player"]
        results = []

        for enemy in c["enemies"]:
            if enemy.get("hp", 0) <= 0:
                continue

            attack_bonus = enemy.get("attack_bonus", 4)
            damage_dice  = enemy.get("damage_dice", "d6")

            d20   = self.dice.roll("d20")
            nat   = d20["rolls"][0]
            total = nat + attack_bonus
            crit   = nat == 20
            fumble = nat == 1
            hit    = crit or (not fumble and total >= p["ac"])

            damage = 0
            damage_str = ""
            if hit:
                dmg = self.dice.roll(damage_dice)
                if crit:
                    dmg2 = self.dice.roll(damage_dice)
                    damage = dmg["total"] + dmg2["total"]
                    damage_str = f"{dmg['breakdown']}+{dmg2['breakdown']} (crit)"
                else:
                    damage = dmg["total"]
                    damage_str = dmg["breakdown"]

                p["hp"]["current"] = max(0, p["hp"]["current"] - damage)
                if p["hp"]["current"] == 0 and "unconscious" not in p["conditions"]:
                    p["conditions"].append("unconscious")

            results.append({
                "name":        enemy["name"],
                "nat":         nat,
                "total":       total,
                "attack_bonus": attack_bonus,
                "player_ac":   p["ac"],
                "hit":         hit,
                "crit":        crit,
                "fumble":      fumble,
                "damage":      damage,
                "damage_str":  damage_str,
                "player_hp":   p["hp"]["current"],
                "player_hp_max": p["hp"]["max"],
            })

        # Advance round after all enemies have acted
        if c["active"]:
            c["round"] += 1

        return results

    # ------------------------------------------------------------------
    # Combat context injection
    # ------------------------------------------------------------------

    def combat_context(self, player_input: str) -> str:
        """Append current combat status to a player action string."""
        c = self.state["combat"]
        if not c["active"] or not c["enemies"]:
            return player_input
        enemy_str = ", ".join(
            f"{e.get('name','?')}(HP:{e.get('hp','?')},AC:{e.get('ac','?')})"
            for e in c["enemies"]
        )
        return f"{player_input} [Round {c['round']} — Enemies: {enemy_str}]"

    # ------------------------------------------------------------------
    # History and persistence
    # ------------------------------------------------------------------

    def add_history(self, speaker: str, text: str, max_entries=20):
        self.state["history"].append({"speaker": speaker, "text": text})
        if len(self.state["history"]) > max_entries:
            self.state["history"] = self.state["history"][-max_entries:]

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.state, f, indent=2)


# ---------------------------------------------------------------------------
# Character creation helpers (used by main.py)
# ---------------------------------------------------------------------------

_CLASS_DEFAULTS = {
    "Fighter":   {"hp": 12, "ac": 16, "stats": {"str": 16, "dex": 12, "con": 15, "int": 10, "wis": 12, "cha": 10}},
    "Barbarian": {"hp": 15, "ac": 14, "stats": {"str": 17, "dex": 13, "con": 16, "int": 8,  "wis": 11, "cha": 9}},
    "Wizard":    {"hp": 6,  "ac": 12, "stats": {"str": 8,  "dex": 14, "con": 12, "int": 17, "wis": 13, "cha": 10}},
    "Sorcerer":  {"hp": 6,  "ac": 12, "stats": {"str": 8,  "dex": 14, "con": 12, "int": 12, "wis": 10, "cha": 17}},
    "Rogue":     {"hp": 8,  "ac": 14, "stats": {"str": 10, "dex": 17, "con": 13, "int": 12, "wis": 12, "cha": 14}},
    "Cleric":    {"hp": 8,  "ac": 18, "stats": {"str": 14, "dex": 10, "con": 13, "int": 10, "wis": 16, "cha": 12}},
    "Paladin":   {"hp": 10, "ac": 18, "stats": {"str": 16, "dex": 10, "con": 14, "int": 8,  "wis": 12, "cha": 15}},
    "Ranger":    {"hp": 10, "ac": 15, "stats": {"str": 13, "dex": 16, "con": 13, "int": 10, "wis": 14, "cha": 10}},
    "Bard":      {"hp": 8,  "ac": 13, "stats": {"str": 10, "dex": 14, "con": 12, "int": 12, "wis": 10, "cha": 16}},
    "Druid":     {"hp": 8,  "ac": 14, "stats": {"str": 10, "dex": 12, "con": 13, "int": 12, "wis": 16, "cha": 10}},
    "Monk":      {"hp": 8,  "ac": 15, "stats": {"str": 12, "dex": 16, "con": 13, "int": 10, "wis": 15, "cha": 8}},
    "Warlock":   {"hp": 8,  "ac": 13, "stats": {"str": 10, "dex": 14, "con": 12, "int": 12, "wis": 10, "cha": 16}},
}

_STARTING_GEAR = {
    "Fighter":   ["longsword", "shield", "chain mail armor", "5x javelins", "adventurer's pack"],
    "Barbarian": ["greataxe", "2x handaxes", "explorer's pack", "4x javelins"],
    "Wizard":    ["quarterstaff", "arcane focus", "scholar's pack", "spellbook", "2x daggers"],
    "Sorcerer":  ["light crossbow", "20 bolts", "arcane focus", "dungeoneer's pack", "2x daggers"],
    "Rogue":     ["rapier", "shortbow", "20 arrows", "leather armor", "thieves' tools", "burglar's pack", "2x daggers"],
    "Cleric":    ["mace", "shield", "scale mail", "holy symbol", "priest's pack", "light crossbow", "20 bolts"],
    "Paladin":   ["longsword", "shield", "chain mail armor", "holy symbol", "priest's pack", "5x javelins"],
    "Ranger":    ["2x shortswords", "leather armor", "dungeoneer's pack", "longbow", "20 arrows"],
    "Bard":      ["rapier", "lute", "leather armor", "entertainer's pack", "dagger"],
    "Druid":     ["wooden shield", "scimitar", "leather armor", "druidic focus", "explorer's pack"],
    "Monk":      ["shortsword", "dungeoneer's pack", "10x darts"],
    "Warlock":   ["light crossbow", "20 bolts", "arcane focus", "scholar's pack", "leather armor", "2x daggers"],
}

_STARTING_SPELLS = {
    "Wizard":   {"1st": {"total": 2, "used": 0}},
    "Sorcerer": {"1st": {"total": 2, "used": 0}},
    "Cleric":   {"1st": {"total": 2, "used": 0}},
    "Paladin":  {},
    "Bard":     {"1st": {"total": 2, "used": 0}},
    "Druid":    {"1st": {"total": 2, "used": 0}},
    "Warlock":  {"1st": {"total": 1, "used": 0}},
}

_COMMON_GEAR = ["torch x3", "rations x5", "rope (50 ft)", "waterskin", "50 gp"]


def build_new_character(name, race, char_class, backstory=""):
    """Return a fresh game state dict for a new character."""
    defaults = _CLASS_DEFAULTS.get(char_class, _CLASS_DEFAULTS["Fighter"])
    inventory = _STARTING_GEAR.get(char_class, _STARTING_GEAR["Fighter"]) + _COMMON_GEAR
    spell_slots = _STARTING_SPELLS.get(char_class, {})
    lv = 1
    return {
        "player": {
            "name": name,
            "race": race,
            "class": char_class,
            "level": lv,
            "hp": {"current": defaults["hp"], "max": defaults["hp"]},
            "ac": defaults["ac"],
            "stats": defaults["stats"],
            "xp": 0,
            "xp_to_next": _XP_TABLE[lv + 1],
            "gold": 50,
            "inventory": inventory,
            "spell_slots": spell_slots,
            "conditions": [],
            "backstory": backstory,
        },
        "world": {
            "location": "The Rusty Flagon tavern, village of Millhaven",
            "time": "evening",
            "weather": "stormy",
            "active_quest": None,
            "notes": [],
        },
        "combat": {
            "active": False,
            "enemies": [],
            "round": 0,
        },
        "history": [],
    }

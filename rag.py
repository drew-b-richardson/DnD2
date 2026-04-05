"""Rules RAG — retrieval-augmented generation for D&D 5e mechanics.

Uses ChromaDB + sentence-transformers for semantic search when available.
Falls back to fast keyword overlap scoring when not installed.

Usage:
    rag = RulesRAG()
    context = rag.query("I try to sneak past the guards")
    # → returns relevant rule text to inject into the LLM prompt
"""

# ---------------------------------------------------------------------------
# Embedded SRD rule snippets
# ---------------------------------------------------------------------------

_RULES = [
    ("attack_roll", "attack hit roll d20 weapon melee ranged",
     "Attack Roll: Roll d20 + ability modifier + proficiency bonus (if proficient). "
     "Compare to target AC. Equal or higher = hit. Natural 20 = critical hit (roll damage "
     "dice twice). Natural 1 = automatic miss regardless of modifiers."),

    ("critical_hit", "critical hit crit nat 20 double damage",
     "Critical Hit: On a natural 20, roll all damage dice twice and add modifiers once. "
     "Some features (e.g. Champion Fighter) expand the crit range to 19–20 or 18–20."),

    ("saving_throw", "saving throw save DC resist",
     "Saving Throw: Roll d20 + relevant ability modifier. Add proficiency bonus if "
     "proficient in that save. Meet or beat DC to succeed."),

    ("skill_check", "ability check skill stealth perception athletics acrobatics persuasion",
     "Ability Check: Roll d20 + ability modifier. Add proficiency bonus if proficient in "
     "the skill. Meet or beat DC to succeed. Passive check = 10 + modifiers (no roll)."),

    ("advantage_disadvantage", "advantage disadvantage roll twice",
     "Advantage: Roll 2d20, take higher. Disadvantage: Roll 2d20, take lower. "
     "Advantage and disadvantage cancel each other out regardless of count."),

    ("initiative", "initiative combat order turn start fight",
     "Initiative: Roll d20 + DEX modifier. Highest result acts first. "
     "On a tie, higher DEX modifier goes first; further ties are broken by player choice "
     "or DM decision."),

    ("death_saves", "death saving throw unconscious dying 0 hp fallen",
     "Death Saving Throws (at 0 HP): Each turn roll d20. 10+ = success; 9 or lower = "
     "failure. 3 successes = stabilise. 3 failures = death. Nat 20 = regain 1 HP. "
     "Nat 1 = two failures. Taking damage while down = 1 failure; crit = 2 failures."),

    ("opportunity_attack", "opportunity attack reaction leave reach move away",
     "Opportunity Attack: When a hostile creature you can see moves out of your reach, "
     "use your reaction to make one melee attack against it. Does not trigger if the "
     "creature uses the Disengage action."),

    ("two_weapon_fighting", "two weapon fighting bonus action offhand light",
     "Two-Weapon Fighting: When you attack with a light melee weapon, you may use a "
     "bonus action to attack with a different light weapon in your off hand. No ability "
     "modifier is added to the off-hand damage (unless negative)."),

    ("grapple", "grapple grab hold restrain wrestling",
     "Grapple: Use the Attack action. Make a Strength (Athletics) check contested by "
     "the target's Strength (Athletics) or Dexterity (Acrobatics). Success = target is "
     "grappled (speed 0). To escape: target uses their action for a contested check."),

    ("shove", "shove push knock prone",
     "Shove: Use the Attack action to shove a creature. Make a Strength (Athletics) check "
     "contested by the target's Strength (Athletics) or Dexterity (Acrobatics). Success: "
     "choose to push target 5 ft away or knock it prone."),

    ("dash", "dash action double movement speed",
     "Dash: Take the Dash action to gain extra movement equal to your speed this turn. "
     "Difficult terrain and other movement costs still apply."),

    ("disengage", "disengage action retreat flee escape",
     "Disengage: Your movement does not provoke opportunity attacks for the rest of "
     "this turn."),

    ("dodge", "dodge action defense avoid attacks",
     "Dodge: Until your next turn starts, attacks against you have disadvantage (if you "
     "can see the attacker), and you have advantage on DEX saving throws."),

    ("hide", "hide action stealth hidden invisible unseen",
     "Hide: Make a Dexterity (Stealth) check. On success you become hidden from creatures "
     "that can't perceive you. You lose the hidden condition when you make noise, attack, "
     "or are seen."),

    ("help", "help action assist ally advantage",
     "Help: Give one ally advantage on their next ability check or attack roll "
     "against a creature within 5 ft of you."),

    ("sneak_attack", "sneak attack rogue damage extra finesse ranged",
     "Sneak Attack (Rogue): Once per turn, deal extra damage when you hit with a finesse "
     "or ranged weapon if you have advantage OR an ally is within 5 ft of the target "
     "(and not incapacitated). Extra damage: 1d6 at lv1, +1d6 per two rogue levels "
     "(2d6 at lv3, 3d6 at lv5, etc.)."),

    ("fireball", "fireball spell fire damage 3rd level wizard sorcerer",
     "Fireball (3rd level, Evocation): 150 ft range, 20-ft radius sphere. Each creature "
     "in range makes DC 14 DEX save. Fail = 8d6 fire damage. Success = half. "
     "Ignites flammable objects. +1d6 per slot level above 3rd."),

    ("magic_missile", "magic missile spell force damage wizard sorcerer 1st level",
     "Magic Missile (1st level, Evocation): Create three glowing darts, each dealing "
     "1d4+1 force damage. Each dart hits automatically. You can direct them freely. "
     "+1 dart per slot level above 1st."),

    ("healing_word", "healing word spell heal 1st level cleric bard",
     "Healing Word (1st level, Evocation): Bonus action. 60 ft range. "
     "Target regains 1d4 + spellcasting modifier HP. +1d4 per slot level above 1st."),

    ("cure_wounds", "cure wounds spell heal 1st level cleric paladin ranger druid",
     "Cure Wounds (1st level, Evocation): Touch range. "
     "Target regains 1d8 + spellcasting modifier HP. +1d8 per slot level above 1st."),

    ("shield_spell", "shield spell wizard sorcerer reaction +5 AC",
     "Shield (1st level, Abjuration): Reaction, triggered when you are hit or targeted "
     "by magic missile. Until start of next turn: +5 AC (including against the triggering "
     "attack) and immunity to magic missile."),

    ("thunderwave", "thunderwave spell thunder damage push 1st level wizard cleric",
     "Thunderwave (1st level, Evocation): 15-ft cube originating from you. Each creature "
     "makes DC 13 CON save. Fail = 2d8 thunder damage and pushed 10 ft away. Success = "
     "half damage. Unsecured objects push automatically. Audible up to 300 ft."),

    ("conditions_poisoned", "poisoned condition poison disadvantage",
     "Poisoned: Disadvantage on attack rolls and ability checks."),

    ("conditions_prone", "prone condition knocked down crawling",
     "Prone: Attack rolls against the creature have advantage if within 5 ft, "
     "disadvantage if farther. The creature's own attacks have disadvantage. "
     "Standing up costs half movement speed."),

    ("conditions_paralyzed", "paralyzed condition",
     "Paralyzed: Incapacitated, can't move or speak. Auto-fails STR and DEX saves. "
     "Attacks against have advantage. Any hit within 5 ft is a critical hit."),

    ("conditions_frightened", "frightened condition scared fear",
     "Frightened: Disadvantage on ability checks and attacks while source of fear "
     "is in line of sight. Can't willingly move closer to source."),

    ("conditions_stunned", "stunned condition",
     "Stunned: Incapacitated, can't move, can only speak falteringly. Auto-fails STR "
     "and DEX saves. Attacks against have advantage."),

    ("conditions_blinded", "blinded condition blind darkness",
     "Blinded: Can't see, auto-fails sight-based checks. Attacks have disadvantage; "
     "attacks against have advantage."),

    ("short_rest", "short rest recover hit dice 1 hour",
     "Short Rest: 1 hour of downtime (no strenuous activity). Spend any number of "
     "hit dice: roll each and add CON modifier, regaining that many HP. Warlocks "
     "recover all spell slots on a short rest."),

    ("long_rest", "long rest full recovery 8 hours sleep",
     "Long Rest: At least 8 hours (6 sleep + 2 light activity). Regain all HP. "
     "Recover all expended spell slots and half max hit dice (min 1). "
     "Only one long rest benefit per 24 hours."),

    ("concentration", "concentration spell maintain lose distracted",
     "Concentration: Some spells require concentration. You can only concentrate on "
     "one spell at a time. Taking damage requires a CON save DC 10 or half damage taken "
     "(whichever is higher) or the spell ends. Incapacitation also breaks concentration."),

    ("flanking", "flanking advantage melee two sides",
     "Flanking (optional rule): If two attackers have a creature between them "
     "(opposite sides), both have advantage on melee attacks against it."),

    ("encumbrance", "encumbrance carry weight inventory",
     "Carrying Capacity: STR score × 15 pounds. Push/Drag/Lift limit = STR × 30 lb. "
     "Encumbered (> STR×5 lb): speed −10. Heavily encumbered (> STR×10 lb): speed −20, "
     "disadvantage on STR/DEX/CON checks/saves/attacks."),
]


# ---------------------------------------------------------------------------
# RulesRAG class
# ---------------------------------------------------------------------------

class RulesRAG:
    """Retrieves relevant D&D rules for a given player action."""

    def __init__(self, extra_docs: list = None):
        """extra_docs: list of (id, topic, text) tuples for custom content."""
        self._rules = list(_RULES)
        if extra_docs:
            self._rules.extend(extra_docs)

        self._collection = None
        self._use_chroma = False
        self._init_chroma()

    # ------------------------------------------------------------------
    # ChromaDB initialisation
    # ------------------------------------------------------------------

    def _init_chroma(self):
        try:
            import chromadb  # noqa: F401
            from chromadb.utils import embedding_functions  # noqa: F401
            import chromadb as _chromadb

            client = _chromadb.Client()
            try:
                ef = embedding_functions.DefaultEmbeddingFunction()
                col = client.get_or_create_collection("dnd_rules", embedding_function=ef)
            except Exception:
                col = client.get_or_create_collection("dnd_rules")

            if col.count() == 0:
                col.add(
                    ids=[r[0] for r in self._rules],
                    documents=[r[2] for r in self._rules],
                    metadatas=[{"topic": r[1]} for r in self._rules],
                )
                print(f"  [RAG: indexed {len(self._rules)} rules into ChromaDB]")

            self._collection = col
            self._use_chroma = True

        except ImportError:
            print("  [RAG: chromadb not installed — using keyword search. "
                  "Install with: pip install chromadb]")
        except Exception as e:
            print(f"  [RAG: ChromaDB init failed ({e}) — using keyword search]")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, player_action: str, n: int = 3) -> str:
        """Return relevant rule text to inject into the LLM prompt."""
        if self._use_chroma and self._collection:
            try:
                results = self._collection.query(
                    query_texts=[player_action], n_results=min(n, self._collection.count())
                )
                docs = results.get("documents", [[]])[0]
                return "\n\n".join(docs)
            except Exception:
                pass  # fall through to keyword search
        return self._keyword_search(player_action, n)

    def add_rule(self, rule_id: str, topic: str, text: str):
        """Add a custom rule or lore entry at runtime."""
        self._rules.append((rule_id, topic, text))
        if self._use_chroma and self._collection:
            try:
                self._collection.add(ids=[rule_id], documents=[text],
                                     metadatas=[{"topic": topic}])
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Keyword fallback
    # ------------------------------------------------------------------

    def _keyword_search(self, query: str, n: int) -> str:
        query_words = set(query.lower().split())
        scored = []
        for rule_id, topic, text in self._rules:
            topic_words = set(topic.lower().split())
            text_words = set(text.lower().split())
            score = len(query_words & topic_words) * 3 + len(query_words & text_words)
            if score > 0:
                scored.append((score, text))
        scored.sort(reverse=True)
        return "\n\n".join(text for _, text in scored[:n])

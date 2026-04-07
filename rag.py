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

    # -----------------------------------------------------------------------
    # Core combat mechanics
    # -----------------------------------------------------------------------

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
     "On a tie, higher DEX modifier goes first; further ties broken by player choice."),

    ("death_saves", "death saving throw unconscious dying 0 hp fallen",
     "Death Saving Throws (at 0 HP): Each turn roll d20. 10+ = success; 9 or lower = "
     "failure. 3 successes = stabilise. 3 failures = death. Nat 20 = regain 1 HP. "
     "Nat 1 = two failures. Taking damage while down = 1 failure; crit = 2 failures."),

    ("opportunity_attack", "opportunity attack reaction leave reach move away",
     "Opportunity Attack: When a hostile creature you can see moves out of your reach, "
     "use your reaction to make one melee attack against it. Does not trigger if the "
     "creature uses the Disengage action."),

    ("two_weapon_fighting", "two weapon fighting bonus action offhand light dual wield",
     "Two-Weapon Fighting: When you attack with a light melee weapon, you may use a "
     "bonus action to attack with a different light weapon in your off hand. No ability "
     "modifier added to off-hand damage (unless negative)."),

    ("grapple", "grapple grab hold restrain wrestling",
     "Grapple: Use the Attack action. Make a Strength (Athletics) check contested by "
     "the target's Strength (Athletics) or Dexterity (Acrobatics). Success = target is "
     "grappled (speed 0). To escape: target uses their action for a contested check."),

    ("shove", "shove push knock prone",
     "Shove: Use the Attack action to shove a creature. Make a Strength (Athletics) check "
     "contested by the target's Strength (Athletics) or Dexterity (Acrobatics). Success: "
     "choose to push target 5 ft away or knock it prone."),

    ("cover", "cover half three-quarters full concealment hiding behind",
     "Cover: Half cover (low wall, another creature): +2 AC and DEX saves. "
     "Three-quarters cover (arrow slit, thick trunk): +5 AC and DEX saves. "
     "Full cover (completely hidden): can't be targeted directly."),

    ("difficult_terrain", "difficult terrain slow movement mud rubble water",
     "Difficult Terrain: Moving through difficult terrain costs 2 ft of movement per 1 ft "
     "travelled. Climbing, swimming, and crawling also cost double movement."),

    ("falling", "falling fall damage drop height",
     "Falling: Take 1d6 bludgeoning damage per 10 ft fallen, max 20d6. Land prone "
     "unless avoiding damage with a successful DC 15 Acrobatics check."),

    ("suffocation", "suffocation drowning water hold breath",
     "Suffocation/Drowning: A creature can hold breath for 1 + CON modifier minutes "
     "(min 30 sec). When out of breath, it can survive for CON modifier rounds (min 1), "
     "then drops to 0 HP and begins dying."),

    ("surprise", "surprise ambush unseen unheard sneak attack first round",
     "Surprise: If you are hidden from all enemies when combat starts, the DM may grant "
     "a surprise round. Surprised creatures cannot move or take actions on their first "
     "turn, and cannot take reactions until that turn ends."),

    ("ready_action", "ready action wait trigger reaction prepare",
     "Ready Action: Choose an action and a trigger. When the trigger occurs before your "
     "next turn, use your reaction to take that action. If the trigger doesn't occur, "
     "the action is lost (spell slots still expended if a spell was readied)."),

    ("bonus_action", "bonus action extra fast swift",
     "Bonus Action: Certain class features, spells, or abilities let you take a bonus "
     "action on your turn. You can only take one bonus action per turn. You choose when "
     "to take it, but it must be taken during your turn."),

    ("reaction", "reaction interrupt trigger response once per round",
     "Reaction: An instant response to a trigger. You get one reaction per round, "
     "regained at the start of your turn. Examples: opportunity attack, Shield spell, "
     "Counterspell, Uncanny Dodge, Sentinel feat."),

    ("extra_attack", "extra attack multiple attacks fighter ranger paladin monk",
     "Extra Attack: Some classes (Fighter at lv5, Paladin/Ranger at lv5, Monk at lv5) "
     "can attack twice when taking the Attack action. Fighters gain a third attack at "
     "lv11 and a fourth at lv20."),

    ("spell_attack", "spell attack roll hit ranged touch arcane",
     "Spell Attack: Roll d20 + spellcasting ability modifier + proficiency bonus. "
     "Compare to target AC. Natural 20 = critical hit (roll damage dice twice). "
     "Ranged spell attacks have disadvantage if within 5 ft of a hostile creature."),

    ("flanking", "flanking advantage melee two sides",
     "Flanking (optional rule): If two attackers have a creature between them "
     "(opposite sides), both have advantage on melee attacks against it."),

    ("encumbrance", "encumbrance carry weight inventory",
     "Carrying Capacity: STR score × 15 pounds. Push/Drag/Lift = STR × 30 lb. "
     "Encumbered (>STR×5): speed −10. Heavily encumbered (>STR×10): speed −20, "
     "disadvantage on STR/DEX/CON checks/saves/attacks."),

    ("mounted_combat", "mount horse riding mounted combat",
     "Mounted Combat: Mount acts on your initiative. You can control a trained mount "
     "using a bonus action to direct it. If mount is knocked prone, you must DC 15 "
     "Acrobatics or fall off. If you fall, move 5 ft away and land prone."),

    ("underwater_combat", "underwater combat swimming submerged",
     "Underwater Combat: Melee weapons work normally unless the DM rules otherwise. "
     "Ranged weapons are at disadvantage beyond normal range and automatically miss "
     "beyond long range. Fire damage is halved underwater."),

    # -----------------------------------------------------------------------
    # Vision and light
    # -----------------------------------------------------------------------

    ("vision_light", "darkvision dim light darkness bright light blindsight",
     "Vision: Bright light = normal. Dim light = lightly obscured (disadvantage on "
     "Perception). Darkness = heavily obscured (effectively blinded). Darkvision lets "
     "you see in darkness as dim light (greyscale) up to a set range. Blindsight lets "
     "you perceive without sight within range."),

    ("invisibility_rules", "invisible unseen hidden attacker",
     "Invisible Attacker/Target: Attacks against an invisible creature have disadvantage. "
     "Attacks by an invisible creature have advantage. The creature's location can still "
     "be detected by noise, tracks, or other means."),

    # -----------------------------------------------------------------------
    # Combat actions
    # -----------------------------------------------------------------------

    ("dash", "dash action double movement speed run",
     "Dash: Gain extra movement equal to your speed this turn. "
     "With Cunning Action (Rogue) or similar, can be used as a bonus action."),

    ("disengage", "disengage action retreat flee escape",
     "Disengage: Your movement does not provoke opportunity attacks for the rest of "
     "this turn. Rogues can use this as a bonus action via Cunning Action."),

    ("dodge", "dodge action defense avoid attacks",
     "Dodge: Until your next turn starts, attacks against you have disadvantage (if you "
     "can see the attacker), and you have advantage on DEX saving throws. "
     "Effect ends if incapacitated or speed drops to 0."),

    ("hide", "hide action stealth hidden invisible unseen",
     "Hide: Make a Dexterity (Stealth) check. On success you become hidden from creatures "
     "that can't perceive you. You lose hidden when you make noise, attack, or are seen. "
     "Rogues can Hide as a bonus action via Cunning Action."),

    ("help", "help action assist ally advantage",
     "Help: Give one ally advantage on their next ability check or attack roll "
     "against a creature within 5 ft of you."),

    ("use_object", "use object interact item action bonus",
     "Object Interaction: You can interact with one object for free on your turn "
     "(draw weapon, open door). A second interaction requires the Use an Object action. "
     "Magic items typically require an action to activate."),

    # -----------------------------------------------------------------------
    # Conditions
    # -----------------------------------------------------------------------

    ("conditions_blinded", "blinded condition blind darkness",
     "Blinded: Can't see, auto-fails sight-based checks. Attacks have disadvantage; "
     "attacks against the creature have advantage."),

    ("conditions_charmed", "charmed condition charm friend friendly",
     "Charmed: Can't attack the charmer or target them with harmful abilities/spells. "
     "The charmer has advantage on social ability checks against the charmed creature."),

    ("conditions_deafened", "deafened condition deaf hearing sound",
     "Deafened: Can't hear, automatically fails hearing-based Perception checks."),

    ("conditions_exhaustion", "exhaustion tired fatigued levels 1 2 3 4 5 6",
     "Exhaustion (6 levels): 1=disadvantage on ability checks; 2=speed halved; "
     "3=disadvantage on attacks and saves; 4=HP max halved; 5=speed 0; 6=death. "
     "Each long rest removes one level. Require food/water to avoid gaining levels."),

    ("conditions_frightened", "frightened condition scared fear",
     "Frightened: Disadvantage on ability checks and attacks while source of fear "
     "is in line of sight. Can't willingly move closer to source."),

    ("conditions_grappled", "grappled condition grabbed held speed zero",
     "Grappled: Speed becomes 0. The condition ends if the grappler is incapacitated, "
     "or if the grappled creature is moved beyond the grappler's reach."),

    ("conditions_incapacitated", "incapacitated condition no action bonus",
     "Incapacitated: Can't take actions or reactions."),

    ("conditions_invisible", "invisible condition unseen hidden",
     "Invisible: Impossible to see without special sense. Heavily obscured for hiding. "
     "Attacks against have disadvantage; creature's own attacks have advantage."),

    ("conditions_paralyzed", "paralyzed condition",
     "Paralyzed: Incapacitated, can't move or speak. Auto-fails STR and DEX saves. "
     "Attacks against have advantage. Any hit within 5 ft is a critical hit."),

    ("conditions_petrified", "petrified condition stone transformed",
     "Petrified: Transformed to stone. Incapacitated, speed 0. Resistance to all damage. "
     "Immune to poison and disease. Auto-fails STR and DEX saves. Attacks against "
     "have advantage."),

    ("conditions_poisoned", "poisoned condition poison disadvantage",
     "Poisoned: Disadvantage on attack rolls and ability checks."),

    ("conditions_prone", "prone condition knocked down crawling",
     "Prone: Attacks against have advantage within 5 ft, disadvantage beyond. "
     "Creature's attacks have disadvantage. Standing up costs half movement speed."),

    ("conditions_restrained", "restrained condition entangled web held",
     "Restrained: Speed 0. Attacks against have advantage. Creature's own attacks have "
     "disadvantage. DEX saving throws have disadvantage."),

    ("conditions_stunned", "stunned condition",
     "Stunned: Incapacitated, can't move, can only speak falteringly. Auto-fails STR "
     "and DEX saves. Attacks against have advantage."),

    ("conditions_unconscious", "unconscious condition knocked out asleep",
     "Unconscious: Incapacitated, can't move or speak, unaware of surroundings. "
     "Drops everything held, falls prone. Attacks against have advantage. "
     "Any hit within 5 ft is a critical hit. Auto-fails STR and DEX saves."),

    # -----------------------------------------------------------------------
    # Rests and recovery
    # -----------------------------------------------------------------------

    ("short_rest", "short rest recover hit dice 1 hour",
     "Short Rest: 1 hour of downtime. Spend any number of hit dice: roll each + CON "
     "modifier to regain HP. Warlocks recover all spell slots. Some class features "
     "(Fighter Second Wind, Monk Ki) also recharge."),

    ("long_rest", "long rest full recovery 8 hours sleep",
     "Long Rest: At least 8 hours (6 sleep + 2 light activity). Regain all HP and "
     "recover all expended spell slots. Recover half max hit dice (min 1). "
     "Only once per 24 hours. Interrupted by 1+ hour of strenuous activity."),

    ("hit_dice", "hit dice recovery healing rest spend",
     "Hit Dice: Each class has a hit die (d6–d12). You have one per level. Spend them "
     "during a short rest to heal. Regain half your total (rounded up) on a long rest."),

    # -----------------------------------------------------------------------
    # Spellcasting mechanics
    # -----------------------------------------------------------------------

    ("concentration", "concentration spell maintain lose distracted",
     "Concentration: Only one concentration spell active at a time. Taking damage "
     "requires CON save DC 10 or half damage taken (higher). Fail = spell ends. "
     "Incapacitation, death, or casting another concentration spell also ends it."),

    ("spellcasting_general", "spellcasting cast spell slot level component verbal somatic material",
     "Spellcasting: Casting a spell of 1st level or higher expends a spell slot of that "
     "level or higher. Cantrips are free. Most spells require Verbal, Somatic, and/or "
     "Material components. Silenced/bound casters may be unable to cast."),

    ("cantrips", "cantrip 0 level at will free cast unlimited",
     "Cantrips: 0-level spells cast at will with no spell slot required. "
     "Many cantrips scale with character level (e.g. Eldritch Blast, Fire Bolt, "
     "Toll the Dead deal more damage at lv5, 11, and 17)."),

    ("counterspell", "counterspell interrupt cancel magic 3rd level",
     "Counterspell (3rd level, Abjuration): Reaction when a creature within 60 ft casts "
     "a spell. Spells of 3rd level or lower are automatically countered. For higher-level "
     "spells: make an ability check DC 10 + spell's level. Success = countered."),

    ("dispel_magic", "dispel magic end spell effect 3rd level",
     "Dispel Magic (3rd level, Abjuration): End one spell of 3rd level or lower "
     "automatically on a target. Higher-level spells require ability check DC 10 + "
     "spell's level. Can also end magic effects on objects or areas."),

    ("identify_spell", "identify spell attune learn properties magic item",
     "Identify (1st level, Divination): Ritual or 1 minute cast. Learn a magic item's "
     "properties and how to use them, whether it requires attunement, and any active "
     "spells affecting a creature or object."),

    ("bless", "bless spell bonus d4 attack save 1st level cleric paladin",
     "Bless (1st level, Enchantment, Concentration): Up to 3 creatures within 30 ft. "
     "Each adds 1d4 to attack rolls and saving throws. Duration 1 minute. "
     "+1 creature per slot level above 1st."),

    ("bane", "bane spell penalty d4 attack save 1st level cleric",
     "Bane (1st level, Enchantment, Concentration): Up to 3 creatures within 30 ft; "
     "CHA save or subtract 1d4 from attack rolls and saving throws. 1 min duration. "
     "+1 creature per slot level above 1st."),

    ("charm_person", "charm person spell social friendly humanoid 1st level",
     "Charm Person (1st level, Enchantment): One humanoid within 30 ft. WIS save "
     "(advantage if you or allies are fighting it). Fail = charmed for 1 hour or until "
     "harmed. Regards you as friendly. Knows it was charmed afterward."),

    ("sleep_spell", "sleep spell unconscious put to 1st level",
     "Sleep (1st level, Enchantment): Roll 5d8; total = HP of creatures affected, "
     "starting with lowest HP first, within 20-ft radius. Unconscious for 1 minute or "
     "until woken. Undead and immune-to-charm creatures are unaffected. +2d8 per slot "
     "above 1st."),

    ("hold_person", "hold person spell paralyzed restrain humanoid 2nd level",
     "Hold Person (2nd level, Enchantment, Concentration): One humanoid; WIS save or "
     "paralyzed for 1 minute. Repeat save at end of each turn. +1 target per slot "
     "above 2nd."),

    ("invisibility_spell", "invisibility spell invisible 2nd level",
     "Invisibility (2nd level, Illusion, Concentration): One creature becomes invisible "
     "for 1 hour. Ends if the creature attacks or casts a spell. "
     "+1 target per slot above 2nd. Greater Invisibility (4th) doesn't end on attack."),

    ("misty_step", "misty step teleport bonus action 30 ft 2nd level",
     "Misty Step (2nd level, Conjuration): Bonus action. Teleport up to 30 ft to an "
     "unoccupied space you can see. No components required beyond Verbal."),

    ("mirror_image", "mirror image duplicates illusion 2nd level avoid attack",
     "Mirror Image (2nd level, Illusion): Creates 3 duplicates. When attacked, roll d20: "
     "3 images=6+, 2 images=8+, 1 image=11+ means the duplicate is hit instead. "
     "Duplicates destroyed by any hit. Concentration not required."),

    ("haste", "haste speed double action extra fast 3rd level",
     "Haste (3rd level, Transmutation, Concentration): One willing creature. Speed "
     "doubled, +2 AC, advantage on DEX saves, gains extra action each turn (attack, "
     "dash, disengage, hide, or use object only). When it ends: can't move or act for "
     "1 full turn (lethargy)."),

    ("slow", "slow spell reduce speed action 3rd level",
     "Slow (3rd level, Transmutation, Concentration): Up to 6 creatures in 40-ft cube; "
     "WIS save or: speed halved, −2 AC and DEX saves, can't take reactions, only one "
     "action or bonus action (not both) per turn, can't make more than one attack per "
     "action. Repeat save each turn."),

    ("spiritual_weapon", "spiritual weapon spell bonus action floating 2nd level cleric",
     "Spiritual Weapon (2nd level, Evocation): Bonus action to summon a floating weapon "
     "within 60 ft. Each turn, bonus action to move it 20 ft and attack: spell attack "
     "roll, 1d8 + spellcasting modifier force damage. Lasts 1 minute. Not concentration. "
     "+1d8 per two slot levels above 2nd."),

    ("sacred_flame", "sacred flame cantrip cleric dex save fire radiant",
     "Sacred Flame (Evocation Cantrip): Flame-like radiance descends on a creature in "
     "60 ft; DEX save (no cover bonus) or take 1d8 radiant damage. "
     "Damage increases to 2d8 at lv5, 3d8 at lv11, 4d8 at lv17."),

    ("guiding_bolt", "guiding bolt cleric 1st level radiant attack advantage",
     "Guiding Bolt (1st level, Evocation): Ranged spell attack, 120 ft, 4d6 radiant "
     "damage. The next attack roll against the target before your next turn has "
     "advantage. +1d6 per slot level above 1st."),

    ("toll_the_dead", "toll the dead cantrip cleric wizard necromancy damaged",
     "Toll the Dead (Necromancy Cantrip): One creature in 60 ft; WIS save or take 1d8 "
     "necrotic damage (1d12 if missing any HP). Scales to 2d8/2d12 at lv5, "
     "3d8/3d12 at lv11, 4d8/4d12 at lv17."),

    ("eldritch_blast", "eldritch blast warlock cantrip force beam",
     "Eldritch Blast (Evocation Cantrip): 1–4 beams (1 at lv1, 2 at lv5, 3 at lv11, "
     "4 at lv17), each targeting same or different creatures within 120 ft. "
     "Each beam: ranged spell attack for 1d10 force damage. Invocations can modify it."),

    ("hex", "hex warlock spell curse bonus damage disadvantage 1st level",
     "Hex (1st level, Enchantment, Concentration): Bonus action. Curse a creature in "
     "90 ft. Deal +1d6 necrotic damage to it on each hit. Choose one ability — target "
     "has disadvantage on checks with it. If target drops to 0 HP, move curse as "
     "bonus action. Duration: 1/8/24 hours at 1st/3rd/5th level slots."),

    ("hunters_mark", "hunter's mark ranger spell track target bonus damage 1st level",
     "Hunter's Mark (1st level, Divination, Concentration): Bonus action. Mark a "
     "creature within 90 ft. Deal +1d6 damage when you hit it. Advantage on Perception "
     "and Survival to find it. Move mark as bonus action if target dies. "
     "Duration: 1/8/24 hrs at 1st/3rd/5th level slots."),

    ("polymorph", "polymorph transform beast shape 4th level",
     "Polymorph (4th level, Transmutation, Concentration): Transform a creature into a "
     "beast of CR ≤ target's level/CR. Uses beast's stat block but retains personality. "
     "Target reverts when it reaches 0 HP or spell ends; excess damage carries over. "
     "Unwilling targets make WIS save."),

    ("fly_spell", "fly spell flight speed 60 ft 3rd level",
     "Fly (3rd level, Transmutation, Concentration): One willing creature gains 60 ft "
     "fly speed for 10 minutes. If concentration ends while airborne, the creature falls "
     "unless it has its own fly speed. +1 target per slot above 3rd."),

    ("fireball", "fireball spell fire damage 3rd level wizard sorcerer",
     "Fireball (3rd level, Evocation): 150 ft range, 20-ft radius sphere. Each creature "
     "makes DC 14 DEX save. Fail = 8d6 fire damage; success = half. "
     "Ignites flammable objects. +1d6 per slot level above 3rd."),

    ("magic_missile", "magic missile spell force damage wizard sorcerer 1st level",
     "Magic Missile (1st level, Evocation): Three darts, each 1d4+1 force damage, "
     "hitting automatically. Directed freely. +1 dart per slot above 1st."),

    ("healing_word", "healing word spell heal 1st level cleric bard bonus action",
     "Healing Word (1st level, Evocation): Bonus action. 60 ft range. "
     "1d4 + spellcasting modifier HP. +1d4 per slot above 1st."),

    ("cure_wounds", "cure wounds spell heal 1st level cleric paladin ranger druid touch",
     "Cure Wounds (1st level, Evocation): Touch. 1d8 + spellcasting modifier HP. "
     "+1d8 per slot above 1st."),

    ("shield_spell", "shield spell wizard sorcerer reaction +5 AC magic missile",
     "Shield (1st level, Abjuration): Reaction when hit or targeted by Magic Missile. "
     "+5 AC until start of next turn (including the triggering attack). "
     "Immune to Magic Missile."),

    ("thunderwave", "thunderwave spell thunder damage push 1st level",
     "Thunderwave (1st level, Evocation): 15-ft cube from you. CON save or 2d8 thunder "
     "damage and pushed 10 ft; half on success. Audible 300 ft. +1d8 per slot above 1st."),

    ("shatter", "shatter spell thunder damage 2nd level area",
     "Shatter (2nd level, Evocation): 10-ft radius sphere, 60 ft range. CON save or "
     "3d8 thunder damage; half on success. Inorganic material auto-fails. "
     "+1d8 per slot above 2nd. Painful sound; creatures made of stone/metal have "
     "disadvantage on the save."),

    ("web", "web spell restrained difficult terrain 2nd level",
     "Web (2nd level, Conjuration, Concentration): 20-ft cube fills with webs for "
     "1 hour. Difficult terrain. Creatures in area make DEX save or are restrained. "
     "Restrained creatures repeat save each turn. Web is flammable (5 ft cube burns "
     "away each round if ignited)."),

    ("darkness_spell", "darkness spell magical dark 60 ft 2nd level",
     "Darkness (2nd level, Evocation, Concentration): 15-ft radius magical darkness "
     "for 10 minutes within 60 ft. Blocks darkvision. Mundane light and lower-level "
     "spells can't illuminate it. Can be anchored to an object."),

    ("suggestion", "suggestion spell command direct 2nd level enchantment",
     "Suggestion (2nd level, Enchantment, Concentration): One creature that can hear "
     "and understand you; WIS save or follows a reasonable one- or two-sentence "
     "suggestion for up to 8 hours. Activity causing harm ends the spell."),

    ("hypnotic_pattern", "hypnotic pattern spell incapacitated charmed 3rd level",
     "Hypnotic Pattern (3rd level, Illusion, Concentration): 30-ft cube, 120 ft range. "
     "Each creature makes WIS save or is charmed and incapacitated for 1 minute. "
     "Ends for a creature if it takes damage or another creature shakes it awake."),

    # -----------------------------------------------------------------------
    # Class features
    # -----------------------------------------------------------------------

    ("rage", "rage barbarian bonus action angry primal fury",
     "Rage (Barbarian): Bonus action to enter rage for 1 minute. Advantage on STR "
     "checks and saves, bonus damage on STR melee attacks (+2/+3/+4 by tier), "
     "resistance to bludgeoning/piercing/slashing damage. "
     "Ends early if you don't attack or take damage since your last turn."),

    ("reckless_attack", "reckless attack barbarian advantage disadvantage",
     "Reckless Attack (Barbarian): Before making your first attack, choose to attack "
     "recklessly. Gain advantage on STR melee attack rolls this turn, but all attacks "
     "against you have advantage until your next turn."),

    ("bardic_inspiration", "bardic inspiration bard bonus d6 d8 d10 d12 ally",
     "Bardic Inspiration (Bard): Bonus action; give one creature within 60 ft an "
     "Inspiration die (d6 at lv1, d8 at lv5, d10 at lv10, d12 at lv15). "
     "The creature can add it to one attack roll, ability check, or saving throw within "
     "10 minutes. Uses = CHA modifier (min 1); recover on long rest (short rest at lv5)."),

    ("channel_divinity", "channel divinity cleric paladin turn undead sacred",
     "Channel Divinity (Cleric/Paladin): Use a powerful divine effect (varies by "
     "subclass and level). Recharges on short or long rest. Common use: Turn Undead — "
     "undead within 30 ft make WIS save or are turned (flee, can't approach) for 1 min."),

    ("divine_smite", "divine smite paladin bonus radiant damage spell slot",
     "Divine Smite (Paladin): When you hit with a melee weapon attack, expend a spell "
     "slot to deal extra radiant damage: 2d8 for a 1st-level slot, +1d8 per level above "
     "1st (max 5d8). +1d8 if target is undead or a fiend. No action required."),

    ("lay_on_hands", "lay on hands paladin heal pool hit points cure disease poison",
     "Lay on Hands (Paladin): Pool of HP = paladin level × 5. Use action to restore "
     "any number of HP from the pool, or spend 5 HP to cure one disease or poison. "
     "Recharges on long rest."),

    ("second_wind", "second wind fighter bonus action heal self",
     "Second Wind (Fighter): Bonus action to regain 1d10 + fighter level HP. "
     "Recharges on short or long rest."),

    ("action_surge", "action surge fighter extra action once",
     "Action Surge (Fighter): Once per short or long rest, take one additional action "
     "on your turn (not a bonus action). At lv17, usable twice between rests."),

    ("cunning_action", "cunning action rogue bonus dash disengage hide",
     "Cunning Action (Rogue lv2): Use a bonus action to Dash, Disengage, or Hide."),

    ("uncanny_dodge", "uncanny dodge rogue halve damage reaction",
     "Uncanny Dodge (Rogue lv5): When an attacker you can see hits you, use your "
     "reaction to halve the attack's damage."),

    ("sneak_attack", "sneak attack rogue damage extra finesse ranged",
     "Sneak Attack (Rogue): Once per turn, deal extra damage when you hit with a finesse "
     "or ranged weapon if you have advantage OR an ally is within 5 ft of the target. "
     "1d6 at lv1, +1d6 per two rogue levels (2d6 at lv3, 3d6 at lv5, etc.)."),

    ("evasion", "evasion rogue monk dex save no damage half",
     "Evasion (Rogue lv7 / Monk lv7): When you make a DEX save to take half damage, "
     "you take no damage on a success and only half on a failure."),

    ("wild_shape", "wild shape druid transform beast animal",
     "Wild Shape (Druid): Use action (or bonus action at lv2 with Moon Druid) to "
     "transform into a beast of CR up to 1/4 (lv2), 1/2 (lv4), or 1 (lv8 Moon Druid). "
     "Use beast's physical stats; retain mental stats and class features. "
     "Revert when HP reaches 0 (excess damage transfers)."),

    ("martial_arts", "martial arts monk unarmed strike bonus action",
     "Martial Arts (Monk): Unarmed strikes use the higher of STR or DEX modifier. "
     "Martial Arts die: d4 (lv1), d6 (lv5), d8 (lv11), d10 (lv17). "
     "After attacking with unarmed strike or monk weapon, make one unarmed bonus action "
     "attack."),

    ("ki_points", "ki points monk stunning strike step of the wind patient defense",
     "Ki Points (Monk): Points = monk level; recover on short or long rest. "
     "Flurry of Blows (1 ki): 2 extra unarmed bonus attacks. "
     "Patient Defense (1 ki): Dodge as bonus action. "
     "Step of Wind (1 ki): Dash or Disengage as bonus action, jump distance doubled. "
     "Stunning Strike (1 ki): After hitting, target makes CON save or stunned until "
     "your next turn."),

    ("spellcasting_focus", "spellcasting focus arcane holy symbol component",
     "Spellcasting Focus: Wizards use an arcane focus or spellbook. Clerics/Paladins "
     "use a holy symbol. Druids use a druidic focus. Bards use a musical instrument. "
     "A focus replaces material components (unless the component has a cost or is "
     "consumed by the spell)."),

    # -----------------------------------------------------------------------
    # Racial traits
    # -----------------------------------------------------------------------

    ("darkvision_trait", "darkvision race elf dwarf half-orc tiefling gnome see dark",
     "Darkvision (Racial Trait): See in darkness as if it were dim light (greyscale) "
     "up to 60 ft (120 ft for some races). Dim light counts as bright light. "
     "You cannot discern colour in darkness, only shades of grey."),

    ("halfling_lucky", "halfling lucky reroll nat 1 attack check save",
     "Lucky (Halfling): When you roll a 1 on an attack roll, ability check, or saving "
     "throw, you can reroll the die and must use the new roll."),

    ("relentless_endurance", "relentless endurance half-orc 1 hp drop survive",
     "Relentless Endurance (Half-Orc): When reduced to 0 HP but not killed outright, "
     "drop to 1 HP instead. Usable once per long rest."),

    ("dragonborn_breath", "dragonborn breath weapon dragon fire lightning breath",
     "Breath Weapon (Dragonborn): Action to exhale destructive energy in a line or cone "
     "(depending on ancestry). Each creature in the area makes DEX or CON save "
     "(DC 8 + CON modifier + proficiency) or takes 2d6 damage (more at higher levels). "
     "Usable once per short or long rest."),

    ("gnome_cunning", "gnome cunning advantage INT WIS CHA magic save",
     "Gnome Cunning: Advantage on all INT, WIS, and CHA saving throws against magic."),

    # -----------------------------------------------------------------------
    # General adventuring
    # -----------------------------------------------------------------------

    ("proficiency_bonus", "proficiency bonus level scaling attack save skill",
     "Proficiency Bonus: +2 at lv1–4, +3 at lv5–8, +4 at lv9–12, +5 at lv13–16, "
     "+6 at lv17–20. Added to attack rolls, saving throws, and ability checks "
     "you are proficient in."),

    ("inspiration", "inspiration reward roleplaying advantage",
     "Inspiration: Granted by the DM for exceptional roleplaying. Spend it to gain "
     "advantage on any attack roll, saving throw, or ability check. You can give your "
     "inspiration to another player. You can only hold one inspiration at a time."),

    ("passive_perception", "passive perception notice detect ambush surprise",
     "Passive Perception: 10 + Wisdom (Perception) modifier. Used when not actively "
     "searching — e.g. to notice a hidden creature or spot a trap while distracted. "
     "The DM compares against a creature's Stealth check to determine if it's detected."),

    ("attunement", "attunement magic item bond require",
     "Attunement: Some magic items require attunement. Spend a short rest bonding with "
     "the item. Max 3 attuned items at once. Attuning to a 4th requires ending "
     "attunement to another first."),

    ("potion_of_healing", "potion healing drink bonus action hp restore",
     "Potion of Healing: Use an action (or bonus action if a feature allows) to drink. "
     "Regain 2d4+2 HP. Greater Healing: 4d4+4. Superior: 8d4+8. Supreme: 10d4+20."),

    ("short_rest", "short rest recover hit dice 1 hour",
     "Short Rest: 1 hour of downtime. Spend hit dice to heal. Warlocks regain spell "
     "slots. Some features recharge (Fighter Second Wind, Action Surge)."),

    ("long_rest", "long rest full recovery 8 hours sleep",
     "Long Rest: 8 hours. Regain all HP, all spell slots, half max hit dice. "
     "Once per 24 hours. Interrupted by 1+ hour of strenuous activity."),
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

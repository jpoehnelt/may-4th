"""
gen_diverse.py — Append diverse English/Yoda OSV pairs to yoda_osv.jsonl.

Targets the lexical-diversity gap identified at exp-034:
- Broad noun/verb vocabulary across animals, food, tools, nature, professions,
  body parts, household items, vehicles, abstractions.
- Multiple sentence patterns with object-fronting (the OSV transformation).
- Avoids over-representing the existing SWE/Star Wars domains.
"""

import json
import random
from pathlib import Path

random.seed(7)

# ----- Vocabulary pools (broad, multi-domain) -----

PROPER_NAMES = [
    "Anna", "Marcus", "Liang", "Priya", "Diego", "Naomi", "Tomas", "Aisha",
    "Felix", "Saoirse", "Kenji", "Yara", "Boris", "Lila", "Otto", "Mira",
    "Hassan", "Greta", "Ravi", "Iris", "Caleb", "Amara", "Nora", "Soren",
    "Fatima", "Pavel", "Elena", "Theo", "Zara", "Idris", "Esme", "Ronan",
]

COMMON_SUBJECTS_SING = [
    "the chef", "the doctor", "the engineer", "the gardener", "the mechanic",
    "the painter", "the pilot", "the teacher", "the farmer", "the baker",
    "the librarian", "the carpenter", "the sculptor", "the violinist",
    "the architect", "the historian", "the diver", "the climber",
    "the cyclist", "the writer", "the photographer", "the nurse",
    "the firefighter", "the astronomer", "the geologist", "the poet",
    "the cat", "the dog", "the owl", "the fox", "the otter", "the heron",
    "the wolf", "the rabbit", "the bear", "the eagle", "the elephant",
    "the dolphin", "the squirrel", "the tortoise", "the hawk", "the badger",
]

COMMON_SUBJECTS_PLUR = [
    "the chefs", "the doctors", "the engineers", "the gardeners", "the children",
    "the dancers", "the farmers", "the hikers", "the rangers", "the sailors",
    "the bakers", "the climbers", "the painters", "the soldiers", "the cyclists",
    "the wolves", "the geese", "the whales", "the bees", "the otters",
]

PRONOUN_SUBJECTS = ["I", "you", "we", "they", "she", "he"]

# Verbs in 3rd-person singular present (for singular common subjects + proper names)
VERBS_3SG = [
    "fixes", "builds", "writes", "paints", "carries", "cooks", "studies",
    "plants", "delivers", "drafts", "polishes", "carves", "shapes", "tunes",
    "explains", "questions", "celebrates", "remembers", "honors", "designs",
    "translates", "examines", "measures", "weighs", "ignites", "lifts",
    "wraps", "harvests", "navigates", "decorates", "rescues", "hides",
    "follows", "challenges", "watches", "imagines", "recalls", "sketches",
    "publishes", "rehearses", "polishes", "abandons", "restores", "claims",
    "collects", "gathers", "selects", "defends", "protects", "delivers",
]

# Bare-infinitive verbs for plural / pronoun subjects
VERBS_BARE = [
    "fix", "build", "write", "paint", "carry", "cook", "study",
    "plant", "deliver", "draft", "polish", "carve", "shape", "tune",
    "explain", "question", "celebrate", "remember", "honor", "design",
    "translate", "examine", "measure", "weigh", "ignite", "lift",
    "wrap", "harvest", "navigate", "decorate", "rescue", "hide",
    "follow", "challenge", "watch", "imagine", "recall", "sketch",
    "publish", "rehearse", "abandon", "restore", "claim",
    "collect", "gather", "select", "defend", "protect",
]

# Past-tense verbs (work for any subject)
VERBS_PAST = [
    "fixed", "built", "wrote", "painted", "carried", "cooked", "studied",
    "planted", "delivered", "drafted", "polished", "carved", "shaped", "tuned",
    "explained", "questioned", "celebrated", "remembered", "honored", "designed",
    "translated", "examined", "measured", "weighed", "ignited", "lifted",
    "wrapped", "harvested", "navigated", "decorated", "rescued", "hid",
    "followed", "challenged", "watched", "imagined", "recalled", "sketched",
    "published", "rehearsed", "abandoned", "restored", "claimed",
    "collected", "gathered", "selected", "defended", "protected", "destroyed",
    "discovered", "uncovered", "earned", "ignored", "explored", "embraced",
]

# Modals + bare verb pairs: ("verb", "modal-prefix")
MODAL_PAIRS = [
    ("must", "fix"), ("must", "build"), ("must", "find"),
    ("should", "review"), ("should", "polish"), ("should", "examine"),
    ("can", "carry"), ("can", "lift"), ("can", "shape"),
    ("will", "rescue"), ("will", "deliver"), ("will", "design"),
    ("might", "remember"), ("might", "follow"), ("might", "uncover"),
    ("could", "restore"), ("could", "defend"), ("could", "discover"),
]

# Object NPs across many semantic domains
OBJECTS = [
    # Tools / artifacts
    "the engine", "the bridge", "the lantern", "the kettle", "the saddle",
    "the harp", "the violin", "the canvas", "the manuscript", "the telescope",
    "the compass", "the lighthouse", "the windmill", "the mirror", "the fountain",
    "the chest", "the carriage", "the wagon", "the loom", "the crane",
    # Food / cooking
    "the bread", "the pasta", "the cake", "the soup", "the salad", "the stew",
    "the pie", "the omelet", "the risotto", "the dumpling", "the broth",
    # Nature / geography
    "the river", "the mountain", "the forest", "the meadow", "the valley",
    "the cliff", "the harbor", "the orchard", "the canyon", "the lagoon",
    "the glacier", "the geyser", "the dune", "the marsh", "the reef",
    # Household / objects
    "the lamp", "the chair", "the painting", "the carpet", "the curtain",
    "the pillow", "the drawer", "the bookshelf", "the cupboard", "the chimney",
    # Animals (as objects)
    "the kitten", "the puppy", "the colt", "the calf", "the foal",
    "the chick", "the cub", "the lamb", "the fawn", "the duckling",
    # Body / health
    "the wound", "the scar", "the bruise", "the splint", "the bandage",
    # Abstract / cognitive
    "the dream", "the memory", "the rumor", "the legend", "the puzzle",
    "the riddle", "the mystery", "the prophecy", "the secret", "the promise",
    "the lesson", "the warning", "the apology", "the confession", "the proverb",
    "the calendar", "the schedule", "the agreement", "the contract", "the sketch",
    # Vehicles / travel
    "the boat", "the kayak", "the canoe", "the bicycle", "the carriage",
    "the train", "the airship", "the sailboat", "the rocket", "the gondola",
    # Clothing
    "the cloak", "the scarf", "the boot", "the glove", "the hat",
    "the coat", "the apron", "the sash", "the helmet", "the tunic",
]

# Adverbs of manner
ADVERBS = [
    "carefully", "quickly", "patiently", "boldly", "silently", "joyfully",
    "stubbornly", "calmly", "fiercely", "tenderly", "skillfully", "loudly",
    "gracefully", "diligently", "anxiously", "earnestly", "swiftly",
]

# Prepositional phrases (time / place)
PP_TIME = [
    "before sunrise", "after the harvest", "during the storm",
    "by candlelight", "at dawn", "at midnight", "on Sunday",
    "before the festival", "after the journey", "during dinner",
    "on the longest day", "before winter", "by the second moon",
    "at the river bend", "after the rain", "before the meeting",
]
PP_PLACE = [
    "in the meadow", "by the river", "near the orchard", "on the cliff",
    "behind the mill", "above the rooftops", "beneath the bridge",
    "across the valley", "at the marketplace", "inside the cabin",
    "outside the temple", "along the coast", "between the ridges",
]


def cap(s):
    return s[:1].upper() + s[1:] if s else s


def lower_first(s):
    # Preserve case for proper names and pronoun "I"
    if not s:
        return s
    first_word = s.split(" ", 1)[0]
    if first_word in PROPER_NAMES or first_word == "I":
        return s
    return s[:1].lower() + s[1:]


def make_subject_verb_pair():
    """Return (subject_phrase, verb_3sg_or_bare_or_past) with matching agreement."""
    pattern = random.random()
    if pattern < 0.30:  # singular common subject + 3sg verb
        return random.choice(COMMON_SUBJECTS_SING), random.choice(VERBS_3SG), "3sg"
    if pattern < 0.50:  # proper name + 3sg verb (present) OR past
        sub = random.choice(PROPER_NAMES)
        if random.random() < 0.5:
            return sub, random.choice(VERBS_3SG), "3sg"
        return sub, random.choice(VERBS_PAST), "past"
    if pattern < 0.70:  # plural subject + bare verb
        return random.choice(COMMON_SUBJECTS_PLUR), random.choice(VERBS_BARE), "bare"
    if pattern < 0.85:  # pronoun subject + bare verb (works for I, you, we, they)
        sub = random.choice(["I", "you", "we", "they"])
        return sub, random.choice(VERBS_BARE), "bare"
    # past tense for any subject
    sub = random.choice(COMMON_SUBJECTS_SING + COMMON_SUBJECTS_PLUR + PROPER_NAMES)
    return sub, random.choice(VERBS_PAST), "past"


def template_basic_svo():
    """S V O. -> O, S V."""
    s, v, _ = make_subject_verb_pair()
    o = random.choice(OBJECTS)
    en = f"{cap(s)} {v} {o}."
    yoda = f"{cap(o)}, {lower_first(s)} {v}."
    return en, yoda


def template_svo_adverb():
    """S V O ADV. -> O ADV, S V."""
    s, v, _ = make_subject_verb_pair()
    o = random.choice(OBJECTS)
    adv = random.choice(ADVERBS)
    en = f"{cap(s)} {v} {o} {adv}."
    yoda = f"{cap(o)} {adv}, {lower_first(s)} {v}."
    return en, yoda


def template_svo_pp():
    """S V O PP. -> O PP, S V."""
    s, v, _ = make_subject_verb_pair()
    o = random.choice(OBJECTS)
    pp = random.choice(PP_TIME + PP_PLACE)
    en = f"{cap(s)} {v} {o} {pp}."
    yoda = f"{cap(o)} {pp}, {lower_first(s)} {v}."
    return en, yoda


def template_pp_fronted():
    """S V O PP. -> PP, S V O.  (PP fronting instead of object)"""
    s, v, _ = make_subject_verb_pair()
    o = random.choice(OBJECTS)
    pp = random.choice(PP_TIME + PP_PLACE)
    en = f"{cap(s)} {v} {o} {pp}."
    yoda = f"{cap(pp)}, {lower_first(s)} {v} {o}."
    return en, yoda


def template_modal():
    """S MODAL V O. -> O, V S MODAL.  (verb-subject inversion with modal)"""
    modal, v = random.choice(MODAL_PAIRS)
    s = random.choice(["I", "you", "we", "they"] + COMMON_SUBJECTS_SING + COMMON_SUBJECTS_PLUR + PROPER_NAMES)
    o = random.choice(OBJECTS)
    en = f"{cap(s)} {modal} {v} {o}."
    yoda = f"{cap(o)}, {v} {lower_first(s)} {modal}."
    return en, yoda


def template_modal_pp():
    """S MODAL V O PP. -> O PP, V S MODAL."""
    modal, v = random.choice(MODAL_PAIRS)
    s = random.choice(["I", "you", "we", "they"] + COMMON_SUBJECTS_SING + COMMON_SUBJECTS_PLUR + PROPER_NAMES)
    o = random.choice(OBJECTS)
    pp = random.choice(PP_TIME + PP_PLACE)
    en = f"{cap(s)} {modal} {v} {o} {pp}."
    yoda = f"{cap(o)} {pp}, {v} {lower_first(s)} {modal}."
    return en, yoda


TEMPLATES = [
    template_basic_svo,
    template_basic_svo,           # weight basic SVO higher
    template_svo_adverb,
    template_svo_pp,
    template_pp_fronted,
    template_modal,
    template_modal_pp,
]


def main(out_path="data/yoda_osv.jsonl", n=18000):
    """Append n diverse pairs to out_path. Dedup against existing keys."""
    out_path = Path(out_path)
    seen_en = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                pair = json.loads(line)
                seen_en.add(pair["en"].lower())
            except (json.JSONDecodeError, KeyError):
                continue

    new_pairs = []
    attempts = 0
    while len(new_pairs) < n and attempts < n * 6:
        attempts += 1
        tmpl = random.choice(TEMPLATES)
        en, yoda = tmpl()
        key = en.lower()
        if key in seen_en:
            continue
        seen_en.add(key)
        new_pairs.append({"en": en, "yoda": yoda})

    print(f"Generated {len(new_pairs)} new pairs (in {attempts} attempts).")

    with out_path.open("a") as f:
        for pair in new_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"Appended to {out_path}. Total unique 'en' keys now: {len(seen_en)}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 18000
    main(n=n)

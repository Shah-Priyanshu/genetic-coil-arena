"""Builds the GoalSpec schema, few-shot examples and gold-labelled test set (JSONL).

Gold-label conventions (also given to the models in the system prompt):
  eaters : the snakes the instruction is addressed to / that should go eat.
           If no snake is named, all three.
  region : where on the board; 'any' if none stated.
  avoid  : snakes explicitly told to stay away / keep clear / not go there.
           Never filled by an unstated implication ("only red eats" -> avoid []).
  tempo  : 'fast' for hurry/quick/speed/snappy; 'cautious' for careful/slow/safe/
           don't rush/take it easy; otherwise 'normal'.
  hold   : true only when told to wait/hold/freeze/stop/stay put/stand by/pause.
Arrays are compared as sets (order-insensitive).
"""
import json, pathlib

HERE = pathlib.Path(__file__).parent
COLOURS = ["red", "green", "blue"]
REGIONS = ["any", "top_left", "top_right", "bottom_left", "bottom_right",
           "top", "bottom", "left", "right"]
TEMPOS = ["normal", "fast", "cautious"]

SCHEMA = {
    "type": "object",
    "properties": {
        "eaters": {"type": "array", "items": {"type": "string", "enum": COLOURS},
                   "minItems": 1, "uniqueItems": True},
        "region": {"type": "string", "enum": REGIONS},
        "avoid": {"type": "array", "items": {"type": "string", "enum": COLOURS},
                  "uniqueItems": True},
        "tempo": {"type": "string", "enum": TEMPOS},
        "hold": {"type": "boolean"},
    },
    "required": ["eaters", "region", "avoid", "tempo", "hold"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You convert instructions for a cooperative squad of three snakes (red, green, blue) "
    "into a JSON GoalSpec. Output only JSON.\n"
    "- eaters: the snakes the instruction is addressed to / that should go eat (all three if none is named).\n"
    "- region: any, top_left, top_right, bottom_left, bottom_right, top, bottom, left, right "
    "(any if no place is given; 'half' or 'side' means top/bottom/left/right, 'corner' means the diagonal ones).\n"
    "- avoid: snakes explicitly told to stay away / keep clear. Leave empty unless the instruction says so.\n"
    "- tempo: fast if told to hurry/be quick; cautious if told to be careful/slow/not rush; otherwise normal.\n"
    "- hold: true only if told to wait/hold/freeze/stop/stay put."
)

def spec(e, r, a, t, h):
    cmap = {"r": "red", "g": "green", "b": "blue"}
    rmap = {"any": "any", "tl": "top_left", "tr": "top_right", "bl": "bottom_left",
            "br": "bottom_right", "top": "top", "bot": "bottom", "left": "left", "right": "right"}
    tmap = {"n": "normal", "f": "fast", "c": "cautious"}
    order = lambda s: [c for c in COLOURS if c in [cmap[x] for x in s]]
    return {"eaters": order(e), "region": rmap[r], "avoid": order(a), "tempo": tmap[t], "hold": h}

FEWSHOT = [
    ("Red, go eat in the top-left corner.", spec("r", "tl", "", "n", False)),
    ("Green and blue, hurry to the bottom half and grab food; red keep out of the way.",
     spec("gb", "bot", "r", "f", False)),
    ("Everybody wait where you are and be careful.", spec("rgb", "any", "", "c", True)),
    ("Blue eats on the right side, green must stay away from there.",
     spec("b", "right", "g", "n", False)),
]

# ---------- templated (40): explicit tuples, rendered with fixed templates ----------
RPHRASE = {"any": "anywhere", "tl": "in the top-left corner", "tr": "in the top-right corner",
           "bl": "in the bottom-left corner", "br": "in the bottom-right corner",
           "top": "in the top half", "bot": "in the bottom half", "left": "in the left half",
           "right": "in the right half"}
CNAME = {"r": "red", "g": "green", "b": "blue"}

def names(s, style):
    cs = [CNAME[x] for x in "rgb" if x in s]
    if len(cs) == 3:
        return "all three snakes" if style % 2 == 0 else "red, green and blue"
    if len(cs) == 1:
        return cs[0]
    return " and ".join(cs)

def verb_phrase(cs):
    return "stay away" if len(cs) > 1 or True else "stays away"

def render(i, e, r, a, t, h):
    E = names(e, i)
    A = names(a, i) if a else ""
    where = RPHRASE[r]
    v = i % 4
    parts = []
    if h:
        if v == 0:   parts.append(f"Hold position: {E}, {where}." if r != "any" else f"Hold position: {E}.")
        elif v == 1: parts.append(f"{E.capitalize()} must hold {where}." if r != "any" else f"{E.capitalize()} must hold.")
        elif v == 2: parts.append(f"Order: {E} wait {where}." if r != "any" else f"Order: {E} wait.")
        else:        parts.append(f"{E.capitalize()}, freeze {where}." if r != "any" else f"{E.capitalize()}, freeze.")
    else:
        if v == 0:   parts.append(f"Send {E} to eat {where}.")
        elif v == 1: parts.append(f"{E.capitalize()} should eat {where}.")
        elif v == 2: parts.append(f"Order: {E} eat {where}.")
        else:        parts.append(f"{E.capitalize()}, go eat {where}.")
    if a:
        if v == 0:   parts.append(f"{A.capitalize()} must stay away.")
        elif v == 1: parts.append(f"{A.capitalize()} should keep clear.")
        elif v == 2: parts.append(f"Avoid: {A}.")
        else:        parts.append(f"{A.capitalize()}, stay away.")
    if t == "f":
        parts.append(["Move fast.", "Be quick.", "Tempo: fast.", "Hurry."][v])
    elif t == "c":
        parts.append(["Be cautious.", "Go slowly and carefully.", "Tempo: cautious.", "Take care."][v])
    else:
        parts.append(["Normal speed.", "", "Tempo: normal.", ""][v])
    return " ".join(p for p in parts if p)

# (eaters, region, avoid, tempo, hold)
TEMPLATED = [
    ("r", "tr", "", "n", False), ("g", "bl", "", "f", False), ("b", "br", "", "c", False),
    ("rg", "top", "", "n", False), ("gb", "left", "r", "f", False), ("rb", "right", "g", "c", False),
    ("rgb", "any", "", "n", False), ("rgb", "bot", "", "f", False), ("r", "any", "gb", "n", False),
    ("g", "tl", "b", "c", False), ("b", "tr", "rg", "f", False), ("rg", "br", "b", "n", False),
    ("rb", "bl", "", "c", False), ("g", "top", "", "f", False), ("rgb", "left", "", "c", False),
    ("r", "right", "", "f", False), ("gb", "tl", "", "n", False), ("b", "bot", "r", "c", False),
    ("rgb", "tr", "", "f", False), ("r", "bl", "g", "f", False),
    # hold items
    ("rgb", "any", "", "n", True), ("rgb", "any", "", "c", True), ("r", "tl", "", "n", True),
    ("gb", "bot", "", "f", True), ("rg", "right", "", "c", True), ("b", "br", "", "n", True),
    ("rgb", "top", "", "f", True), ("g", "left", "", "n", True), ("rb", "bl", "", "c", True),
    ("rgb", "tr", "", "n", True),
    # more mixed
    ("g", "any", "", "c", False), ("b", "any", "", "f", False), ("rb", "any", "g", "n", False),
    ("rg", "tl", "", "f", False), ("r", "top", "b", "c", False), ("gb", "br", "", "c", False),
    ("b", "left", "rg", "n", False), ("rgb", "bl", "", "n", False), ("rb", "top", "", "f", False),
    ("g", "right", "r", "f", False),
]

# ---------- natural paraphrases (hand-labelled) ----------
# (text, eaters, region, avoid, tempo, hold)
PARA = [
    # casual
    ("red take the food up in the top right", "r", "tr", "", "n", False),
    ("yo blue n green, grab stuff bottom left, quick!", "gb", "bl", "", "f", False),
    ("all of you, lower right corner, and don't rush it", "rgb", "br", "", "c", False),
    ("green only. left side. slowly.", "g", "left", "", "c", False),
    ("red: eat. others: stay away", "r", "any", "gb", "n", False),
    ("hold it right there, all of you.", "rgb", "any", "", "n", True),
    ("freeze everyone", "rgb", "any", "", "n", True),
    ("all snakes pause", "rgb", "any", "", "n", True),
    ("quick, everyone, top left corner, go go go", "rgb", "tl", "", "f", False),
    ("green and blue, speed it up, upper-left", "gb", "tl", "", "f", False),
    ("everything in the bottom half, all 3 of u, carefully", "rgb", "bot", "", "c", False),
    ("RED GO EAT BOTTOM LEFT FAST", "r", "bl", "", "f", False),
    ("go fast: red, green, blue - right side", "rgb", "right", "", "f", False),
    ("Blue should eat wherever. Everyone else keep away from the food.", "b", "any", "rg", "n", False),
    ("everybody go bottom right, safe and slow", "rgb", "br", "", "c", False),
    ("make it snappy: all three in the bottom half", "rgb", "bot", "", "f", False),
    ("ALL STOP", "rgb", "any", "", "n", True),
    ("blue snake hurry to the top half", "b", "top", "", "f", False),
    ("top-right: blue. quick.", "b", "tr", "", "f", False),
    ("Red go. Anywhere. Fast.", "r", "any", "", "f", False),
    ("green and red, stand by.", "rg", "any", "", "n", True),
    ("red n green eat bottom right; blue avoid", "rg", "br", "b", "n", False),
    ("go left, red, quickly. green and blue keep your distance", "r", "left", "gb", "f", False),
    ("i need green to chill in the bottom-left and not move, super careful", "g", "bl", "", "c", True),
    ("Blue, green: freeze. No moving.", "gb", "any", "", "n", True),
    # terse
    ("green: eat in the top-left corner, quickly", "g", "tl", "", "f", False),
    ("blue and red go to the upper right. quickly. green stay clear", "rb", "tr", "g", "f", False),
    ("carefully, all of you, eat on the top side", "rgb", "top", "", "c", False),
    ("red green blue bottom left corner careful", "rgb", "bl", "", "c", False),
    ("go go go, blue, right half", "b", "right", "", "f", False),
    ("red hold position in the top-left corner", "r", "tl", "", "n", True),
    ("all hold, cautious, bottom-left", "rgb", "bl", "", "c", True),
    ("Squad-wide hold. Top half. Cautious.", "rgb", "top", "", "c", True),
    ("all snakes: eat, any region, regular speed", "rgb", "any", "", "n", False),
    ("eat fast, red and blue, bottom half. green avoid.", "rb", "bot", "g", "f", False),
    ("stay put, all of you, and keep it cautious", "rgb", "any", "", "c", True),
    ("bottom right, red and blue, safe approach, green not allowed there", "rb", "br", "g", "c", False),
    ("green goes to the left, careful, others avoid", "g", "left", "rb", "c", False),
    ("lets go: red green blue, upper half, normal speed", "rgb", "top", "", "n", False),
    # typos
    ("red an blu eatt in the botom right, quikly", "rb", "br", "", "f", False),
    ("grean, go to the top lefft. carefuly", "g", "tl", "", "c", False),
    ("evryone wiat", "rgb", "any", "", "n", True),
    ("reed and gren stay awya from the left, blue eats ther", "b", "left", "rg", "n", False),
    ("all 3 holld position in the upper rigt, slowly", "rgb", "tr", "", "c", True),
    ("gren and blue eat top, red stay away. dont rush", "gb", "top", "r", "c", False),
    ("carefuly now, only blue eats, over on the right.", "b", "right", "", "c", False),
    ("blu, go eat in the lowr left, hurry", "b", "bl", "", "f", False),
    # verbose / polite
    ("Please have the red snake and the blue snake collect food from the upper half of the board, as fast as they can, and ask the green one to stay clear of that area.",
     "rb", "top", "g", "f", False),
    ("Could you send the green snake to the top-right region, and tell the red and blue snakes not to go near it?",
     "g", "tr", "rb", "n", False),
    ("Hey team, I'd like the red and the green snake to head over to the bottom-right and eat as quickly as possible, while the blue snake should please avoid that corner entirely.",
     "rg", "br", "b", "f", False),
    ("Green and blue forage on the left side, cautiously, and red stays well clear of it.", "gb", "left", "r", "c", False),
    ("Red eats at the bottom, slow and steady, green keep away please, blue as well.", "r", "bot", "gb", "c", False),
    ("Blue: eat top left. Red & green: stay away from the top left. Move with care.", "b", "tl", "rg", "c", False),
    ("Hold position, red and blue, in the lower-right corner, and take it easy.", "rb", "br", "", "c", True),
    ("Just the red snake, please. Upper right. Carefully.", "r", "tr", "", "c", False),
    ("Red, wait in the bottom-right area. Blue and green steer clear of it. Take extra care.", "r", "br", "gb", "c", True),
    ("Pause. Green and blue stay put in the bottom-left; red keep clear.", "gb", "bl", "r", "n", True),
    ("Red and green should be speedy and eat in the left half, blue keep out.", "rg", "left", "b", "f", False),
    ("Would the green and the blue snakes please go and eat on the left half? Take it slow.", "gb", "left", "", "c", False),
    ("I want green to eat in the top left at a normal pace, and red and blue keep clear.", "g", "tl", "rb", "n", False),
    ("Wherever the food is, it doesn't matter, blue and green just eat there at a normal pace.", "gb", "any", "", "n", False),
    ("Red and blue, freeze in the top-left, and be quick about it.", "rb", "tl", "", "f", True),
    ("Hold still, red and green, top right, and stay careful. Blue, go away.", "rg", "tr", "b", "c", True),
    ("Take it slow, red, in the lower-left. Green and blue stay out of that corner.", "r", "bl", "gb", "c", False),
    ("Red goes bottom-right, eats, and that's it. Normal pace.", "r", "br", "", "n", False),
    ("Don't let green or blue near the top-left; red eats there, normally.", "r", "tl", "gb", "n", False),
    ("Plain and simple: green eats. That's it.", "g", "any", "", "n", False),
    ("Hurry hurry, eat on the right side, red and blue; green avoid.", "rb", "right", "g", "f", False),
    ("Green, wait in the top half, carefully.", "g", "top", "", "c", True),
    ("Can green and red hold in place, please?", "rg", "any", "", "n", True),
    ("Red, green and blue: the lower left corner, hold there, and watch your pace - be cautious.", "rgb", "bl", "", "c", True),
    ("The blue snake alone should go eat in the bottom half, quickly, and the others must not go there.", "b", "bot", "rg", "f", False),
    ("Everyone eat on the left, and be fast about it.", "rgb", "left", "", "f", False),
    ("Red and blue head for the upper-left, no rush, green keep out.", "rb", "tl", "g", "c", False),
    ("red, green: top right, hurry, blue not allowed.", "rg", "tr", "b", "f", False),
    ("Everyone hold in the bottom-right corner, carefully.", "rgb", "br", "", "c", True),
    ("blue, quickly, take the left; red and green stay out", "b", "left", "rg", "f", False),
    ("gree an red: top half, slow plz", "rg", "top", "", "c", False),
    ("Hold the squad still, please, upper-left.", "rgb", "tl", "", "n", True),
    ("green eats in the lower right corner and red and blue stay far away, hurry up", "g", "br", "rb", "f", False),
]

def build():
    items = []
    for i, tup in enumerate(TEMPLATED):
        e, r, a, t, h = tup
        items.append({"id": f"T{i:02d}", "kind": "templated", "text": render(i, *tup),
                      "gold": spec(e, r, a, t, h)})
    for i, (text, e, r, a, t, h) in enumerate(PARA):
        items.append({"id": f"P{i:02d}", "kind": "paraphrase", "text": text,
                      "gold": spec(e, r, a, t, h)})
    # held-out: (tempo, hold, has_avoid) triple AND region both unseen in the few-shots
    seen_triples = {(g["tempo"], g["hold"], bool(g["avoid"])) for _, g in FEWSHOT}
    seen_regions = {g["region"] for _, g in FEWSHOT}
    for it in items:
        g = it["gold"]
        it["heldout"] = ((g["tempo"], g["hold"], bool(g["avoid"])) not in seen_triples
                         and g["region"] not in seen_regions)
        # sanity: gold is schema-valid, eaters/avoid disjoint
        assert g["eaters"] and set(g["eaters"]).isdisjoint(g["avoid"]), it
    texts = [it["text"] for it in items]
    assert len(set(texts)) == len(texts), "duplicate instruction text"
    assert not set(texts) & {t for t, _ in FEWSHOT}
    return items

if __name__ == "__main__":
    items = build()
    with open(HERE / "p1_testset.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    (HERE / "p1_schema.json").write_text(json.dumps(SCHEMA, indent=2), encoding="utf-8")
    (HERE / "p1_fewshot.json").write_text(json.dumps(
        [{"text": t, "gold": g} for t, g in FEWSHOT], indent=2), encoding="utf-8")
    from collections import Counter
    print("items", len(items), Counter(i["kind"] for i in items), "heldout", sum(i["heldout"] for i in items))
    print("regions", Counter(i["gold"]["region"] for i in items))
    print("tempo", Counter(i["gold"]["tempo"] for i in items))
    print("hold", Counter(i["gold"]["hold"] for i in items), "avoid-nonempty", sum(bool(i["gold"]["avoid"]) for i in items))
    print("n eaters", Counter(len(i["gold"]["eaters"]) for i in items))

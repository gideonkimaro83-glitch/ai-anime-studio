"""Episode script generation: local Ollama LLM with a deterministic built-in fallback.

A scene is a list of beats:
  {"type": "narration", "text": ...}
  {"type": "dialogue", "character": ..., "line": ...}
  {"type": "action", "character": attacker, "target": defender, "move": technique name}
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import urllib.error
import urllib.request

from characters import kit_for

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "600"))

MOODS = ["calm", "hopeful", "tense", "mysterious", "action", "triumphant", "melancholy"]

MOOD_KEYWORDS = {
    "action": ["action", "battle", "fight", "chase", "intense", "explosive", "combat", "clash"],
    "triumphant": ["triumph", "victory", "heroic", "epic", "celebrat", "glorious", "climactic"],
    "tense": ["tense", "danger", "fear", "dread", "suspense", "threat", "anxious", "urgent"],
    "mysterious": ["myster", "eerie", "strange", "secret", "magic", "dream", "ominous", "uncanny"],
    "melancholy": ["sad", "melanchol", "grief", "lonely", "bittersweet", "loss", "somber", "despair"],
    "hopeful": ["hope", "warm", "bright", "uplift", "joy", "cheer", "determin", "inspir", "wonder"],
    "calm": ["calm", "peace", "quiet", "gentle", "serene", "relax", "tranquil", "cozy"],
}

# max beats per scene / max action beats per scene
LENGTHS = {"short": (3, 0), "standard": (6, 1), "long": (12, 3)}

REFERENCES = {
    "taisho": ("Take tone and visual inspiration from Demon Slayer: Taisho-era Japan, demon-hunting swordsmen "
               "in patterned haori jackets, named elemental sword techniques (water, flame, thunder, wind...), "
               "night battles in mountain forests, and deep emotional family bonds."),
    "medieval": ("Take tone and visual inspiration from The Seven Deadly Sins: bright medieval fantasy kingdoms, "
                 "knights in shining armor with capes, powerful magic, cozy taverns, comedic found-family banter "
                 "mixed with epic battles."),
}


def normalize_mood(text: str) -> str:
    t = (text or "").lower()
    if t in MOODS:
        return t
    for mood, words in MOOD_KEYWORDS.items():
        if any(w in t for w in words):
            return mood
    return "calm"


def _clean(value, limit: int, default: str = "") -> str:
    s = re.sub(r"\s+", " ", str(value if value is not None else "")).strip()
    if not s:
        return default
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def _length(req) -> str:
    v = getattr(req, "scene_length", "standard")
    return v if v in LENGTHS else "standard"


# --------------------------------------------------------------------------- Ollama


def ollama_status() -> dict:
    status = {"available": False, "model": OLLAMA_MODEL, "model_installed": False, "url": OLLAMA_URL}
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return status
    status["available"] = True
    names = [m.get("name", "") for m in data.get("models", [])]
    wanted = OLLAMA_MODEL if ":" in OLLAMA_MODEL else f"{OLLAMA_MODEL}:latest"
    status["model_installed"] = wanted in names
    status["installed_models"] = names
    return status


def _build_prompt(req) -> str:
    max_beats, max_actions = LENGTHS[_length(req)]
    kit = kit_for(f"{req.genre} {req.style}")
    reference = REFERENCES.get(kit, "")
    if reference:
        reference += (" Use these works ONLY as inspiration for tone and style. Invent everything: never use their "
                      "characters, names, places, organizations or technique names.")
    action_rule = (f"Conflict scenes include 1 to {max_actions} action beats (named attacks between two characters)."
                   if max_actions else "Do not include action beats.")
    return f"""You are a professional anime screenwriter. Write an ORIGINAL anime episode script.
Do not use existing copyrighted characters, names, or franchises.
{reference}

Episode title: {req.title}
Premise: {req.premise}
Genre: {req.genre}
Visual style: {req.style}
Number of scenes: exactly {req.scenes}

Respond with ONLY a JSON object in this exact shape:
{{
  "title": "string",
  "logline": "one sentence summary",
  "characters": [
    {{"name": "string", "role": "protagonist|mentor|rival|friend|antagonist",
      "appearance": "hair color and style, eye color, outfit, weapon, element",
      "personality": "short description"}}
  ],
  "scenes": [
    {{
      "title": "short scene title",
      "setting": "where the scene takes place",
      "mood": "one of: {', '.join(MOODS)}",
      "description": "2-3 sentences describing the visuals",
      "beats": [
        {{"type": "narration", "text": "short narration"}},
        {{"type": "dialogue", "character": "character name", "line": "spoken line under 120 characters"}},
        {{"type": "action", "character": "attacker name", "target": "defender name", "move": "original technique name"}}
      ]
    }}
  ]
}}

Rules:
- 3 to 5 characters. Each appearance must mention hair color and style, eye color, an outfit (a patterned haori,
  armor with a cape, a robe, or a jacket), a weapon (katana, broadsword, staff, spear, or fists) and an element
  (water, flame, thunder, wind, moon, shadow, light, ice, or earth).
- Exactly {req.scenes} scenes forming a complete arc (setup, rising conflict, climax, resolution).
- Each scene has {max(3, max_beats - 3)} to {max_beats} beats, mostly dialogue. {action_rule}
- Every dialogue and action beat uses names from the character list."""


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except ValueError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _call_ollama(req) -> dict:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": _build_prompt(req),
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.8, "num_ctx": 8192},
    }
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return _extract_json(data.get("response", ""))


# --------------------------------------------------------------------------- fallback story

NAMES = {
    "japanese": {
        "hero": ["Aoi Tachibana", "Haruto Seno", "Mei Kanzaki", "Ren Amano", "Sora Hidaka", "Yuki Morimoto",
                 "Kaito Ishikawa", "Hana Sakuraba", "Riku Nanase", "Akari Fujimura", "Nao Kirishima", "Itsuki Tamura"],
        "mentor": ["Master Ishida", "Grandmother Tsune", "Captain Mori", "Old Genji", "Sister Ayame", "Master Kuroda"],
        "rival": ["Kuro Shiranui", "Shiden Akatsuki", "Lady Mikage", "Takeshi Onodera", "Ryoma Kagari", "Setsuna Hiiragi"],
        "friend": ["Tomo Hayashi", "Chiyo Asakura", "Daichi Mizuno", "Suzu Okamoto", "Kenta Arai"],
    },
    "western": {
        "hero": ["Elian Thorne", "Aria Vale", "Cedric Ashford", "Lyra Fenwick", "Rowan Gale", "Isolde Brightwater"],
        "mentor": ["Archmage Corvin", "Sir Aldous", "Dame Rosalind", "Old Bram the Sage"],
        "rival": ["Lord Malric", "Seraphine Noct", "Garrick the Crimson", "Vael Duskbane"],
        "friend": ["Pip Willow", "Tamsin Reed", "Benedict Hollow", "Nell Ambers"],
    },
}

WORLDS = {
    "taisho": dict(
        places=["a moonlit mountain forest", "the wisteria-lined mountain path", "a lantern-lit village at dusk",
                "an old mansion with sliding paper doors", "a misty bamboo grove", "the mountain shrine at midnight",
                "the riverside town during the summer festival"],
        foe="the demons", foe_name="Horned Demon", names="japanese"),
    "medieval": dict(
        places=["the royal capital's grand plaza", "a cozy roadside tavern", "the misty enchanted forest",
                "the ruined fortress on the cliffs", "the castle throne hall", "rolling green hills beneath a giant sky"],
        foe="the rogue knights", foe_name="Rogue Knight Captain", names="western"),
    "scifi": dict(
        places=["the neon-lit lower city", "an orbital station corridor", "a rooftop beneath holographic billboards",
                "the abandoned mech hangar", "the data spire core", "a quiet maglev platform at night"],
        foe="the rogue machines", foe_name="Rogue Unit X-9", names="japanese"),
    "school": dict(
        places=["the school rooftop", "a cherry-blossom lined street", "the after-school club room",
                "a small seaside café", "the summer festival grounds", "the riverbank at sunset"],
        foe="the rival gang", foe_name="Delinquent Captain", names="japanese"),
    "horror": dict(
        places=["a fog-covered shrine", "the empty school hallway at night", "an old library archive",
                "a flickering streetlight alley", "the sealed basement", "the bell tower at midnight"],
        foe="the things in the dark", foe_name="The Pale Thing", names="japanese"),
    "fantasy": dict(
        places=["the lantern-lit market town", "an ancient forest shrine", "a cliffside watchtower",
                "the floating ruins above the clouds", "the town square at dusk", "a moonlit lake"],
        foe="the shadow beasts", foe_name="Shadow Beast", names="japanese"),
}

SENTENCE_STARTERS = {"A", "An", "The", "In", "On", "At", "After", "When", "While", "One", "Two", "Three", "Every",
                     "Some", "Someone", "Something", "This", "That", "His", "Her", "Their", "Our", "My", "Years",
                     "Long", "During", "Before", "As", "If", "Deep", "High", "Far", "Each"}


def _world(genre: str, style: str) -> dict:
    kit = kit_for(f"{genre} {style}")
    if kit in WORLDS:
        return WORLDS[kit]
    g = genre.lower()
    if any(k in g for k in ("sci", "space", "cyber", "mech", "robot", "future", "punk")):
        return WORLDS["scifi"]
    if any(k in g for k in ("school", "slice", "romance", "comedy", "sport", "life")):
        return WORLDS["school"]
    if any(k in g for k in ("horror", "mystery", "thriller", "ghost", "supernatural")):
        return WORLDS["horror"]
    return WORLDS["fantasy"]


# (story order, priority, title, mood, description, items)
# items: ("n", text) narration | ("s", role, line) dialogue | ("a", attacker_role, target_role) action
BEATS = [
    (0, 1, "An Ordinary Morning", "calm",
     "{place_cap}. {hero} moves through a quiet routine, unaware that everything is about to change.",
     [("s", "hero", "Another normal day... I guess that's fine."),
      ("s", "friend", "{hero_first}! You're late again. Everyone is waiting for you!"),
      ("n", "Morning light spills over {place}. Somewhere far away, a bell rings."),
      ("s", "mentor", "{hero_first}, ordinary days are the ones most worth protecting."),
      ("a", "hero", "mentor"),
      ("s", "mentor", "Faster. Your feet are still thinking before you do."),
      ("s", "hero", "One of these days I'll land a real hit on you!"),
      ("s", "mentor", "Not yet. But the wind has changed. Keep your eyes open.")]),
    (1, 4, "The Sign", "mysterious",
     "The calm breaks at {place}. {premise_clause}",
     [("s", "hero", "Did you feel that? Something just changed."),
      ("s", "friend", "Tell me I'm not the only one seeing this!"),
      ("n", "The lanterns flicker. A cold silence falls over {place}."),
      ("s", "rival", "So the rumors were true. It has finally begun."),
      ("s", "hero", "Who are you? What do you want with this place?"),
      ("s", "rival", "Want? I only came to watch {foe} wake up."),
      ("s", "friend", "{hero_first}... I don't like this. Let's get out of here."),
      ("s", "hero", "No. If something is coming, I'm not running.")]),
    (2, 5, "Awakening", "hopeful",
     "{foe_cap} pour into {place}. {hero} stands alone between them and the people.",
     [("s", "hero", "Everyone, get behind me!"),
      ("a", "hero", "foe"),
      ("s", "hero", "I... I can actually do this?"),
      ("s", "mentor", "That power answers courage, not certainty. Remember that."),
      ("s", "friend", "That was amazing! Since when could you do that?"),
      ("s", "hero", "Since about five seconds ago, I think."),
      ("n", "But in the shadows, someone is watching."),
      ("s", "rival", "Interesting. Very interesting.")]),
    (3, 8, "Lessons Under the Sky", "hopeful",
     "Training at {place}: failed attempts, scraped knees and small victories as the sun crosses the sky.",
     [("s", "mentor", "Again. Your focus slips the moment you doubt yourself."),
      ("s", "hero", "Then I'll stop doubting. One more time!"),
      ("a", "mentor", "hero"),
      ("a", "hero", "mentor"),
      ("s", "mentor", "...Good. That one was real."),
      ("n", "Days pass. Sunrise to sunset, the training never stops."),
      ("s", "friend", "You two have been at it for nine hours. I brought rice balls!"),
      ("s", "hero", "Best. Friend. Ever.")]),
    (4, 7, "First Clash", "action",
     "{rival} blocks the road at {place}. The air itself seems to hold its breath.",
     [("s", "rival", "You're not ready for what's coming."),
      ("s", "hero", "Maybe not. But I'm not running away."),
      ("a", "rival", "hero"),
      ("s", "rival", "Too slow. Is that everything you've got?"),
      ("a", "hero", "rival"),
      ("s", "hero", "Not even close!"),
      ("s", "rival", "Then show me you're worth the trouble."),
      ("a", "rival", "hero"),
      ("s", "friend", "{hero_first}! Hang in there!")]),
    (5, 9, "What Lies Beneath", "mysterious",
     "Hidden in {place}, an old seal glows in the dark, revealing why the power chose {hero}.",
     [("s", "mentor", "This was sealed away long before you were born."),
      ("s", "hero", "So it was never an accident... it chose me."),
      ("s", "mentor", "It chose your family. Your mother carried this same light."),
      ("s", "hero", "Why didn't you ever tell me?"),
      ("s", "mentor", "Because I hoped you would never need to know."),
      ("n", "The seal pulses once, like a heartbeat."),
      ("s", "rival", "Touching. But the seal is already breaking.")]),
    (6, 10, "Between Friends", "calm",
     "A quiet night at {place}. {hero} and {friend} share food and talk about who they want to become.",
     [("s", "friend", "Whatever happens, you don't have to carry it alone."),
      ("s", "hero", "Thanks. I think I needed to hear that."),
      ("n", "Fireflies drift between them."),
      ("s", "friend", "When this is over, let's go to the festival. Promise?"),
      ("s", "hero", "Promise. I'll win you the biggest prize there."),
      ("s", "friend", "You can't even win at cards."),
      ("s", "hero", "...The second biggest prize, then.")]),
    (7, 6, "The Fall", "melancholy",
     "At {place}, everything goes wrong at once.",
     [("a", "rival", "hero"),
      ("s", "hero", "It wasn't enough... I wasn't enough."),
      ("s", "rival", "Stay down. This is where your story ends."),
      ("a", "rival", "mentor"),
      ("s", "mentor", "Run, {hero_first}! Live, and grow stronger!"),
      ("s", "hero", "No... not like this!"),
      ("n", "Rain falls as the lights of {place} go out one by one.")]),
    (8, 11, "The Chase", "action",
     "A desperate pursuit through {place} as {hero} races to reach the heart of the danger in time.",
     [("s", "friend", "Go! I'll hold them off here!"),
      ("a", "friend", "foe"),
      ("s", "hero", "I'll come back for you. That's a promise!"),
      ("a", "hero", "foe"),
      ("s", "friend", "You'd better! You still owe me a festival prize!")]),
    (9, 12, "Calm Before the Storm", "tense",
     "Night settles over {place}. Everyone prepares in silence, knowing tomorrow decides everything.",
     [("s", "hero", "No matter what happens tomorrow... thank you, everyone."),
      ("s", "mentor", "You've grown. More than you know."),
      ("s", "friend", "We're doing this together. No arguments."),
      ("s", "rival", "Rest while you can. I won't hold back."),
      ("s", "hero", "Good. Neither will I.")]),
    (10, 3, "Everything on the Line", "action",
     "The final battle erupts at {place}. {hero} rises, determined to protect what matters.",
     [("s", "rival", "This ends now!"),
      ("s", "hero", "Everything I care about is right here. I won't let it go!"),
      ("a", "rival", "hero"),
      ("a", "hero", "rival"),
      ("s", "rival", "Impossible... where is this strength coming from?"),
      ("s", "hero", "From everyone who believed in me!"),
      ("a", "hero", "rival"),
      ("n", "A blinding light fills the sky over {place}.")]),
    (11, 2, "A New Dawn", "triumphant",
     "Morning light spills over {place}. The danger has passed, and {hero} looks toward the horizon.",
     [("s", "friend", "We did it. We actually did it!"),
      ("s", "mentor", "Well done. But this is only the beginning."),
      ("s", "rival", "...You fought well. Next time, I'll win."),
      ("s", "hero", "I'll be waiting. Maybe we'll be friends by then."),
      ("s", "hero", "Then I'll be ready for whatever comes next.")]),
]

ELEMENT_PICKS = ["water", "flame", "thunder", "wind", "moon", "shadow", "light", "ice", "earth"]
PATTERN_PICKS = ["wave", "hemp", "tortoiseshell", "checkered", "striped"]
HAIR_PICKS = ["black", "brown", "silver", "red", "blue", "blonde", "purple", "pink", "white", "teal"]
EYE_PICKS = ["amber", "blue", "green", "violet", "red", "gold", "cyan"]
COAT_PICKS = ["navy", "crimson", "teal", "purple", "black", "white", "orange", "green", "gold"]


def _appearances(kit: str, rng: random.Random) -> dict:
    els = rng.sample(ELEMENT_PICKS, 5)
    hair = rng.sample(HAIR_PICKS, 5)
    eyes = [rng.choice(EYE_PICKS) for _ in range(5)]
    coats = rng.sample(COAT_PICKS, 5)
    pats = [rng.choice(PATTERN_PICKS) for _ in range(5)]
    if kit == "taisho":
        return {
            "hero": f"Spiky {hair[0]} hair, {eyes[0]} eyes, a {coats[0]} {pats[0]}-patterned haori over a dark uniform, katana, {els[0]} techniques",
            "mentor": f"Long {hair[1]} hair tied back, {eyes[1]} eyes, a {coats[1]} {pats[1]}-patterned haori, scarred hands, katana, {els[1]} techniques",
            "rival": f"Long {hair[2]} hair, sharp {eyes[2]} eyes, a {coats[2]} haori, a scar across the cheek, katana, {els[2]} techniques",
            "friend": f"Short {hair[3]} hair, round glasses, {eyes[3]} eyes, a {coats[3]} {pats[3]}-patterned haori, katana, {els[3]} techniques",
            "foe": f"Wild {hair[4]} hair, cold {eyes[4]} eyes, a black striped haori, fists, shadow curse",
        }
    if kit == "medieval":
        return {
            "hero": f"Short {hair[0]} hair, {eyes[0]} eyes, silver armor with a {coats[0]} cape, broadsword, {els[0]} magic",
            "mentor": f"Long {hair[1]} hair, {eyes[1]} eyes, a {coats[1]} robe, wooden staff with a glowing orb, {els[1]} magic, elderly",
            "rival": f"Spiky {hair[2]} hair, cold {eyes[2]} eyes, dark armor with a {coats[2]} cape, broadsword, {els[2]} magic",
            "friend": f"{hair[3].capitalize()} ponytail, {eyes[3]} eyes, {coats[3]} armor, spear, {els[3]} magic",
            "foe": f"Wild {hair[4]} hair, cold {eyes[4]} eyes, black armor with a purple cape, broadsword, shadow curse",
        }
    return {
        "hero": f"Windswept {hair[0]} hair, bright {eyes[0]} eyes, a {coats[0]} jacket with a hand-stitched emblem, {els[0]} powers",
        "mentor": f"{hair[1].capitalize()} hair tied back, {eyes[1]} eyes, a long {coats[1]} coat, a carved wooden staff, {els[1]} powers",
        "rival": f"Sharp {hair[2]} hair, cold {eyes[2]} eyes, a {coats[2]} scarf, a faint glowing scar, sword, {els[2]} powers",
        "friend": f"Short {hair[3]} hair, round glasses, an oversized {coats[3]} hoodie, fists, {els[3]} powers",
        "foe": f"Wild {hair[4]} hair, cold {eyes[4]} eyes, a black coat, fists, shadow powers",
    }


def _seeded_rng(*parts: str) -> random.Random:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _select_items(items, length: str):
    max_beats, max_actions = LENGTHS[length]
    if length == "short":
        return [it for it in items if it[0] == "s"][:2]
    out, actions = [], 0
    for it in items:
        if it[0] == "a":
            if actions >= max_actions:
                continue
            actions += 1
        out.append(it)
        if len(out) >= max_beats:
            break
    return out


def fallback_script(req) -> dict:
    rng = _seeded_rng(req.title, req.premise, req.genre, req.style)
    world = _world(req.genre, req.style)
    kit = kit_for(f"{req.genre} {req.style}")
    pool = NAMES[world["names"]]
    names = {role: rng.choice(pool[role]) for role in ("hero", "mentor", "rival", "friend")}
    names["foe"] = world["foe_name"]
    hero = names["hero"]
    length = _length(req)

    premise = _clean(req.premise, 220).rstrip(".!?… ") or "an ordinary life is about to change"
    first_word = premise.split()[0]
    clause = premise[0].lower() + premise[1:] if first_word in SENTENCE_STARTERS else premise
    places = world["places"][:]
    rng.shuffle(places)

    chosen = sorted(sorted(BEATS, key=lambda b: b[1])[: req.scenes], key=lambda b: b[0])
    scenes = []
    for i, (_, _, title, mood, desc, items) in enumerate(chosen):
        place = places[i % len(places)]
        fmt = {"place": place, "place_cap": place[0].upper() + place[1:], "hero": hero,
               "hero_first": hero.split()[0], "rival": names["rival"], "friend": names["friend"],
               "mentor": names["mentor"], "foe": world["foe"], "foe_cap": world["foe"][0].upper() + world["foe"][1:],
               "premise_clause": f"It begins when {clause}."}
        beats = []
        for it in _select_items(items, length):
            if it[0] == "n":
                beats.append({"type": "narration", "text": it[1].format(**fmt)})
            elif it[0] == "s":
                beats.append({"type": "dialogue", "character": names[it[1]], "line": it[2].format(**fmt)})
            else:
                beats.append({"type": "action", "character": names[it[1]], "target": names[it[2]], "move": ""})
        scenes.append({
            "number": i + 1, "title": title, "setting": place[0].upper() + place[1:], "mood": mood,
            "description": desc.format(**fmt), "beats": beats,
            "dialogue": [b for b in beats if b["type"] == "dialogue"],
        })

    looks = _appearances(kit, rng)
    roles = {"hero": ("protagonist", "Kind-hearted and stubborn; grows braver each scene"),
             "mentor": ("mentor", "Patient, dry humor, hides an old regret"),
             "rival": ("rival", "Proud and relentless, tests the hero to see if they are worthy"),
             "friend": ("friend", "Loyal, talkative, braver than they look"),
             "foe": ("minion", "A dangerous enemy serving a darker power")}
    used = {b.get(k) for s in scenes for b in s["beats"] for k in ("character", "target")}
    characters = [{"name": names[r], "role": roles[r][0], "appearance": looks[r], "personality": roles[r][1]}
                  for r in ("hero", "mentor", "rival", "friend", "foe") if r != "foe" or names["foe"] in used]
    return {
        "title": _clean(req.title, 120, "Untitled Episode"),
        "logline": f"{premise}. An original {req.genre.lower()} episode in {req.scenes} scenes.",
        "genre": req.genre,
        "style": req.style,
        "scene_length": length,
        "characters": characters,
        "scenes": scenes,
    }


# --------------------------------------------------------------------------- normalization


def _parse_beats(s: dict, names: list[str], max_beats: int, max_actions: int) -> list[dict]:
    raw = s.get("beats")
    if not isinstance(raw, list):
        raw = [{"type": "dialogue", **d} for d in (s.get("dialogue") or []) if isinstance(d, dict)]
    beats, actions = [], 0
    for b in raw:
        if not isinstance(b, dict):
            continue
        kind = str(b.get("type", "dialogue")).lower()
        who = b.get("character") or b.get("speaker") or b.get("name") or b.get("attacker")
        if kind.startswith("act") or kind in ("attack", "fight", "battle"):
            if actions >= max_actions or not who:
                continue
            target = b.get("target") or b.get("defender") or next((n for n in names if n != who), None)
            if not target or target == who:
                continue
            beats.append({"type": "action", "character": _clean(who, 40), "target": _clean(target, 40),
                          "move": _clean(b.get("move") or b.get("technique"), 40)})
            actions += 1
        elif kind.startswith("narr"):
            text = b.get("text") or b.get("line")
            if text:
                beats.append({"type": "narration", "text": _clean(text, 220)})
        else:
            line = b.get("line") or b.get("text")
            if who and line:
                beats.append({"type": "dialogue", "character": _clean(who, 40), "line": _clean(line, 160)})
        if len(beats) >= max_beats:
            break
    return beats


def _normalize_ai(raw: dict, req, fallback: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("model did not return a JSON object")
    max_beats, max_actions = LENGTHS[_length(req)]
    chars = []
    for c in raw.get("characters") or []:
        if isinstance(c, dict) and c.get("name"):
            chars.append({
                "name": _clean(c.get("name"), 40),
                "role": _clean(c.get("role"), 30, "character"),
                "appearance": _clean(c.get("appearance"), 260, "Distinctive anime character design"),
                "personality": _clean(c.get("personality"), 220, "Complex and determined"),
            })
    chars = chars[:6] or fallback["characters"]
    names = [c["name"] for c in chars]

    # Only full names are swapped, plus the protagonist's given name (used alone in template lines);
    # swapping other first tokens would clobber titles like "Master" or "Captain".
    rename = {fc["name"]: chars[i]["name"] for i, fc in enumerate(fallback["characters"]) if i < len(chars)}
    hero_old, hero_new = fallback["characters"][0]["name"], chars[0]["name"]
    used_fallback = False

    def renamed(text: str) -> str:
        nonlocal used_fallback
        used_fallback = True
        for old, new in rename.items():
            text = text.replace(old, new)
        if hero_old != hero_new:
            text = text.replace(hero_old.split()[0], hero_new.split()[0])
        return text

    def renamed_beats(beats):
        return [{k: (renamed(v) if isinstance(v, str) and k != "type" else v) for k, v in b.items()} for b in beats]

    scenes = []
    for s in (raw.get("scenes") or [])[: req.scenes]:
        if not isinstance(s, dict):
            continue
        scenes.append({
            "title": _clean(s.get("title"), 60, f"Scene {len(scenes) + 1}"),
            "setting": _clean(s.get("setting"), 90, "An unnamed place"),
            "mood": normalize_mood(str(s.get("mood", ""))),
            "description": _clean(s.get("description"), 400, "The story continues."),
            "beats": _parse_beats(s, names, max_beats, max_actions),
        })

    for i in range(len(scenes), req.scenes):
        fs = fallback["scenes"][i]
        scenes.append({**fs, "description": renamed(fs["description"]), "beats": renamed_beats(fs["beats"])})
    for i, s in enumerate(scenes):
        s["number"] = i + 1
        if not any(b["type"] == "dialogue" for b in s["beats"]):
            s["beats"] = renamed_beats(fallback["scenes"][i]["beats"])
        s["dialogue"] = [b for b in s["beats"] if b["type"] == "dialogue"]

    # Template characters that had no AI counterpart but now appear in padded scenes.
    if used_fallback:
        present = {c["name"] for c in chars}
        appearing = {b.get(k) for s in scenes for b in s["beats"] for k in ("character", "target")}
        chars += [fc for fc in fallback["characters"] if fc["name"] not in present and fc["name"] in appearing]

    return {
        "title": _clean(raw.get("title"), 120, fallback["title"]),
        "logline": _clean(raw.get("logline"), 300, fallback["logline"]),
        "genre": req.genre,
        "style": req.style,
        "scene_length": _length(req),
        "characters": chars,
        "scenes": scenes,
    }


def generate_script(req) -> tuple[dict, str, str]:
    """Return (script, source, note). source is 'ollama' or 'fallback'."""
    fallback = fallback_script(req)
    if not req.use_ai:
        return fallback, "fallback", "Local AI disabled; used the built-in story template."
    status = ollama_status()
    if not status["available"]:
        return fallback, "fallback", f"Ollama not reachable at {OLLAMA_URL}; used the built-in story template."
    if not status["model_installed"]:
        return fallback, "fallback", (f"Model '{OLLAMA_MODEL}' is not installed (run: ollama pull {OLLAMA_MODEL}); "
                                      "used the built-in story template.")
    try:
        raw = _call_ollama(req)
        return _normalize_ai(raw, req, fallback), "ollama", f"Script written locally by {OLLAMA_MODEL}."
    except Exception as exc:  # noqa: BLE001 - any model/network failure should fall back
        return fallback, "fallback", f"Ollama generation failed ({type(exc).__name__}: {exc}); used the template."

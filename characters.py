"""Procedural anime-style characters: designs, heads, full-body posed figures, and portraits.

Designs are deterministic: the same name + description always yields the same character, so a
character looks identical in every scene. Keywords in the appearance text are honored, e.g.
"silver hair", "crimson scarf", "glasses", "scar", "ponytail", "wave-patterned haori", "armor",
"robe", "katana", "staff", "spear", "flame techniques". Anything unspecified is picked from the name.

Two visual kits are supported, chosen from the genre/style text:
  taisho   - swordsman look: dark uniform + patterned haori, katana with element-tinted blade
  medieval - knight look: plate armor + cape (or mage robe), broadsword / staff / spear
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BUST_W, BUST_H = 400, 520
SS = 2  # supersampling factor for smooth edges

COLORS = {
    "black": (38, 34, 48), "dark": (48, 42, 58), "raven": (30, 30, 45), "brown": (112, 72, 46),
    "chestnut": (122, 66, 40), "auburn": (150, 62, 40), "blonde": (242, 208, 124), "blond": (242, 208, 124),
    "golden": (236, 192, 84), "gold": (236, 192, 84), "silver": (206, 212, 226), "white": (236, 236, 242),
    "grey": (162, 162, 174), "gray": (162, 162, 174), "red": (192, 48, 52), "crimson": (172, 28, 48),
    "scarlet": (202, 38, 42), "blue": (62, 104, 204), "navy": (38, 48, 104), "teal": (42, 150, 150),
    "pink": (240, 142, 182), "green": (62, 152, 92), "emerald": (30, 140, 90), "purple": (122, 72, 172),
    "violet": (142, 92, 204), "orange": (236, 132, 52), "cyan": (72, 200, 222), "amber": (230, 150, 40),
    "copper": (184, 104, 60), "lavender": (186, 160, 226), "yellow": (240, 210, 70),
}
HAIR_NOUNS = {"hair", "ponytail", "braid", "braids", "bun", "locks", "bangs", "mane"}
EYE_NOUNS = {"eyes", "eye", "gaze", "irises"}
OUTFIT_NOUNS = {"jacket", "coat", "hoodie", "uniform", "robe", "robes", "cloak", "armor", "armour", "shirt",
                "dress", "kimono", "sweater", "suit", "vest", "blazer", "tunic", "outfit", "jumpsuit", "haori"}
SKIN_WORDS = {"pale": 0, "fair": 0, "light": 1, "olive": 2, "tan": 2, "tanned": 2, "bronze": 3, "brown": 3,
              "dark": 4, "deep": 4}
SKIN = [(255, 228, 208), (248, 210, 180), (228, 184, 150), (194, 144, 108), (142, 98, 72)]

HAIR_DEFAULT = ["black", "brown", "blonde", "silver", "red", "blue", "pink", "purple", "chestnut", "teal"]
EYE_DEFAULT = ["blue", "amber", "green", "violet", "brown", "red", "cyan", "gold"]
OUTFIT_DEFAULT = ["navy", "red", "teal", "purple", "black", "white", "green", "orange", "brown"]

ELEMENTS = ["water", "flame", "thunder", "wind", "moon", "shadow", "light", "ice", "earth"]
ELEMENT_WORDS = {
    "water": ["water", "wave", "tide", "ocean", "sea", "river", "rain"],
    "flame": ["flame", "fire", "ember", "blaze", "sun", "burning", "inferno"],
    "thunder": ["thunder", "lightning", "storm", "electric", "spark"],
    "wind": ["wind", "gale", "air", "breeze", "feather", "tempest"],
    "moon": ["moon", "lunar", "crescent"],
    "shadow": ["shadow", "void", "curse", "abyss", "umbral"],
    "light": ["holy", "light", "radiant", "divine", "sacred", "angel"],
    "ice": ["ice", "frost", "snow", "winter", "glacier"],
    "earth": ["earth", "stone", "rock", "quake", "iron"],
}
ELEMENT_COLORS = {  # main, light, core
    "water": ((40, 110, 220), (150, 210, 255), (255, 255, 255)),
    "flame": ((230, 70, 30), (255, 170, 60), (255, 240, 180)),
    "thunder": ((240, 200, 40), (255, 240, 140), (255, 255, 255)),
    "wind": ((50, 180, 130), (170, 250, 210), (240, 255, 250)),
    "moon": ((130, 90, 220), (200, 170, 255), (255, 240, 255)),
    "shadow": ((90, 25, 110), (190, 50, 90), (255, 150, 170)),
    "light": ((240, 190, 70), (255, 235, 160), (255, 255, 255)),
    "ice": ((90, 190, 240), (200, 240, 255), (255, 255, 255)),
    "earth": ((150, 100, 50), (215, 175, 115), (255, 235, 200)),
}
PATTERN_WORDS = {"checkered": "ichimatsu", "checker": "ichimatsu", "wave": "seigaiha", "hemp": "asanoha",
                 "star": "asanoha", "tortoiseshell": "kikko", "hexagon": "kikko", "striped": "stripes"}

LINE = (32, 24, 38, 255)
LW = 3 * SS


def _rng(name: str) -> random.Random:
    return random.Random(int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:16], 16))


def _color_before(tokens: list[str], nouns: set[str]):
    for i, t in enumerate(tokens):
        if t in nouns:
            for w in reversed(tokens[max(0, i - 3):i]):
                if w in COLORS:
                    return COLORS[w]
    return None


def _shade(c, f: float):
    return tuple(max(0, min(255, int(v * f))) for v in c[:3])


def _mix(a, b, t: float):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def kit_for(context: str) -> str:
    c = (context or "").lower()
    if any(k in c for k in ("taish", "slayer", "swordsman", "demon", "samurai", "katana", "ukiyo", "edo")):
        return "taisho"
    if any(k in c for k in ("knight", "sins", "medieval", "holy", "kingdom", "paladin")):
        return "medieval"
    return "default"


def design_for(char: dict, context: str = "") -> dict:
    name = str(char.get("name") or "?")
    role = str(char.get("role") or "").lower()
    text = f"{char.get('appearance') or ''} {char.get('personality') or ''}".lower()
    tokens = re.findall(r"[a-z]+", str(char.get("appearance") or "").lower())
    words = set(tokens)
    all_words = set(re.findall(r"[a-z]+", text))
    rng = _rng(name)
    kit = kit_for(context)

    if words & {"ponytail", "tied"}:
        style = "ponytail"
    elif "bun" in words:
        style = "bun"
    elif "long" in words:
        style = "long"
    elif words & {"spiky", "messy", "windswept", "wild", "unruly", "tousled"}:
        style = "spiky"
    elif words & {"short", "cropped", "buzzed"}:
        style = "short"
    else:
        style = rng.choice(["long", "short", "spiky", "ponytail", "short"])

    skin = None
    for i, t in enumerate(tokens):
        if t in ("skin", "skinned"):
            for w in tokens[max(0, i - 2):i]:
                if w in SKIN_WORDS:
                    skin = SKIN[SKIN_WORDS[w]]
    scarf = None
    if "scarf" in words:
        scarf = _color_before(tokens, {"scarf"}) or COLORS[rng.choice(["red", "teal", "gold", "white"])]

    element = next((e for e in ELEMENTS if any(w in all_words for w in ELEMENT_WORDS[e])), None)
    element = element or rng.choice(ELEMENTS)

    # weapon
    if words & {"staff", "wand", "scepter"} or (words & {"mage", "wizard", "sorcerer", "witch"}):
        weapon = "staff"
    elif words & {"spear", "lance", "halberd", "naginata"}:
        weapon = "spear"
    elif words & {"broadsword", "greatsword", "claymore"}:
        weapon = "broadsword"
    elif words & {"katana", "blade", "sword", "nichirin"}:
        weapon = "broadsword" if kit == "medieval" else "katana"
    elif words & {"fists", "gauntlets", "brawler", "unarmed"}:
        weapon = "none"
    elif kit == "taisho":
        weapon = "katana"
    elif kit == "medieval":
        weapon = {"mentor": "staff", "friend": "spear"}.get(role, "broadsword")
    else:
        weapon = rng.choice(["katana", "broadsword", "staff", "none"])

    # outfit
    if "haori" in words or (kit == "taisho" and not words & {"armor", "armour", "robe", "robes"}):
        outfit_kind = "haori"
    elif words & {"armor", "armour", "plate", "knight", "paladin"} or (kit == "medieval" and weapon != "staff"):
        outfit_kind = "armor"
    elif words & {"robe", "robes", "cloak"} or (kit == "medieval" and weapon == "staff"):
        outfit_kind = "robe"
    else:
        outfit_kind = "jacket"

    pattern = next((PATTERN_WORDS[w] for w in tokens if w in PATTERN_WORDS), None)
    pattern = pattern or rng.choice(["seigaiha", "asanoha", "kikko", "ichimatsu", "stripes", "plain"])
    outfit = _color_before(tokens, OUTFIT_NOUNS) or COLORS[rng.choice(OUTFIT_DEFAULT)]
    pattern2 = _mix(outfit, (255, 255, 255) if sum(outfit) < 360 else (20, 20, 30), 0.55)

    return {
        "name": name, "kit": kit, "style": style,
        "hair": _color_before(tokens, HAIR_NOUNS) or COLORS[rng.choice(HAIR_DEFAULT)],
        "eyes": _color_before(tokens, EYE_NOUNS) or COLORS[rng.choice(EYE_DEFAULT)],
        "outfit": outfit, "pattern": pattern, "pattern2": pattern2,
        "inner": (34, 34, 46) if outfit_kind == "haori" else _shade(outfit, 0.45),
        "outfit_kind": outfit_kind,
        "cape": scarf or COLORS[rng.choice(["red", "navy", "purple", "crimson", "teal"])],
        "skin": skin or rng.choice(SKIN),
        "scarf": scarf if outfit_kind != "armor" else None,
        "glasses": bool(words & {"glasses", "spectacles", "monocle"}),
        "scar": "scar" in words,
        "older": "mentor" in role or bool(words & {"old", "elderly", "aged", "wrinkled", "bearded"}),
        "sharp": any(r in role for r in ("rival", "antagonist", "villain")) or bool(words & {"sharp", "cold", "piercing"}),
        "element": element, "weapon": weapon,
        "jitter": [rng.uniform(-8, 8) for _ in range(12)],
    }


# --------------------------------------------------------------------------- technique names

STYLE_NAMES = {"water": "Tide", "flame": "Ember", "thunder": "Thunderclap", "wind": "Gale", "moon": "Crescent",
               "shadow": "Umbral", "light": "Radiant", "ice": "Frost", "earth": "Bedrock"}
MOVE_NAMES = {
    "water": ["Rising Current", "Spiral Undertow", "Crashing Tide", "Still Mirror Cut"],
    "flame": ["Cinder Arc", "Blazing Wheel", "Phoenix Rise", "Sunset Severance"],
    "thunder": ["Split-Sky Flash", "Sixfold Spark", "Rolling Thunder", "Heaven's Drum"],
    "wind": ["Whirling Blades", "Gale Spiral", "Sky Dragon Gust", "Severing Breeze"],
    "moon": ["Pale Crescent", "Moonlit Bloom", "Eclipse Ring", "Waning Arc"],
    "shadow": ["Night Fang", "Hollow Veil", "Black Lotus", "Abyss Coil"],
    "light": ["Radiant Judgment", "Halo Breaker", "Dawn Lance", "Sanctum Ray"],
    "ice": ["Frost Petal", "Glacier Fang", "White Silence", "Shard Storm"],
    "earth": ["Stone Fist", "Mountain Breaker", "Quake Step", "Iron Root"],
}
ORDINALS = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth"]


def technique_name(d: dict, n: int, move: str = "") -> tuple[str, str]:
    """Return (kicker, move name) for an attack callout."""
    el = d["element"]
    move = move or MOVE_NAMES[el][n % 4]
    if d["kit"] == "medieval":
        return f"{STYLE_NAMES[el].upper()} ART", move
    return f"{STYLE_NAMES[el].upper()} STYLE  ·  {ORDINALS[(n + 2) % len(ORDINALS)].upper()} FORM", move


# --------------------------------------------------------------------------- head / bust


def draw_bust(d: dict, eyes_open: bool = True, mouth_open: bool = False) -> Image.Image:
    img = Image.new("RGBA", (BUST_W * SS, BUST_H * SS), (0, 0, 0, 0))
    g = ImageDraw.Draw(img, "RGBA")

    def P(*xy):
        return [v * SS for v in xy]

    hair, hair_dk = d["hair"] + (255,), _shade(d["hair"], 0.72) + (255,)
    skin, skin_dk = d["skin"] + (255,), _shade(d["skin"], 0.86) + (255,)
    outfit, outfit_dk = d["outfit"] + (255,), _shade(d["outfit"], 0.7) + (255,)
    style, j = d["style"], d["jitter"]

    if style == "long":
        g.polygon(P(102, 150, 90, 300, 82, 478, 140, 456, 200, 478, 260, 456, 318, 478, 310, 300, 298, 150),
                  fill=hair_dk, outline=LINE, width=LW)
    elif style == "ponytail":
        g.polygon(P(250, 120, 332, 150, 356, 262, 338, 396, 302, 304, 288, 200), fill=hair_dk, outline=LINE, width=LW)
    elif style == "bun":
        g.ellipse(P(156, 36, 244, 116), fill=hair_dk, outline=LINE, width=LW)

    g.polygon(P(36, 520, 50, 440, 110, 396, 290, 396, 350, 440, 364, 520), fill=outfit, outline=LINE, width=LW)
    g.polygon(P(176, 296, 224, 296, 228, 400, 172, 400), fill=skin_dk, outline=LINE, width=LW)
    g.polygon(P(172, 398, 200, 452, 228, 398), fill=skin_dk)
    g.polygon(P(146, 396, 200, 476, 174, 398), fill=outfit_dk, outline=LINE, width=LW)
    g.polygon(P(254, 396, 200, 476, 226, 398), fill=outfit_dk, outline=LINE, width=LW)
    if d["scarf"]:
        sc = d["scarf"] + (255,)
        g.polygon(P(214, 398, 246, 398, 258, 486, 226, 490), fill=_shade(d["scarf"], 0.8) + (255,), outline=LINE, width=LW)
        g.rounded_rectangle(P(146, 366, 254, 414), radius=20 * SS, fill=sc, outline=LINE, width=LW)

    if style == "spiky":
        for tri in ((118, 136, 88 + j[0], 66, 152, 100), (158, 94, 166 + j[1], 26, 206, 82),
                    (200, 80, 252 + j[2], 30, 248, 96), (252, 100, 314, 72 + j[3], 288, 144)):
            g.polygon(P(*tri), fill=hair, outline=LINE, width=LW)

    g.ellipse(P(100, 70, 300, 262), fill=hair, outline=LINE, width=LW)
    g.ellipse(P(104, 204, 130, 256), fill=skin, outline=LINE, width=LW)
    g.ellipse(P(270, 204, 296, 256), fill=skin, outline=LINE, width=LW)

    face = []
    for a in range(180, 361, 10):
        r = math.radians(a)
        face += [200 + 84 * math.cos(r), 215 + 107 * math.sin(r)]
    face += [276, 262, 250, 306, 200, 336, 150, 306, 124, 262]
    g.polygon(P(*face), fill=skin, outline=LINE, width=LW)
    g.ellipse(P(138, 270, 176, 287), fill=(255, 120, 140, 60))
    g.ellipse(P(224, 270, 262, 287), fill=(255, 120, 140, 60))

    iris, pupil = d["eyes"] + (255,), _shade(d["eyes"], 0.45) + (255,)
    top, bot = (-14, 18) if d["sharp"] else (-22, 26)
    for cx, o in ((163, -1), (237, 1)):
        cy = 238
        if eyes_open:
            g.ellipse(P(cx - 25, cy + top, cx + 25, cy + bot), fill=(255, 255, 255, 255))
            g.ellipse(P(cx - 16, cy + top, cx + 16, cy + bot), fill=iris)
            g.ellipse(P(cx - 8, cy + top + 12, cx + 8, cy + bot - 10), fill=pupil)
            g.ellipse(P(cx - 12, cy + top + 6, cx - 2, cy + top + 16), fill=(255, 255, 255, 255))
            g.ellipse(P(cx + 5, cy + bot - 14, cx + 10, cy + bot - 9), fill=(255, 255, 255, 230))
            g.line(P(cx + 28 * o, cy + top + 6, cx + 12 * o, cy + top - 4, cx - 10 * o, cy + top - 4,
                     cx - 26 * o, cy + top + 2), fill=LINE, width=6 * SS, joint="curve")
            g.line(P(cx - 12, cy + bot + 2, cx + 12, cy + bot + 2), fill=_shade(LINE, 1.6) + (200,), width=2 * SS)
        else:
            g.arc(P(cx - 24, cy - 10, cx + 24, cy + 18), 15, 165, fill=LINE, width=5 * SS)
        brow = (cx + 22 * o, cy - 48, cx - 18 * o, cy - 36) if d["sharp"] else (cx + 22 * o, cy - 38, cx - 18 * o, cy - 44)
        g.line(P(*brow), fill=_shade(d["hair"], 0.5) + (255,), width=5 * SS)
        if d["older"]:
            g.line(P(cx + 30 * o, cy + 6, cx + 38 * o, cy + 12), fill=_shade(d["skin"], 0.7) + (255,), width=2 * SS)

    g.line(P(200, 266, 196, 278), fill=_shade(d["skin"], 0.7) + (255,), width=2 * SS)
    if mouth_open:
        g.ellipse(P(189, 288, 211, 310), fill=(150, 48, 62, 255), outline=LINE, width=2 * SS)
        g.ellipse(P(193, 300, 207, 309), fill=(226, 110, 120, 255))
    else:
        g.arc(P(186, 282, 214, 300), 20, 160, fill=LINE, width=3 * SS)

    if d["glasses"]:
        for cx in (163, 237):
            g.rounded_rectangle(P(cx - 32, 206, cx + 32, 270), radius=14 * SS, outline=(40, 40, 52, 255), width=3 * SS)
            g.rounded_rectangle(P(cx - 30, 208, cx + 30, 268), radius=13 * SS, fill=(200, 230, 255, 35))
        g.line(P(195, 236, 205, 236), fill=(40, 40, 52, 255), width=3 * SS)
    if d["scar"]:
        g.line(P(236, 258, 262, 290), fill=(196, 84, 84, 255), width=3 * SS)

    lock_end = 250 if style in ("short", "spiky") else 316
    g.polygon(P(104, 156, 94, lock_end, 122, lock_end + 18, 130, 200), fill=hair, outline=LINE, width=LW)
    g.polygon(P(296, 156, 306, lock_end, 278, lock_end + 18, 270, 200), fill=hair, outline=LINE, width=LW)
    bangs = [104, 206, 108, 140, 140, 94, 200, 80, 260, 94, 292, 140, 296, 206]
    tips = [(275, 160), (255, 200), (232, 150), (210, 194), (188, 148), (165, 198), (142, 154), (120, 202)]
    for k, (x, y) in enumerate(tips):
        bangs += [x + j[4 + k] * 0.6, y + (j[4 + k] if y > 170 else 0)]
    g.polygon(P(*bangs), fill=hair, outline=LINE, width=LW)
    g.arc(P(132, 96, 268, 190), 205, 250, fill=(255, 255, 255, 95), width=6 * SS)

    return img.resize((BUST_W, BUST_H), Image.LANCZOS)


def dimmed(img: Image.Image, amount: float = 0.42) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    arr[..., :3] = arr[..., :3] * (1 - amount) + np.array([16, 16, 36], np.float32) * amount
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


# --------------------------------------------------------------------------- full-body figure
# Canvas in base units; the figure faces screen-right. Angles: 0 = straight down, 90 = forward
# (screen right), 180 = up, -90 = backward.

FIG_W, FIG_H = 640, 980
NECK = (320, 250)
SHOULDER_B, SHOULDER_F = (256, 286), (384, 286)
HIP_B, HIP_F = (294, 500), (346, 500)
HEAD_SCALE = 0.5
HEAD_CROP = (40, 20, 360, 350)  # in bust units; chin ~ (160, 316) inside the crop

UPPER_ARM, FOREARM, THIGH, SHIN = 118, 108, 186, 180

POSES = {
    #          back arm       front arm       back leg        front leg      weapon(abs) lean  flap drawn
    "idle":    dict(b=(-10, -4), f=(12, 26), lb=(-5, -3), lf=(7, 4), w=None, lean=0, flap=0, drawn=False),
    "idle2":   dict(b=(-6, 2), f=(16, 40), lb=(-3, -2), lf=(12, 9), w=None, lean=1.5, flap=0, drawn=False),
    "talk":    dict(b=(-10, -4), f=(38, 118), lb=(-5, -3), lf=(7, 4), w=None, lean=2, flap=0, drawn=False),
    "talk2":   dict(b=(-28, -62), f=(22, 64), lb=(-6, -4), lf=(9, 6), w=None, lean=-1, flap=0, drawn=False),
    "walk1":   dict(b=(22, 44), f=(-24, -10), lb=(-22, -50), lf=(26, 14), w=None, lean=4, flap=8, drawn=False),
    "walk2":   dict(b=(-22, -8), f=(24, 46), lb=(24, 12), lf=(-20, -52), w=None, lean=4, flap=8, drawn=False),
    "ready":   dict(b=(36, 96), f=(30, 100), lb=(-24, -8), lf=(30, 16), w=142, lean=5, flap=4, drawn=True),
    "run1":    dict(b=(46, 108), f=(-42, -18), lb=(-38, -80), lf=(52, 20), w=-112, lean=16, flap=30, drawn=True),
    "run2":    dict(b=(-34, -6), f=(36, 100), lb=(46, 16), lf=(-32, -84), w=150, lean=16, flap=30, drawn=True),
    "wind":    dict(b=(158, 208), f=(164, 214), lb=(-30, -10), lf=(36, 24), w=232, lean=-7, flap=10, drawn=True),
    "slash":   dict(b=(70, 96), f=(76, 90), lb=(-46, -46), lf=(56, 6), w=96, lean=13, flap=26, drawn=True),
    "slash_up": dict(b=(128, 158), f=(140, 166), lb=(-40, -40), lf=(48, 10), w=176, lean=7, flap=18, drawn=True),
    "guard":   dict(b=(100, 150), f=(110, 160), lb=(-26, -10), lf=(26, 12), w=268, lean=-4, flap=6, drawn=True),
    "hurt":    dict(b=(-72, -104), f=(60, 92), lb=(-16, -6), lf=(32, 42), w=22, lean=-15, flap=-14, drawn=True),
    "cast":    dict(b=(64, 100), f=(150, 172), lb=(-20, -8), lf=(22, 12), w=180, lean=3, flap=6, drawn=True),
    "victory": dict(b=(-14, -6), f=(168, 178), lb=(-8, -4), lf=(10, 6), w=180, lean=0, flap=4, drawn=True),
}


def _vec(a: float):
    r = math.radians(a)
    return math.sin(r), math.cos(r)


def _limb(g, p0, ang, length, w0, w1, fill, P):
    dx, dy = _vec(ang)
    p1 = (p0[0] + dx * length, p0[1] + dy * length)
    nx, ny = -dy, dx
    g.ellipse(P(p0[0] - w0 / 2, p0[1] - w0 / 2, p0[0] + w0 / 2, p0[1] + w0 / 2), fill=fill, outline=LINE, width=LW)
    g.polygon(P(p0[0] + nx * w0 / 2, p0[1] + ny * w0 / 2, p1[0] + nx * w1 / 2, p1[1] + ny * w1 / 2,
                p1[0] - nx * w1 / 2, p1[1] - ny * w1 / 2, p0[0] - nx * w0 / 2, p0[1] - ny * w0 / 2),
              fill=fill, outline=LINE, width=LW)
    g.ellipse(P(p1[0] - w1 / 2, p1[1] - w1 / 2, p1[0] + w1 / 2, p1[1] + w1 / 2), fill=fill)
    return p1


def _weapon(g, hand, ang, d, P):
    kind = d["weapon"]
    if kind == "none":
        return
    dx, dy = _vec(ang)
    nx, ny = -dy, dx

    def pt(t, o=0.0):
        return (hand[0] + dx * t + nx * o, hand[1] + dy * t + ny * o)

    def poly(*pairs, fill):
        flat = []
        for t, o in pairs:
            flat += list(pt(t, o))
        g.polygon(P(*flat), fill=fill, outline=LINE, width=LW)

    el_main, el_light, _ = ELEMENT_COLORS[d["element"]]
    steel = _mix((222, 228, 240), el_main, 0.42 if d["kit"] != "medieval" else 0.12) + (255,)
    if kind == "katana":
        poly((-48, -6), (14, -6), (14, 6), (-48, 6), fill=(40, 30, 34, 255))
        for k in range(-40, 10, 14):
            poly((k, -6), (k + 6, 6), (k + 3, 6), (k - 3, -6), fill=(230, 220, 200, 255))
        poly((12, -15), (20, -15), (20, 15), (12, 15), fill=(70, 60, 50, 255))
        poly((20, -6), (214, -7), (240, 2), (214, 5), (20, 6), fill=steel)
        a, b = pt(24, -3), pt(212, -4)
        g.line(P(*a, *b), fill=el_light + (255,), width=2 * SS)
    elif kind == "broadsword":
        c = pt(-50, 0)
        g.ellipse(P(c[0] - 10, c[1] - 10, c[0] + 10, c[1] + 10), fill=(214, 170, 70, 255), outline=LINE, width=LW)
        poly((-44, -7), (10, -7), (10, 7), (-44, 7), fill=(90, 60, 40, 255))
        poly((10, -36), (20, -36), (20, 36), (10, 36), fill=(214, 170, 70, 255))
        poly((20, -12), (212, -11), (248, 0), (212, 11), (20, 12), fill=steel)
        a, b = pt(26, 0), pt(205, 0)
        g.line(P(*a, *b), fill=_shade(steel, 0.75) + (255,), width=3 * SS)
    elif kind == "staff":
        poly((-140, -6), (226, -6), (226, 6), (-140, 6), fill=(120, 80, 50, 255))
        c = pt(250, 0)
        g.ellipse(P(c[0] - 34, c[1] - 34, c[0] + 34, c[1] + 34), fill=el_main + (60,))
        g.ellipse(P(c[0] - 21, c[1] - 21, c[0] + 21, c[1] + 21), fill=el_light + (255,), outline=LINE, width=LW)
        g.ellipse(P(c[0] - 10, c[1] - 14, c[0] - 2, c[1] - 6), fill=(255, 255, 255, 230))
        poly((222, -16), (232, -16), (232, 16), (222, 16), fill=(214, 170, 70, 255))
    elif kind == "spear":
        poly((-120, -5), (256, -5), (256, 5), (-120, 5), fill=(110, 76, 48, 255))
        poly((252, -14), (306, 0), (252, 14), fill=steel)
        poly((246, -9), (256, -9), (256, 9), (246, 9), fill=(214, 170, 70, 255))


_PATTERN_CACHE: dict = {}


def _pattern_canvas(d: dict) -> Image.Image:
    key = (d["pattern"], d["outfit"], d["pattern2"])
    if key in _PATTERN_CACHE:
        return _PATTERN_CACHE[key]
    w, h = FIG_W * SS, FIG_H * SS
    c1, c2 = d["outfit"] + (255,), d["pattern2"] + (255,)
    img = Image.new("RGBA", (w, h), c1)
    g = ImageDraw.Draw(img)
    s = SS
    p = d["pattern"]
    if p == "ichimatsu":
        step = 30 * s
        for y in range(0, h, step):
            for x in range(0, w, step):
                if (x // step + y // step) % 2:
                    g.rectangle((x, y, x + step, y + step), fill=c2)
    elif p == "seigaiha":
        r, row = 26 * s, 13 * s
        for k, y in enumerate(range(0, h + r, row)):
            off = r if k % 2 else 0
            for x in range(-r + off, w + r, 2 * r):
                for i, rr in enumerate((r, int(r * 0.72), int(r * 0.46), int(r * 0.22))):
                    g.ellipse((x - rr, y - rr, x + rr, y + rr), fill=c2 if i % 2 == 0 else c1)
    elif p == "asanoha":
        step = 40 * s
        for y in range(0, h + step, step):
            for x in range(0, w + step, step):
                cx, cy = x + step // 2, y + step // 2
                for ex, ey in ((x, y), (x + step, y), (x, y + step), (x + step, y + step),
                               (cx, y), (cx, y + step), (x, cy), (x + step, cy)):
                    g.line((cx, cy, ex, ey), fill=c2, width=2 * s)
    elif p == "kikko":
        r = 22 * s
        hx, hy = r * 1.5, r * math.sqrt(3)
        for i in range(int(w / hx) + 2):
            for jy in range(int(h / hy) + 2):
                cx, cy = i * hx, jy * hy + (hy / 2 if i % 2 else 0)
                pts = [(cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))) for a in range(0, 360, 60)]
                g.polygon(pts, outline=c2, width=3 * s)
    elif p == "stripes":
        step = 24 * s
        for x in range(0, w, step * 2):
            g.rectangle((x, 0, x + step, h), fill=c2)
    _PATTERN_CACHE[key] = img
    return img


def _poly_mask(points, P) -> Image.Image:
    m = Image.new("L", (FIG_W * SS, FIG_H * SS), 0)
    ImageDraw.Draw(m).polygon(P(*points), fill=255)
    return m


def draw_body(d: dict, pose_name: str):
    """Return (back_layer, front_layer, foot_y) at base resolution for a pose (no head)."""
    pose = POSES[pose_name]
    back = Image.new("RGBA", (FIG_W * SS, FIG_H * SS), (0, 0, 0, 0))
    front = Image.new("RGBA", back.size, (0, 0, 0, 0))
    gb, gf = ImageDraw.Draw(back, "RGBA"), ImageDraw.Draw(front, "RGBA")

    def P(*xy):
        return [v * SS for v in xy]

    kind = d["outfit_kind"]
    skin = d["skin"] + (255,)
    outfit = d["outfit"] + (255,)
    inner = d["inner"] + (255,)
    flap = pose["flap"]
    sleeve_f = {"haori": outfit, "armor": (196, 202, 216, 255), "robe": outfit, "jacket": outfit}[kind]
    sleeve_b = _shade(sleeve_f, 0.78) + (255,)
    leg_col = {"haori": inner, "armor": (150, 156, 172, 255), "robe": _shade(d["outfit"], 0.6) + (255,),
               "jacket": (44, 44, 60, 255)}[kind]
    boot_col = (236, 236, 236, 255) if kind == "haori" else (70, 46, 34, 255)
    arm_w0, arm_w1 = (44, 38) if kind in ("haori", "robe") else (32, 26)

    # cape and long hair behind everything
    if kind == "armor":
        cape = d["cape"] + (255,)
        gb.polygon(P(250, 280, 390, 280, 420 - flap * 0.3, 560, 440 - flap, 790, 200 - flap, 800, 222 - flap * 0.3, 560),
                   fill=_shade(d["cape"], 0.8) + (255,), outline=LINE, width=LW)
        gb.polygon(P(262, 282, 378, 282, 400 - flap * 0.3, 560, 410 - flap, 770, 240 - flap, 776, 244 - flap * 0.3, 560),
                   fill=cape, outline=LINE, width=LW)
    if d["style"] == "long":
        gb.polygon(P(272, 262, 368, 262, 384, 420, 256, 420), fill=_shade(d["hair"], 0.72) + (255,), outline=LINE, width=LW)
    elif d["style"] == "ponytail":
        gb.polygon(P(352, 196, 410, 226, 404, 330, 372, 300), fill=_shade(d["hair"], 0.72) + (255,), outline=LINE, width=LW)

    # sheathed weapon behind the hip
    if not pose["drawn"] and d["weapon"] in ("katana", "broadsword"):
        hip = (300, 492)
        ang = -122 if d["weapon"] == "katana" else -158
        _limb(gb, hip, ang, 206, 16, 14, (46, 32, 36, 255), P)  # scabbard
        _limb(gb, hip, ang + 180, 48, 12, 12, (40, 30, 34, 255), P)  # hilt

    # back arm and legs
    e = _limb(gb, SHOULDER_B, pose["b"][0], UPPER_ARM, arm_w0, arm_w1, sleeve_b, P)
    hand_b = _limb(gb, e, pose["b"][1], FOREARM, arm_w1, arm_w1 - 4, sleeve_b, P)
    gb.ellipse(P(hand_b[0] - 14, hand_b[1] - 14, hand_b[0] + 14, hand_b[1] + 14), fill=_shade(d["skin"], 0.9) + (255,),
               outline=LINE, width=LW)
    feet = []
    for hip, (th, sh), shade in ((HIP_B, pose["lb"], 0.8), (HIP_F, pose["lf"], 1.0)):
        col = _shade(leg_col, shade) + (255,)
        lw0 = 50 if kind == "haori" else 40
        knee = _limb(gb, hip, th, THIGH, lw0, lw0 - 8, col, P)
        ankle = _limb(gb, knee, sh, SHIN, lw0 - 8, 28, col, P)
        if kind == "haori":  # leg wraps
            _limb(gb, (knee[0] * 0.35 + ankle[0] * 0.65, knee[1] * 0.35 + ankle[1] * 0.65), sh, SHIN * 0.35, 30, 28,
                  (232, 232, 236, 255), P)
        fx, fy = ankle
        gb.polygon(P(fx - 14, fy - 8, fx + 36, fy + 4, fx + 38, fy + 18, fx - 14, fy + 18), fill=_shade(boot_col, shade) + (255,),
                   outline=LINE, width=LW)
        feet.append(fy + 18)

    # torso and outer garment
    torso = (252, 282, 388, 282, 378, 420, 372, 506, 268, 506, 262, 420)
    gb.polygon(P(*torso), fill=inner if kind != "armor" else (120, 126, 140, 255), outline=LINE, width=LW)
    if kind == "haori":
        pat = _pattern_canvas(d)
        for panel in ((246, 280, 304, 284, 300, 660, 208 - flap, 646),
                      (336, 284, 394, 280, 432 - flap, 646, 340, 660)):
            back.paste(pat, (0, 0), _poly_mask(panel, P))
            gb.polygon(P(*panel), outline=LINE, width=LW)
        gb.rectangle(P(296, 470, 344, 490), fill=(236, 236, 236, 255), outline=LINE, width=LW)
        for by in (330, 380, 430):
            gb.ellipse(P(316, by, 324, by + 8), fill=(214, 180, 90, 255))
    elif kind == "armor":
        plate = (200, 206, 220, 255)
        gb.polygon(P(256, 284, 384, 284, 374, 420, 362, 480, 278, 480, 266, 420), fill=plate, outline=LINE, width=LW)
        gb.line(P(320, 296, 320, 470), fill=(150, 156, 172, 255), width=3 * SS)
        gb.polygon(P(270, 480, 370, 480, 384, 566, 256, 566), fill=_shade(plate, 0.85) + (255,), outline=LINE, width=LW)
        gb.line(P(262, 522, 378, 522), fill=LINE, width=2 * SS)
        gb.ellipse(P(306, 330, 334, 360), fill=ELEMENT_COLORS[d["element"]][1] + (255,), outline=(214, 170, 70, 255), width=3 * SS)
        gb.polygon(P(256, 284, 384, 284, 382, 296, 258, 296), fill=(214, 170, 70, 255))
        gb.ellipse(P(222, 262, 292, 322), fill=_shade(plate, 0.9) + (255,), outline=LINE, width=LW)
    elif kind == "robe":
        gb.polygon(P(250, 280, 390, 280, 432 - flap * 0.5, 880, 208 - flap, 880), fill=outfit, outline=LINE, width=LW)
        gb.line(P(320, 290, 320 - flap * 0.3, 880), fill=d["pattern2"] + (255,), width=10 * SS)
        gb.rectangle(P(262, 468, 378, 490), fill=d["pattern2"] + (255,), outline=LINE, width=LW)
    else:  # jacket
        gb.polygon(P(248, 280, 392, 280, 398, 560, 242 - flap * 0.4, 560), fill=outfit, outline=LINE, width=LW)
        gb.polygon(P(298, 282, 342, 282, 320, 360), fill=(236, 236, 240, 255), outline=LINE, width=LW)
        gb.line(P(320, 360, 320, 556), fill=LINE, width=2 * SS)
    if d["scarf"]:
        gb.rounded_rectangle(P(270, 252, 370, 296), radius=18 * SS, fill=d["scarf"] + (255,), outline=LINE, width=LW)
        gb.polygon(P(270, 280, 300, 282, 268 - flap * 1.4, 380, 240 - flap * 1.4, 370), fill=_shade(d["scarf"], 0.8) + (255,),
                   outline=LINE, width=LW)
    gb.polygon(P(306, 236, 334, 236, 336, 290, 304, 290), fill=_shade(d["skin"], 0.86) + (255,), outline=LINE, width=LW)

    # front arm + weapon
    e = _limb(gf, SHOULDER_F, pose["f"][0], UPPER_ARM, arm_w0, arm_w1, sleeve_f, P)
    hand = _limb(gf, e, pose["f"][1], FOREARM, arm_w1, arm_w1 - 4, sleeve_f, P)
    w_ang = pose["w"]
    if d["weapon"] in ("staff", "spear") and not pose["drawn"]:
        w_ang = 178
    if w_ang is not None:
        _weapon(gf, hand, w_ang, d, P)
    gf.ellipse(P(hand[0] - 15, hand[1] - 15, hand[0] + 15, hand[1] + 15), fill=skin, outline=LINE, width=LW)
    if kind == "armor":
        gf.ellipse(P(SHOULDER_F[0] - 36, SHOULDER_F[1] - 26, SHOULDER_F[0] + 36, SHOULDER_F[1] + 34),
                   fill=(206, 212, 226, 255), outline=LINE, width=LW)
        gf.arc(P(SHOULDER_F[0] - 30, SHOULDER_F[1] - 20, SHOULDER_F[0] + 30, SHOULDER_F[1] + 28), 200, 340,
               fill=(214, 170, 70, 255), width=3 * SS)

    size = (FIG_W, FIG_H)
    return back.resize(size, Image.LANCZOS), front.resize(size, Image.LANCZOS), max(feet)


class Figure:
    """A character's full-body sprites with caching of heads, bodies and final composites."""

    def __init__(self, design: dict):
        self.d = design
        self._heads: dict = {}
        self._bodies: dict = {}

    def head(self, eyes_open: bool, mouth_open: bool) -> Image.Image:
        key = (eyes_open, mouth_open)
        if key not in self._heads:
            bust = draw_bust(self.d, eyes_open, mouth_open).crop(HEAD_CROP)
            self._heads[key] = bust.resize((int(bust.width * HEAD_SCALE), int(bust.height * HEAD_SCALE)), Image.LANCZOS)
        return self._heads[key]

    def body(self, pose: str):
        if pose not in self._bodies:
            self._bodies[pose] = draw_body(self.d, pose)
        return self._bodies[pose]

    def compose(self, pose: str, eyes_open=True, mouth_open=False):
        """Return (image, anchor_x, anchor_y) where the anchor is the ground point under the hips."""
        back, front, foot_y = self.body(pose)
        img = back.copy()
        head = self.head(eyes_open, mouth_open)
        img.alpha_composite(head, (int(NECK[0] - 160 * HEAD_SCALE), int(NECK[1] - 300 * HEAD_SCALE)))
        img.alpha_composite(front)
        lean = POSES[pose]["lean"]
        if lean:
            img = img.rotate(-lean, resample=Image.BICUBIC, center=(320, foot_y))
        bbox = img.getbbox() or (0, 0, 1, 1)
        img = img.crop(bbox)
        return img, 320 - bbox[0], foot_y - bbox[1]


class Cast:
    """Looks up figures by character name and caches transformed sprites (scale / flip / dim)."""

    def __init__(self, characters: list[dict], context: str = "", max_cache: int = 260):
        self.context = context
        self.figures = {str(c.get("name", "")).strip().lower(): Figure(design_for(c, context)) for c in characters}
        self._cache: OrderedDict = OrderedDict()
        self.max_cache = max_cache

    def figure(self, name: str) -> Figure:
        key = name.strip().lower()
        if key not in self.figures:
            self.figures[key] = Figure(design_for({"name": name}, self.context))
        return self.figures[key]

    def design(self, name: str) -> dict:
        return self.figure(name).d

    def sprite(self, name: str, pose: str = "idle", eyes_open=True, mouth_open=False, scale=1.0,
               flip=False, dim=False):
        key = (name.strip().lower(), pose, eyes_open, mouth_open, scale, flip, dim)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        img, ax, ay = self.figure(name).compose(pose, eyes_open, mouth_open)
        if scale != 1.0:
            img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
            ax, ay = ax * scale, ay * scale
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            ax = img.width - ax
        if dim:
            img = dimmed(img)
        out = (img, ax, ay)
        self._cache[key] = out
        if len(self._cache) > self.max_cache:
            self._cache.popitem(last=False)
        return out


def save_portraits(characters: list[dict], out_dir: Path, prefix: str, context: str = "") -> list[str]:
    names = []
    for i, c in enumerate(characters):
        fname = f"{prefix}_char{i + 1}.png"
        img, _, _ = Figure(design_for(c, context)).compose("idle")
        img.save(out_dir / fname)
        names.append(fname)
    return names

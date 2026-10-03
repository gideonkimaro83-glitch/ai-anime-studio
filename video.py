"""Render an episode script into a 720p MP4: scenery, moving characters, fights, captions and audio."""

from __future__ import annotations

import math
import random
import subprocess
import tempfile
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from characters import ELEMENT_COLORS, Cast, kit_for, technique_name
from fx import draw_flash, draw_speed_lines, draw_trail, make_callout
from textutil import font, wrap

W, H, FPS = 1280, 720, 24
BW, BH = int(W * 1.15), int(H * 1.15)  # oversized background for Ken Burns pan/zoom
SR = 44100

STAGE_SCALE, ACTION_SCALE = 0.74, 0.86
STAGE_GROUND, ACTION_GROUND = H - 14, H - 6
STAGE_SLOTS = {1: [0.5], 2: [0.27, 0.73], 3: [0.18, 0.5, 0.82]}
MAX_ON_STAGE = 3
EXCHANGE = 2.2  # seconds per attack exchange in a fight shot

# top sky, bottom sky, accent, silhouette
PALETTES = {
    "calm": ((34, 52, 110), (240, 170, 150), (255, 236, 200), (30, 30, 62)),
    "hopeful": ((70, 130, 205), (255, 205, 150), (255, 248, 205), (40, 60, 95)),
    "tense": ((40, 10, 25), (170, 50, 40), (255, 125, 85), (15, 5, 10)),
    "mysterious": ((10, 15, 45), (72, 40, 115), (190, 210, 255), (8, 8, 26)),
    "action": ((25, 20, 60), (232, 92, 52), (255, 220, 120), (15, 10, 25)),
    "triumphant": ((60, 92, 182), (255, 190, 92), (255, 245, 200), (30, 36, 72)),
    "melancholy": ((30, 40, 60), (112, 122, 152), (220, 226, 242), (20, 25, 36)),
}
NIGHT_MOODS = {"mysterious", "tense", "melancholy", "action", "calm"}

CHORDS = {
    "calm": [48, 60, 64, 67, 71],
    "hopeful": [41, 53, 57, 60, 67],
    "tense": [45, 57, 60, 64, 65],
    "mysterious": [38, 50, 53, 57, 64],
    "action": [40, 52, 55, 59, 64],
    "triumphant": [43, 55, 59, 62, 67],
    "melancholy": [45, 57, 60, 64, 67],
}

LEN_CFG = {
    "short": dict(lead=0.6, narr=(3.2, 5.0), line=(2.4, 5.0), action=2.8, tail=0.6),
    "standard": dict(lead=1.4, narr=(3.6, 6.0), line=(2.8, 6.0), action=3.2, tail=0.8),
    "long": dict(lead=2.4, narr=(4.5, 8.0), line=(3.2, 7.0), action=5.0, tail=1.2),
}


def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _ease(p):
    p = _clamp(p, 0.0, 1.0)
    return p * p * (3 - 2 * p)


def _look(script: dict) -> str:
    return {"taisho": "ukiyo", "medieval": "bright"}.get(kit_for(f"{script.get('genre', '')} {script.get('style', '')}"),
                                                         "default")


# --------------------------------------------------------------------------- backgrounds

_VIGNETTE = None


def _vignette() -> np.ndarray:
    y, x = np.mgrid[0:BH, 0:BW].astype(np.float32)
    d = np.sqrt(((x - BW / 2) / (BW / 2)) ** 2 + ((y - BH / 2) / (BH / 2)) ** 2)
    return (1 - 0.55 * np.clip(d - 0.55, 0, 1))[..., None]


def _saturate(c, f):
    g = sum(c) / 3
    return tuple(int(_clamp(g + (v - g) * f, 0, 255)) for v in c)


def _has(s: str, *words) -> bool:
    return any(w in s for w in words)


def _mountains(draw, rng, base_y, color, li):
    pts, x, y = [(0, BH)], 0, base_y
    while x <= BW:
        pts.append((x, y))
        x += rng.randint(60, 160)
        y = base_y + rng.randint(-110 + li * 25, 50)
    draw.polygon(pts + [(BW, base_y), (BW, BH)], fill=color)


def _hills(draw, rng, base_y, color, li):
    pts = [(0, BH)]
    ph, amp = rng.uniform(0, 6), 40 - li * 8
    for x in range(0, BW + 20, 20):
        pts.append((x, base_y + amp * math.sin(x / (180 + li * 60) + ph)))
    draw.polygon(pts + [(BW, BH)], fill=color)


def _pines(draw, rng, base_y, color, tall):
    draw.rectangle((0, base_y, BW, BH), fill=color)
    x = -20
    while x < BW:
        h = rng.randint(140, 260) + tall
        w = h * 0.42
        for k in range(3):
            top = base_y - h + k * h * 0.26
            draw.polygon([(x, top), (x - w / 2 * (0.6 + k * 0.2), top + h * 0.42), (x + w / 2 * (0.6 + k * 0.2), top + h * 0.42)],
                         fill=color)
        draw.rectangle((x - 6, base_y - h * 0.2, x + 6, base_y + 4), fill=color)
        x += rng.randint(50, 110)


def _bamboo(draw, rng, color, accent):
    x = rng.randint(-10, 40)
    while x < BW:
        w = rng.randint(14, 24)
        draw.rectangle((x, 0, x + w, BH), fill=color)
        y = rng.randint(30, 120)
        while y < BH:
            draw.line((x, y, x + w, y), fill=_lerp(color, accent, 0.25), width=3)
            y += rng.randint(90, 150)
        x += rng.randint(60, 140)


def _castle(draw, rng, base_y, color, lit, look):
    cx = rng.randint(int(BW * 0.3), int(BW * 0.7))
    if look == "ukiyo":  # tiered Japanese keep
        draw.rectangle((cx - 120, base_y - 40, cx + 120, BH), fill=color)
        w, y = 150, base_y - 40
        for _ in range(4):
            draw.polygon([(cx - w - 26, y), (cx - w + 10, y - 34), (cx + w - 10, y - 34), (cx + w + 26, y)], fill=color)
            draw.rectangle((cx - w + 24, y - 80, cx + w - 24, y - 34), fill=color)
            for wx in range(int(cx - w + 40), int(cx + w - 40), 30):
                if rng.random() < 0.5:
                    draw.rectangle((wx, y - 66, wx + 10, y - 50), fill=lit)
            y -= 80
            w *= 0.72
        draw.polygon([(cx - w - 20, y), (cx, y - 50), (cx + w + 20, y)], fill=color)
        return
    draw.rectangle((cx - 210, base_y - 90, cx + 210, BH), fill=color)
    for bx in range(cx - 210, cx + 210, 24):
        draw.rectangle((bx, base_y - 108, bx + 12, base_y - 90), fill=color)
    for tx, th, tw in ((cx - 210, 230, 60), (cx + 150, 230, 60), (cx - 50, 330, 100)):
        draw.rectangle((tx, base_y - th, tx + tw, BH), fill=color)
        draw.polygon([(tx - 12, base_y - th), (tx + tw / 2, base_y - th - tw * 1.3), (tx + tw + 12, base_y - th)], fill=color)
        for wy in range(int(base_y - th + 30), int(base_y - 20), 46):
            draw.rectangle((tx + tw / 2 - 5, wy, tx + tw / 2 + 5, wy + 18), fill=lit)


def _houses(draw, rng, base_y, color, lit, look):
    x = rng.randint(-60, 0)
    while x < BW:
        w, h = rng.randint(110, 190), rng.randint(70, 130)
        draw.rectangle((x, base_y - h, x + w, BH), fill=color)
        if look == "ukiyo":
            draw.polygon([(x - 22, base_y - h), (x + 18, base_y - h - 46), (x + w - 18, base_y - h - 46), (x + w + 22, base_y - h)],
                         fill=color)
        else:
            draw.polygon([(x - 12, base_y - h), (x + w / 2, base_y - h - w * 0.45), (x + w + 12, base_y - h)], fill=color)
        for wx in range(x + 18, x + w - 26, 40):
            if rng.random() < 0.6:
                draw.rectangle((wx, base_y - h + 22, wx + 18, base_y - h + 46), fill=lit)
        x += w + rng.randint(10, 50)


def _skyline(draw, rng, base_y, color, lit, li):
    x = -rng.randint(0, 40)
    while x < BW:
        bw_, bh_ = rng.randint(50, 130), rng.randint(60, 240 - li * 40)
        draw.rectangle((x, base_y - bh_, x + bw_, BH), fill=color)
        if li == 2:
            for wy in range(base_y - bh_ + 12, base_y + 60, 22):
                for wx in range(x + 8, x + bw_ - 12, 18):
                    if rng.random() < 0.35:
                        draw.rectangle((wx, wy, wx + 7, wy + 10), fill=lit)
        x += bw_ + rng.randint(-10, 12)


def _torii(draw, cx, base_y, color):
    draw.rectangle((cx - 120, base_y - 260, cx - 96, BH), fill=color)
    draw.rectangle((cx + 96, base_y - 260, cx + 120, BH), fill=color)
    draw.rectangle((cx - 140, base_y - 220, cx + 140, base_y - 204), fill=color)
    draw.polygon([(cx - 180, base_y - 278), (cx - 150, base_y - 262), (cx + 150, base_y - 262), (cx + 180, base_y - 278),
                  (cx + 150, base_y - 250), (cx - 150, base_y - 250)], fill=color)


def make_background(mood: str, seed: int, setting: str = "", look: str = "default") -> Image.Image:
    global _VIGNETTE
    top, bottom, accent, sil = PALETTES[mood]
    if look == "bright":
        top, bottom = _saturate(top, 1.35), _saturate(bottom, 1.3)
    rng = random.Random(seed)
    s = setting.lower()

    t = np.linspace(0, 1, BH, dtype=np.float32)[:, None, None]
    sky = np.array(top, np.float32) * (1 - t) + np.array(bottom, np.float32) * t
    img = Image.fromarray(np.broadcast_to(sky, (BH, BW, 3)).astype(np.uint8))
    draw = ImageDraw.Draw(img, "RGBA")

    if mood in NIGHT_MOODS:
        for _ in range(160 if mood == "mysterious" else 70):
            x, y = rng.randrange(BW), rng.randrange(int(BH * 0.55))
            b = rng.randint(150, 255)
            r = rng.choice([1, 1, 1, 2])
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(b, b, b))

    glow = Image.new("RGB", (BW, BH), (0, 0, 0))
    gd = ImageDraw.Draw(glow)
    cx, cy = rng.randint(int(BW * 0.58), int(BW * 0.85)), rng.randint(int(BH * 0.12), int(BH * 0.34))
    radius = rng.randint(45, 85) if look != "ukiyo" else rng.randint(80, 120)
    gd.ellipse((cx - radius * 3, cy - radius * 3, cx + radius * 3, cy + radius * 3), fill=_lerp((0, 0, 0), accent, 0.35))
    glow = glow.filter(ImageFilter.GaussianBlur(radius * 1.4))
    img = Image.fromarray(np.clip(np.asarray(img, np.int16) + np.asarray(glow, np.int16), 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img, "RGBA")
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=accent)

    if look == "bright" and mood not in ("tense", "mysterious"):
        for _ in range(5):
            x, y = rng.randint(0, BW), rng.randint(40, int(BH * 0.35))
            for _ in range(6):
                rr = rng.randint(30, 70)
                ox, oy = x + rng.randint(-80, 80), y + rng.randint(-20, 20)
                draw.ellipse((ox - rr, oy - rr * 0.6, ox + rr, oy + rr * 0.6), fill=(255, 255, 255, 120))
    if look == "ukiyo":
        mist = _lerp(bottom, (255, 255, 255), 0.45)
        for k in range(3):
            y = int(BH * (0.28 + 0.12 * k)) + rng.randint(-20, 20)
            x = rng.randint(-200, 200)
            while x < BW:
                w = rng.randint(220, 420)
                draw.rounded_rectangle((x, y, x + w, y + 34), radius=17, fill=mist + (95,))
                x += w + rng.randint(80, 240)

    lit = _lerp(accent, (255, 230, 160), 0.4)
    forest = _has(s, "forest", "wood", "grove", "jungle", "trees")
    bamboo = "bamboo" in s
    castle = _has(s, "castle", "kingdom", "capital", "fortress", "palace", "throne", "citadel", "keep")
    village = _has(s, "village", "town", "street", "inn", "tavern", "market", "mansion", "house", "festival", "plaza")
    city = _has(s, "city", "station", "rooftop", "tower", "school", "neon", "hangar", "platform", "library", "corridor")
    shrine = _has(s, "shrine", "temple", "torii")
    hills = _has(s, "hill", "meadow", "field", "plain")
    water = _has(s, "lake", "river", "sea", "ocean", "beach", "harbor", "harbour", "shore")

    if bamboo:
        _bamboo(draw, rng, _lerp(_lerp(bottom, top, 0.4), sil, 0.45), accent)
    for li in range(3):
        color = _lerp(_lerp(bottom, top, 0.4), sil, (li + 1) / 3)
        base_y = int(BH * (0.52 + 0.13 * li))
        if li == 0:
            (_hills if hills else _mountains)(draw, rng, base_y, color, li)
        elif li == 1:
            if castle:
                _castle(draw, rng, base_y, color, lit, look)
            elif city:
                _skyline(draw, rng, base_y, color, lit, li)
            elif forest:
                _pines(draw, rng, base_y, color, 60)
            else:
                (_hills if hills or water else _mountains)(draw, rng, base_y, color, li)
        else:
            if village:
                _houses(draw, rng, base_y, color, lit, look)
            elif city:
                _skyline(draw, rng, base_y, color, lit, li)
            elif forest or bamboo:
                _pines(draw, rng, base_y + 40, color, 120)
            else:
                (_hills if hills else _mountains)(draw, rng, base_y, color, li)
    if shrine:
        _torii(draw, rng.randint(int(BW * 0.25), int(BW * 0.75)), int(BH * 0.86), _lerp((150, 30, 30), sil, 0.55))
    if water:
        wy = int(BH * 0.82)
        draw.rectangle((0, wy, BW, BH), fill=_lerp(top, bottom, 0.5) + (255,))
        for _ in range(60):
            x, y = rng.randint(0, BW), rng.randint(wy + 4, BH)
            draw.line((x, y, x + rng.randint(20, 80), y), fill=accent + (rng.randint(60, 140),), width=2)
    if "wisteria" in s:
        for x in range(0, BW, 16):
            length = rng.randint(50, 170)
            for k in range(0, length, 9):
                r = 9 * (1 - k / length) + 2
                draw.ellipse((x - r + rng.randint(-3, 3), k - r, x + r, k + r),
                             fill=(rng.randint(160, 200), rng.randint(120, 150), rng.randint(220, 250), 210))

    arr = np.asarray(img, np.float32)
    if look == "ukiyo":  # warm paper grain
        noise = np.random.default_rng(seed).normal(0, 5, (BH, BW, 1)).astype(np.float32)
        arr = arr * np.array([1.0, 0.97, 0.9], np.float32) + 10 + noise
    if _VIGNETTE is None:
        _VIGNETTE = _vignette()
    return Image.fromarray(np.clip(arr * _VIGNETTE, 0, 255).astype(np.uint8))


def _tinted(bg: Image.Image, element: str) -> Image.Image:
    main = ELEMENT_COLORS[element][0]
    dark = Image.new("RGB", bg.size, _lerp(main, (10, 10, 20), 0.6))
    return Image.blend(bg, dark, 0.5)


# --------------------------------------------------------------------------- overlays


def _header_overlay(scene: dict, accent) -> Image.Image:
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    d.rounded_rectangle((36, 30, 36 + 560, 30 + 112), radius=18, fill=(0, 0, 0, 130))
    d.text((60, 42), f"SCENE {scene['number']:02d}  ·  {scene['mood'].upper()}", font=font(18, True), fill=accent + (255,))
    title = (wrap(scene["title"], font(34, True), 510, 1) or [""])[0]
    d.text((60, 66), title, font=font(34, True), fill=(255, 255, 255, 255))
    setting = (wrap(scene["setting"], font(18), 510, 1) or [""])[0]
    d.text((60, 110), setting, font=font(18), fill=(225, 225, 235, 230))
    return ov


def _line_overlay(label: str, text: str, accent, narration: bool) -> Image.Image:
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    body_font = font(25 if narration else 28, False if narration else True)
    lines = wrap(text, body_font, W - 320, 3) or [""]
    if not narration:  # quote after wrapping so truncation can't eat the closing mark
        lines[0], lines[-1] = "“" + lines[0], lines[-1] + "”"
    box_h = 46 + 36 * len(lines)
    top = H - 26 - box_h
    d.rounded_rectangle((120, top, W - 120, H - 26), radius=20, fill=(8, 8, 20, 190), outline=accent + (150,), width=2)
    tag_font = font(19, True)
    tag_w = int(tag_font.getlength(label)) + 34
    d.rounded_rectangle((150, top - 17, 150 + tag_w, top + 19), radius=12,
                        fill=(accent if not narration else (90, 90, 110)) + (255,))
    d.text((167, top - 11), label, font=tag_font, fill=(15, 15, 25, 255) if not narration else (240, 240, 250, 255))
    for i, line in enumerate(lines):
        d.text((160, top + 26 + i * 36), line, font=body_font,
               fill=(232, 232, 244, 255) if narration else (255, 255, 255, 255))
    return ov


def _card_overlay(heading: str, sub_lines: list[str], kicker: str, accent) -> Image.Image:
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    d.rectangle((0, 0, W, H), fill=(0, 0, 0, 90))
    kf, hf, sf = font(20, True), font(64, True), font(26)
    heads = wrap(heading, hf, W - 200, 2)
    y = H // 2 - (len(heads) * 74 + len(sub_lines) * 36) // 2 - 20
    d.text((W // 2, y - 40), kicker, font=kf, fill=accent + (255,), anchor="mm")
    for h in heads:
        d.text((W // 2 + 3, y + 37), h, font=hf, fill=(0, 0, 0, 160), anchor="mm")
        d.text((W // 2, y + 34), h, font=hf, fill=(255, 255, 255, 255), anchor="mm")
        y += 74
    d.line((W // 2 - 80, y + 14, W // 2 + 80, y + 14), fill=accent + (220,), width=3)
    y += 40
    for s in sub_lines:
        d.text((W // 2, y + 14), s, font=sf, fill=(230, 230, 240, 240), anchor="mm")
        y += 36
    return ov


def _cast_heading_overlay(title: str, accent) -> Image.Image:
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    d.text((W // 2, 44), "MAIN CAST", font=font(20, True), fill=accent + (255,), anchor="mm")
    head = (wrap(title, font(36, True), W - 200, 1) or [""])[0]
    d.text((W // 2 + 2, 84), head, font=font(36, True), fill=(0, 0, 0, 150), anchor="mm")
    d.text((W // 2, 82), head, font=font(36, True), fill=(255, 255, 255, 255), anchor="mm")
    return ov


def _name_tag(name: str, role: str, w: int, accent) -> Image.Image:
    img = Image.new("RGBA", (w, 74), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    nf, rf = font(20, True), font(14, True)
    d.rounded_rectangle((4, 2, w - 4, 72), radius=12, fill=(8, 8, 20, 205), outline=accent + (170,), width=2)
    d.text((w // 2, 25), (wrap(name, nf, w - 18, 1) or [""])[0], font=nf, fill=(255, 255, 255, 255), anchor="mm")
    d.text((w // 2, 52), (wrap(role.upper(), rf, w - 18, 1) or [""])[0], font=rf, fill=accent + (255,), anchor="mm")
    return img


def _blit(frame: Image.Image, img: Image.Image, x: float, y: float, alpha: float = 1.0):
    """alpha_composite that clips to the frame and supports fading."""
    x, y = int(x), int(y)
    left, top = max(0, -x), max(0, -y)
    right, bottom = min(img.width, W - x), min(img.height, H - y)
    if right <= left or bottom <= top or alpha <= 0:
        return
    part = img if (left, top, right, bottom) == (0, 0, img.width, img.height) else img.crop((left, top, right, bottom))
    if alpha < 1:
        part = part.copy()
        part.putalpha(part.getchannel("A").point(lambda v: int(v * alpha)))
    frame.alpha_composite(part, (x + left, y + top))


# --------------------------------------------------------------------------- particles

PARTICLE_KIND = {"calm": "petal", "hopeful": "petal", "melancholy": "rain", "mysterious": "firefly",
                 "tense": "ember", "action": "ember", "triumphant": "sparkle"}


def _particles(seed: int, n: int = 46):
    rng = random.Random(seed * 7 + 3)
    return [(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(30, 90), rng.uniform(0, math.tau),
             rng.uniform(3, 7)) for _ in range(n)]


def _draw_particles(img: Image.Image, kind: str, parts, t: float, accent):
    d = ImageDraw.Draw(img, "RGBA")
    for x0, y0, speed, phase, size in parts:
        if kind == "rain":
            y = (y0 + speed * 9 * t) % (H + 40) - 20
            x = (x0 - speed * 1.5 * t) % W
            d.line((x, y, x - 4, y + 18), fill=(200, 210, 240, 90), width=2)
            continue
        rising = kind in ("ember", "firefly", "sparkle")
        y = (y0 + (-1 if rising else 1) * speed * t) % (H + 40) - 20
        x = x0 + 34 * math.sin(t * 0.9 + phase)
        if kind == "petal":
            d.ellipse((x, y, x + size * 1.8, y + size), fill=(255, 185, 210, 200))
        elif kind == "ember":
            d.ellipse((x, y, x + size * 0.8, y + size * 0.8), fill=(255, 150, 70, 210))
        else:
            a = int(120 + 120 * (0.5 + 0.5 * math.sin(t * 3 + phase)))
            r = size * 0.6
            d.ellipse((x - r * 2.5, y - r * 2.5, x + r * 2.5, y + r * 2.5), fill=accent + (a // 6,))
            d.ellipse((x - r, y - r, x + r, y + r), fill=accent + (a,))


# --------------------------------------------------------------------------- audio


def _midi_hz(m: int) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def _sfx(kind: str, rng) -> np.ndarray:
    if kind == "swoosh":
        n = int(0.32 * SR)
        t = np.arange(n) / SR
        noise = rng.standard_normal(n)
        k = 12
        smooth = np.convolve(noise, np.ones(k) / k, mode="same")
        env = np.sin(np.pi * np.clip(t / 0.32, 0, 1)) ** 2
        return 0.9 * (noise - smooth) * env
    n = int(0.5 * SR)  # impact
    t = np.arange(n) / SR
    thump = np.sin(2 * np.pi * (90 - 50 * t) * t) * np.exp(-t * 9)
    crack = rng.standard_normal(n) * np.exp(-t * 22)
    return 1.1 * thump + 0.5 * crack


def synth_audio(segments: list[dict], path: Path):
    chunks = []
    sfx_track = []
    offset = 0
    for seg in segments:
        n = int(round(seg["frames"] / FPS * SR))
        t = np.arange(n, dtype=np.float64) / SR
        rng = np.random.default_rng(seg["seed"])
        notes = CHORDS[seg["mood"]]
        pad = np.zeros(n)
        for j, m in enumerate(notes):
            f = _midi_hz(m)
            lfo = 0.6 + 0.4 * np.sin(2 * np.pi * (0.07 + 0.03 * j) * t + j)
            amp = 0.22 if j == 0 else 0.12
            pad += amp * lfo * (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * f * 2.003 * t))
        beat = 0.5 if seg["mood"] in ("action", "tense") else 0.9
        for k in range(int(t[-1] / beat) if n else 0):
            if rng.random() < 0.65:
                start = int(k * beat * SR)
                m = notes[rng.integers(1, len(notes))] + 12
                length = min(n - start, int(1.6 * SR))
                tt = np.arange(length) / SR
                pad[start:start + length] += 0.10 * np.exp(-tt * 3.2) * np.sin(2 * np.pi * _midi_hz(int(m)) * tt)
        if seg["mood"] in ("action", "tense"):
            pulse = (np.sin(2 * np.pi * (2.0 if seg["mood"] == "action" else 1.0) * t) > 0.6).astype(float)
            pad += 0.18 * pulse * np.sin(2 * np.pi * _midi_hz(notes[0] - 12) * t)
        noise = np.convolve(rng.standard_normal(n), np.ones(80) / 80, mode="same")
        pad += 0.05 * noise
        fade = min(int(0.6 * SR), n // 2)
        env = np.ones(n)
        if fade:
            env[:fade] = np.linspace(0, 1, fade)
            env[-fade:] = np.linspace(1, 0, fade)
        chunks.append(pad * env)
        for when, kind in seg.get("sfx", []):
            sfx_track.append((offset + int(when * SR), kind))
        offset += n
    music = np.concatenate(chunks) if chunks else np.zeros(SR)
    music = music / (np.max(np.abs(music)) or 1) * 0.42
    rng = np.random.default_rng(7)
    for start, kind in sfx_track:
        s = _sfx(kind, rng) * 0.35
        end = min(len(music), start + len(s))
        if start < end:
            music[start:end] += s[: end - start]
    music = np.clip(music, -0.98, 0.98)
    stereo = np.stack([music, np.roll(music, int(0.012 * SR))], axis=1)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((stereo * 32767).astype("<i2").tobytes())


# --------------------------------------------------------------------------- timeline


def _scene_beats(scene: dict) -> list[dict]:
    if scene.get("beats"):
        return scene["beats"]
    return [{"type": "dialogue", **d} for d in scene.get("dialogue", [])]


def _build_segments(script: dict, cast: Cast) -> list[dict]:
    """Segments hold timed items: dicts with start, end, kind and kind-specific data."""
    look = _look(script)
    cfg = LEN_CFG.get(script.get("scene_length", "standard"), LEN_CFG["standard"])
    segs = []
    first_mood = script["scenes"][0]["mood"] if script["scenes"] else "hopeful"
    accent = PALETTES[first_mood][2]
    logline = wrap(script.get("logline", ""), font(26), W - 260, 3)
    segs.append({"mood": first_mood, "seed": 1, "setting": "", "look": look, "frames": int(4.5 * FPS),
                 "items": [dict(start=0.3, end=4.5, kind="card", overlay=_card_overlay(
                     script["title"], logline, f"{script.get('genre', '')}  ·  {script.get('style', '')}".upper(), accent))]})

    intro = [(c["name"], c.get("role", "")) for c in script.get("characters", [])[:6]]
    if intro:
        dur = 1.2 + 0.4 * len(intro) + 3.2
        segs.append({"mood": first_mood, "seed": 2, "setting": "", "look": look, "frames": int(dur * FPS), "intro": intro,
                     "items": [dict(start=0.2, end=dur, kind="card", overlay=_cast_heading_overlay(script["title"], accent))]})

    move_counter: dict = {}
    for scene in script["scenes"]:
        mood = scene["mood"]
        accent = PALETTES[mood][2]
        items, sfx, cursor = [], [], cfg["lead"]
        desc = scene["description"] if len(scene["description"]) <= 260 else scene["description"][:257] + "…"
        dur = _clamp(len(desc) / 18, *cfg["narr"])
        items.append(dict(start=cursor, end=cursor + dur, kind="narration", overlay=_line_overlay("SCENE", desc, accent, True)))
        cursor += dur + 0.2
        stage: list[str] = []

        def on_stage(name):
            if name and name not in stage and len(stage) < MAX_ON_STAGE:
                stage.append(name)

        for beat in _scene_beats(scene):
            kind = beat.get("type", "dialogue")
            if kind == "narration":
                dur = _clamp(len(beat["text"]) / 18, *cfg["narr"])
                items.append(dict(start=cursor, end=cursor + dur, kind="narration",
                                  overlay=_line_overlay("NARRATION", beat["text"], accent, True)))
            elif kind == "action":
                a, b = beat["character"], beat["target"]
                dur = cfg["action"]
                n_ex = max(1, int((dur - 0.5) // EXCHANGE))
                dur = 0.5 + n_ex * EXCHANGE + 0.3
                exchanges = []
                for k in range(n_ex):
                    x, y = (a, b) if k % 2 == 0 else (b, a)
                    d = cast.design(x)
                    idx = move_counter.get(x, 0)
                    move_counter[x] = idx + 1
                    kicker, move = technique_name(d, idx, beat.get("move", "") if k == 0 else "")
                    exchanges.append(dict(attacker=x, defender=y, element=d["element"],
                                          arc=("down", "up", "spin")[(idx + k) % 3],
                                          callout=make_callout((W, H), kicker, move, d["element"], 1 if k % 2 == 0 else -1)))
                    s = cursor + 0.5 + k * EXCHANGE
                    sfx += [(s + 0.70, "swoosh"), (s + 0.82, "impact")]
                items.append(dict(start=cursor, end=cursor + dur, kind="action", a=a, b=b, exchanges=exchanges,
                                  seed=len(items) * 31 + scene["number"]))
                on_stage(a)
                on_stage(b)
            else:
                dur = _clamp(len(beat["line"]) / 12.5, *cfg["line"])
                items.append(dict(start=cursor, end=cursor + dur, kind="dialogue", speaker=beat["character"],
                                  overlay=_line_overlay(beat["character"].upper(), beat["line"], accent, False)))
                on_stage(beat["character"])
            cursor += dur + 0.15
        total = cursor + cfg["tail"]
        segs.append({"mood": mood, "seed": scene["number"] * 101 + 7, "setting": scene["setting"], "look": look,
                     "frames": int(total * FPS), "header": _header_overlay(scene, accent), "items": items,
                     "stage": stage, "sfx": sfx})

    last_mood = script["scenes"][-1]["mood"] if script["scenes"] else "triumphant"
    segs.append({"mood": last_mood, "seed": 999, "setting": "", "look": look, "frames": int(4 * FPS),
                 "items": [dict(start=0.3, end=4.0, kind="card", overlay=_card_overlay(
                     "To be continued…", ["Storyboard preview generated locally by AI Anime Studio"],
                     script["title"].upper(), PALETTES[last_mood][2]))]})
    return segs


# --------------------------------------------------------------------------- choreography


def _action_state(item: dict, u: float) -> dict:
    """Positions/poses of the two fighters plus effects at time u into a fight shot."""
    a, b = item["a"], item["b"]
    pos = {a: 0.24 * W, b: 0.76 * W}
    pose = {a: "ready", b: "ready"}
    face = {a: 1, b: -1}
    st = dict(trail=None, flash=None, shake=0.0, callout=None, calpha=0.0,
              element=item["exchanges"][0]["element"], dir=1)
    for k, ex in enumerate(item["exchanges"]):
        s = 0.5 + k * EXCHANGE
        x, y = ex["attacker"], ex["defender"]
        xx, xy = pos[x], pos[y]
        dirn = 1 if xy >= xx else -1
        stop = xy - dirn * 0.2 * W
        knock = _clamp(xy + dirn * 0.1 * W, 0.1 * W, 0.9 * W)
        uu = u - s
        if uu < 0:
            break
        face[x], face[y] = dirn, -dirn
        st["element"], st["dir"] = ex["element"], dirn
        if uu >= EXCHANGE:
            pos[x], pos[y] = stop, knock
            pose[x] = pose[y] = "ready"
            continue
        st["callout"], st["calpha"] = ex["callout"], _clamp(min(uu / 0.15, (1.8 - uu) / 0.3), 0, 1)
        sc = ACTION_SCALE
        arc = {"down": (-130, 50), "up": (110, -60), "spin": (-90, 250)}[ex["arc"]]
        if dirn < 0:
            arc = (180 - arc[0], 180 - arc[1])
        geom = (stop + dirn * 150 * sc, ACTION_GROUND - 430 * sc, 215 * sc, arc[0], arc[1], ex["element"])
        if uu < 0.55:
            pos[x] = xx + (stop - xx) * _ease(uu / 0.55)
            pose[x] = "run1" if int(uu / 0.09) % 2 == 0 else "run2"
        elif uu < 0.72:
            pos[x], pose[x] = stop, "wind"
            pose[y] = "guard" if uu > 0.64 else "ready"
        elif uu < 1.05:
            pos[x], pose[x] = stop, "slash"
            pose[y] = "hurt" if uu > 0.8 else "guard"
            pos[y] = xy + (knock - xy) * _ease((uu - 0.8) / 0.3)
            st["trail"] = (*geom, (uu - 0.72) / 0.26, 1.0)
            st["flash"] = (pos[y], ACTION_GROUND - 420 * sc, max(0.0, 1 - abs(uu - 0.83) / 0.12), ex["element"])
            if uu > 0.8:
                st["shake"] = 16 * max(0.0, 1 - (uu - 0.8) / 0.5)
        elif uu < 1.6:
            pos[x], pose[x], pos[y] = stop, "slash_up", knock
            pose[y] = "hurt" if uu < 1.35 else "ready"
            st["trail"] = (*geom, 1.0, max(0.0, 1 - (uu - 1.05) / 0.45))
            st["shake"] = 16 * max(0.0, 1 - (uu - 0.8) / 0.5)
        else:
            pos[x], pos[y] = stop, knock
            pose[x] = "ready"
        break
    st["fighters"] = [(n, pose[n], pos[n], face[n]) for n in (a, b)]
    return st


def _stage_positions(names: list[str]) -> list[float]:
    return [W * x for x in STAGE_SLOTS[len(names)]]


# --------------------------------------------------------------------------- render


def render_episode(script: dict, out_path: Path, progress=None) -> float:
    """Render the script to an MP4 at out_path. Returns duration in seconds."""
    context = f"{script.get('genre', '')} {script.get('style', '')}"
    cast = Cast(script.get("characters", []), context)
    segments = _build_segments(script, cast)
    total_frames = sum(s["frames"] for s in segments)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        synth_audio(segments, wav)
        log = open(Path(tmp) / "ffmpeg.log", "wb")
        cmd = [ffmpeg, "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
               "-i", str(wav),
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(out_path)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log)
        done = 0
        try:
            for seg in segments:
                for frame in _render_segment(seg, cast):
                    proc.stdin.write(frame.convert("RGB").tobytes())
                    done += 1
                    if progress and done % 12 == 0:
                        progress(done / total_frames)
            proc.stdin.close()
            code = proc.wait()
        except (BrokenPipeError, OSError):
            # FFmpeg exited early; surface its own error message below.
            code = proc.wait()
        except BaseException:
            proc.kill()
            raise
        finally:
            log.close()
        if code != 0:
            raise RuntimeError("FFmpeg failed: " + (Path(tmp) / "ffmpeg.log").read_text(errors="replace")[-800:])
    if progress:
        progress(1.0)
    return total_frames / FPS


def _render_segment(seg: dict, cast: Cast):
    black = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    bg = make_background(seg["mood"], seg["seed"], seg["setting"], seg["look"])
    accent = PALETTES[seg["mood"]][2]
    kind = PARTICLE_KIND[seg["mood"]]
    parts = _particles(seg["seed"])
    rng = random.Random(seg["seed"])
    pan_x, pan_y = rng.choice([-1, 1]), rng.choice([-1, 1])
    n = seg["frames"]
    seg_len = n / FPS
    stage = seg.get("stage") or []
    slots = _stage_positions(stage) if stage else []
    tinted: dict = {}
    cam = {"z": 1.0, "fx": W / 2, "fy": H / 2}
    shake_rng = random.Random(seg["seed"] + 5)

    intro = []
    if seg.get("intro"):
        k = len(seg["intro"])
        sc = min(0.5, (W - 80) / (k * 330))
        col_w = W / k
        for i, (nm, role) in enumerate(seg["intro"]):
            spr, ax, ay = cast.sprite(nm, "idle", scale=sc, flip=i >= k / 2 and k > 1)
            intro.append((spr, ax, ay, col_w * (i + 0.5), _name_tag(nm, role, int(min(220, col_w - 12)), accent)))

    for f in range(n):
        t = f / FPS
        active, a = None, 0.0
        for item in seg["items"]:
            if item["start"] <= t < item["end"]:
                active, a = item, min(1.0, (t - item["start"]) / 0.3, (item["end"] - t) / 0.3)
                break
        fighting = active is not None and active["kind"] == "action"

        # slow pan across the oversized background (a plain crop: much cheaper than a resize)
        ease = _ease(f / max(1, n - 1))
        ox = int((BW - W) / 2 + pan_x * (BW - W) / 2 * 0.9 * (ease - 0.5) * 2)
        oy = int((BH - H) / 2 + pan_y * (BH - H) / 2 * 0.6 * (ease - 0.5) * 2)
        source = bg
        if fighting:
            el = active["exchanges"][0]["element"]
            if el not in tinted:
                tinted[el] = _tinted(bg, el)
            source = tinted[el]
        frame = source.crop((ox, oy, ox + W, oy + H)).convert("RGBA")
        _draw_particles(frame, kind, parts, t, accent)

        target_z, target_fx, shake = 1.0, W / 2, 0.0
        st = None
        if fighting:
            u = t - active["start"]
            st = _action_state(active, u)
            draw_speed_lines(frame, t, ELEMENT_COLORS[st["element"]][1], active["seed"], direction=st["dir"])
            for name, pose, x, face in st["fighters"]:
                blink = ((t + len(name)) % 3.6) < 0.12
                spr, ax, ay = cast.sprite(name, pose, not blink, pose in ("slash", "wind", "run1", "run2"),
                                          ACTION_SCALE, face < 0, False)
                _blit(frame, spr, x - ax, ACTION_GROUND - ay)
            if st["trail"]:
                tcx, tcy, tr, a0, a1, el, prog, alpha = st["trail"]
                draw_trail(frame, tcx, tcy, tr, a0, a1, el, prog, alpha, active["seed"])
            if st["flash"]:
                fx_, fy_, strength, el = st["flash"]
                draw_flash(frame, fx_, fy_, strength, el, active["seed"])
            target_z = 1.07
            target_fx = sum(x for _, _, x, _ in st["fighters"]) / 2
            shake = st["shake"]
        elif stage:
            speaker = active.get("speaker") if active else None
            for i, name in enumerate(stage):
                x0 = slots[i]
                facing = 1 if x0 <= W / 2 else -1
                enter = 0.25 + 0.4 * i
                walk = 1.6
                if t < enter:
                    continue
                speaking = speaker == name
                blink = ((t + i * 1.37) % 3.8) < 0.13
                if t < enter + walk:
                    start_x = -160 if x0 <= W / 2 else W + 160
                    q = (t - enter) / walk
                    x = start_x + (x0 - start_x) * (1 - (1 - q) ** 2)
                    pose = "walk1" if int(t / 0.2) % 2 == 0 else "walk2"
                    facing = 1 if x0 > start_x else -1
                    bob = -6 * abs(math.sin(t * math.pi / 0.2))
                    mouth = False
                else:
                    step = 34 * (1 if x0 < W / 2 else -1 if x0 > W / 2 else 0)
                    x = x0 + (step * _ease(a / 0.6) if speaking else 0)
                    if speaking:
                        pose = ("talk", "idle2", "talk2")[int((t - active["start"]) / 0.55) % 3]
                        mouth = a > 0.6 and int(t * 7) % 2 == 0
                    else:
                        pose = "idle" if int((t + i * 1.1) / 2.3) % 2 == 0 else "idle2"
                        mouth = False
                    if x0 == W / 2 and speaker in stage and speaker != name:
                        facing = 1 if slots[stage.index(speaker)] > x0 else -1
                    bob = 2.5 * math.sin(t * 2.0 + i * 1.3)
                dim = speaker is not None and not speaking
                spr, ax, ay = cast.sprite(name, pose, not blink, mouth, STAGE_SCALE, facing < 0, dim)
                _blit(frame, spr, x - ax, STAGE_GROUND - ay + bob)
                if speaking:
                    target_z, target_fx = 1.06, x
        elif intro:
            for i, (spr, ax, ay, xc, tag) in enumerate(intro):
                q = _clamp((t - (0.6 + 0.4 * i)) / 0.5, 0, 1)
                if q <= 0:
                    continue
                e = 1 - (1 - q) ** 3
                ground = H - 110 + 40 * (1 - e)
                _blit(frame, spr, xc - ax, ground - ay, q)
                _blit(frame, tag, xc - tag.width / 2, H - 96 + 40 * (1 - e), q)

        # camera: smooth zoom toward the action, plus impact shake
        cam["z"] += (target_z - cam["z"]) * 0.08
        cam["fx"] += (target_fx - cam["fx"]) * 0.08
        if cam["z"] > 1.002 or shake > 0.5:
            z = max(cam["z"], 1 + 2.5 * shake / H)  # leave room inside the frame for the shake offset
            vw, vh = W / z, H / z
            fx_c = _clamp(cam["fx"] + shake_rng.uniform(-shake, shake), vw / 2, W - vw / 2)
            fy_c = _clamp(H * 0.55 + shake_rng.uniform(-shake, shake), vh / 2, H - vh / 2)
            box = (fx_c - vw / 2, fy_c - vh / 2, fx_c + vw / 2, fy_c + vh / 2)
            frame = frame.resize((W, H), Image.BILINEAR, box=box)

        if seg.get("header") is not None and not fighting:
            ha = _clamp(min(t / 0.4, (5.5 - t) / 0.5), 0, 1)
            if ha >= 1:
                frame.alpha_composite(seg["header"])
            elif ha > 0:
                frame = Image.blend(frame, Image.alpha_composite(frame, seg["header"]), ha)

        if fighting:
            if st["callout"] is not None and st["calpha"] > 0:
                _blit(frame, st["callout"], 0, 0, st["calpha"])
            cut = t - active["start"]
            if cut < 0.15:  # white cut-in flash when the fight starts
                frame.alpha_composite(Image.new("RGBA", (W, H), (255, 255, 255, int(170 * (1 - cut / 0.15)))))
        elif active is not None:
            with_text = Image.alpha_composite(frame, active["overlay"])
            frame = with_text if a >= 1 else Image.blend(frame, with_text, a)

        fade = min(1.0, t / 0.4, (seg_len - t) / 0.4)
        if fade < 1:
            frame = Image.blend(black, frame, max(0.0, fade))
        yield frame

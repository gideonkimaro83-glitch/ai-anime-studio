"""Action effects: elemental technique trails, impact flashes, speed lines, technique callouts."""

from __future__ import annotations

import math
import random

from PIL import Image, ImageDraw, ImageFilter

from characters import ELEMENT_COLORS
from textutil import font, wrap


def _arc_points(cx, cy, r, a0, a1, n):
    return [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * k / n)),
             cy + r * math.sin(math.radians(a0 + (a1 - a0) * k / n))) for k in range(n + 1)]


def _ribbon(g, cx, cy, r, a0, a1, width, color, n=40):
    """Crescent-shaped stroke: thin at the tail, thick near the leading edge."""
    if abs(a1 - a0) < 1:
        return
    outer, inner = [], []
    for k in range(n + 1):
        u = k / n
        th = width * (0.15 + 0.85 * math.sin(math.pi * min(1.0, u * 0.92 + 0.04)) ** 0.8) * (0.4 + 0.6 * u)
        a = math.radians(a0 + (a1 - a0) * u)
        outer.append((cx + (r + th / 2) * math.cos(a), cy + (r + th / 2) * math.sin(a)))
        inner.append((cx + (r - th / 2) * math.cos(a), cy + (r - th / 2) * math.sin(a)))
    g.polygon(outer + inner[::-1], fill=color)


def draw_trail(frame: Image.Image, cx, cy, r, a0, a1, element: str, progress: float, alpha: float, seed: int):
    """Draw an elemental slash arc that has swept `progress` (0..1) of its path, faded by `alpha`."""
    if progress <= 0 or alpha <= 0:
        return
    main, light, core = ELEMENT_COLORS[element]
    a_end = a0 + (a1 - a0) * min(1.0, progress)
    a_start = a0 + (a1 - a0) * max(0.0, progress - 0.85)
    layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    g = ImageDraw.Draw(layer, "RGBA")
    rng = random.Random(seed)

    if element == "thunder":
        for b in range(3):
            pts = [(x + rng.uniform(-18, 18), y + rng.uniform(-18, 18))
                   for x, y in _arc_points(cx, cy, r + b * 14 - 14, a_start, a_end, 14)]
            g.line(pts, fill=main + (230,), width=12 - b * 3, joint="curve")
            g.line(pts, fill=core + (255,), width=4)
            for x, y in pts[2::4]:
                g.line((x, y, x + rng.uniform(-60, 60), y + rng.uniform(-60, 60)), fill=light + (200,), width=3)
    elif element == "wind":
        for k, off in enumerate((-30, 0, 30)):
            _ribbon(g, cx, cy, r + off, a_start + k * 6, a_end, 26, light + (150,))
            _ribbon(g, cx, cy, r + off, a_start + k * 6, a_end, 9, core + (200,))
    else:
        _ribbon(g, cx, cy, r, a_start, a_end, 96, main + (215,))
        _ribbon(g, cx, cy, r + 6, a_start, a_end, 54, light + (230,))
        _ribbon(g, cx, cy, r + 10, a_start, a_end, 16, core + (255,))
        pts = _arc_points(cx, cy, r, a_start, a_end, 10)
        if element == "water":  # ukiyo-e curls and foam
            _ribbon(g, cx, cy, r - 46, a_start + 10, a_end - 4, 30, main + (170,))
            for x, y in pts[1::2]:
                cr = rng.uniform(14, 26)
                g.arc((x - cr, y - cr, x + cr, y + cr), 0, 270, fill=core + (240,), width=5)
                g.arc((x - cr / 2, y - cr / 2, x + cr / 2, y + cr / 2), 90, 360, fill=light + (240,), width=4)
                for _ in range(3):
                    fx, fy = x + rng.uniform(-40, 40), y + rng.uniform(-40, 40)
                    fr = rng.uniform(3, 7)
                    g.ellipse((fx - fr, fy - fr, fx + fr, fy + fr), fill=core + (230,))
        elif element == "flame":
            for x, y in pts:
                ang = math.atan2(y - cy, x - cx)
                L = rng.uniform(40, 90)
                tip = (x + math.cos(ang) * L + rng.uniform(-15, 15), y + math.sin(ang) * L + rng.uniform(-15, 15))
                side = (math.cos(ang + 1.57) * 16, math.sin(ang + 1.57) * 16)
                g.polygon([(x + side[0], y + side[1]), tip, (x - side[0], y - side[1])], fill=main + (210,))
                g.polygon([(x + side[0] / 2, y + side[1] / 2), ((x + tip[0]) / 2, (y + tip[1]) / 2),
                           (x - side[0] / 2, y - side[1] / 2)], fill=light + (230,))
        elif element == "moon":
            cutter = ImageDraw.Draw(layer)  # non-blending draw, so a transparent fill erases
            for x, y in pts[::2]:
                cr = rng.uniform(12, 22)
                out = math.atan2(y - cy, x - cx)
                ox, oy = x + math.cos(out) * 55, y + math.sin(out) * 55
                g.ellipse((ox - cr, oy - cr, ox + cr, oy + cr), fill=light + (230,))
                cutter.ellipse((ox - cr + 7, oy - cr - 4, ox + cr + 7, oy + cr - 4), fill=(0, 0, 0, 0))
        elif element in ("light", "ice"):
            for x, y in pts:
                s = rng.uniform(8, 18)
                g.polygon([(x, y - s), (x + s / 4, y - s / 4), (x + s, y), (x + s / 4, y + s / 4), (x, y + s),
                           (x - s / 4, y + s / 4), (x - s, y), (x - s / 4, y - s / 4)], fill=core + (240,))
        elif element == "shadow":
            for x, y in pts[::2]:
                g.line((x, y, x + rng.uniform(-10, 10), y + rng.uniform(30, 70)), fill=light + (200,), width=5)
        elif element == "earth":
            for x, y in pts[::2]:
                s = rng.uniform(10, 22)
                g.polygon([(x - s, y), (x - s / 3, y - s), (x + s, y - s / 2), (x + s / 2, y + s)], fill=light + (240,))

    glow = layer.resize((frame.width // 4, frame.height // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(6))
    glow = glow.resize(frame.size, Image.BILINEAR)
    for lay in (glow, layer):
        if alpha < 1:
            lay.putalpha(lay.getchannel("A").point(lambda v: int(v * alpha)))
        frame.alpha_composite(lay)


def draw_flash(frame: Image.Image, x, y, strength: float, element: str, seed: int):
    if strength <= 0:
        return
    _, light, core = ELEMENT_COLORS[element]
    layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    g = ImageDraw.Draw(layer, "RGBA")
    rng = random.Random(seed)
    pts = []
    for k in range(28):
        a = math.tau * k / 28
        rad = (rng.uniform(140, 320) if k % 2 == 0 else rng.uniform(30, 60)) * (0.6 + 0.4 * strength)
        pts.append((x + math.cos(a) * rad, y + math.sin(a) * rad))
    g.polygon(pts, fill=light + (int(200 * strength),))
    rr = 70 * strength
    g.ellipse((x - rr, y - rr, x + rr, y + rr), fill=core + (int(255 * strength),))
    frame.alpha_composite(layer)
    if strength > 0.5:
        white = Image.new("RGBA", frame.size, (255, 255, 255, int(110 * (strength - 0.5) * 2)))
        frame.alpha_composite(white)


def draw_speed_lines(frame: Image.Image, t: float, color, seed: int, direction: int = 1, count: int = 38):
    g = ImageDraw.Draw(frame, "RGBA")
    rng = random.Random(seed)
    w, h = frame.size
    for _ in range(count):
        y = rng.uniform(0, h)
        length = rng.uniform(120, 420)
        speed = rng.uniform(900, 1800)
        x = (rng.uniform(0, w + length) - direction * t * speed) % (w + length) - length
        g.line((x, y, x + length, y), fill=color + (rng.randint(40, 110),), width=rng.choice([2, 3, 4]))


def make_callout(size, kicker: str, name: str, element: str, side: int = 1) -> Image.Image:
    """Big technique-name card; side=1 places it upper-left, -1 upper-right."""
    w, h = size
    main, light, core = ELEMENT_COLORS[element]
    card = Image.new("RGBA", (900, 200), (0, 0, 0, 0))
    g = ImageDraw.Draw(card, "RGBA")
    g.polygon([(30, 40), (880, 22), (860, 176), (10, 190)], fill=(8, 8, 18, 175))
    g.polygon([(30, 40), (880, 22), (878, 34), (28, 52)], fill=main + (255,))
    kf, nf = font(22, True), font(64, True)
    g.text((54, 62), kicker, font=kf, fill=light + (255,))
    title = (wrap(name, nf, 800, 1) or [""])[0]
    g.text((50, 92), title, font=nf, fill=core + (255,), stroke_width=4, stroke_fill=main + (255,))
    card = card.rotate(4 * side, resample=Image.BICUBIC, expand=True)
    ov = Image.new("RGBA", size, (0, 0, 0, 0))
    x = 30 if side > 0 else w - card.width - 30
    ov.alpha_composite(card, (max(0, x), 60))
    return ov

"""The picture layer: one procedural scene per act, drawn as ink densities.

Every scene is `draw_scene_X(ink, red, t)`: `ink` and `red` are cairo
contexts on A8 (alpha only) surfaces in scene units, 0..100 across the whole
frame; alpha is ink density. `t` is the film time in seconds. The compositor
softens, fades and colours the densities (compose.py), so scenes only think
about shapes and motion.

The code occupies the whole frame, and the picture can only be at full
strength away from the finder patterns and timing lines, so the compositions
are centred on the free canvas (roughly x, y in 30..85).
"""

from __future__ import annotations

import math

import cairo

import compose
import config

TAU = 2 * math.pi

# Canvas anchor (scene units) - centre of the area where the picture is at
# full strength, see Compositor.canvas_box().
CX, CY = 57.5, 57.0

# Act boundaries in seconds.
T_TITLE = (0.0, config.TITLE_END_SEC)
T_FLOWER = (config.TITLE_END_SEC, 14.0)
T_BIRD = (14.0, 24.0)
T_SEA = (24.0, 34.0)
T_CITY = (34.0, 44.0)
T_EYE = (44.0, 52.0)
T_RETURN = (52.0, config.LINK_START_SEC)
T_LINK = (config.LINK_START_SEC, config.DURATION_SEC)

ACTS = [
    ("title", T_TITLE), ("flower", T_FLOWER), ("bird", T_BIRD), ("sea", T_SEA),
    ("city", T_CITY), ("eye", T_EYE), ("return", T_RETURN), ("link", T_LINK),
]


def act_at(t: float) -> str:
    for name, (a, b) in ACTS:
        if a <= t < b:
            return name
    return ACTS[-1][0]


# ------------------------------------------------------------------ easing

def clamp01(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def ramp(t, t0, t1):
    return clamp01((t - t0) / (t1 - t0))


def smooth(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def ease_in_out(x):
    return 0.5 - 0.5 * math.cos(math.pi * clamp01(x))


def ease_out(x):
    x = clamp01(x)
    return 1 - (1 - x) ** 3


def ease_in(x):
    x = clamp01(x)
    return x ** 3


def ease_out_back(x, s=1.6):
    x = clamp01(x)
    return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2


def lerp(a, b, u):
    return a + (b - a) * u


def lerp2(p, q, u):
    return (lerp(p[0], q[0], u), lerp(p[1], q[1], u))


# ------------------------------------------------------------------ shapes

def fill(ctx, density):
    ctx.set_source_rgba(0, 0, 0, clamp01(density))
    ctx.fill()


def ellipse(ctx, cx, cy, rx, ry, angle=0.0):
    ctx.save()
    ctx.translate(cx, cy)
    ctx.rotate(angle)
    ctx.scale(max(rx, 1e-3), max(ry, 1e-3))
    ctx.arc(0, 0, 1, 0, TAU)
    ctx.restore()


def petal_path(ctx, x, y, length, width, angle):
    """Almond from the base (x, y) outward along `angle`."""
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(angle)
    w = width / 2
    ctx.move_to(0, 0)
    ctx.curve_to(length * 0.25, -w * 1.25, length * 0.8, -w * 0.9, length, 0)
    ctx.curve_to(length * 0.8, w * 0.9, length * 0.25, w * 1.25, 0, 0)
    ctx.close_path()
    ctx.restore()


def bezier_point(p0, p1, p2, p3, u):
    a = (1 - u) ** 3
    b = 3 * (1 - u) ** 2 * u
    c = 3 * (1 - u) * u * u
    d = u ** 3
    return (a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
            a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1])


def bezier_split(p0, p1, p2, p3, u):
    """First part [0, u] of a cubic bezier (de Casteljau)."""
    q0 = lerp2(p0, p1, u)
    q1 = lerp2(p1, p2, u)
    q2 = lerp2(p2, p3, u)
    r0 = lerp2(q0, q1, u)
    r1 = lerp2(q1, q2, u)
    s = lerp2(r0, r1, u)
    return p0, q0, r0, s


def catmull(points, u):
    """Point and tangent angle on a Catmull-Rom path through `points`, u in 0..1."""
    n = len(points) - 1
    x = clamp01(u) * n
    i = min(int(x), n - 1)
    f = x - i
    p0 = points[max(i - 1, 0)]
    p1 = points[i]
    p2 = points[i + 1]
    p3 = points[min(i + 2, n)]

    def comp(k):
        a = p1[k]
        b = 0.5 * (p2[k] - p0[k])
        c = p0[k] - 2.5 * p1[k] + 2 * p2[k] - 0.5 * p3[k]
        d = -0.5 * p0[k] + 1.5 * p1[k] - 1.5 * p2[k] + 0.5 * p3[k]
        pos = a + b * f + c * f * f + d * f ** 3
        der = b + 2 * c * f + 3 * d * f * f
        return pos, der

    (x0, dx), (y0, dy) = comp(0), comp(1)
    return (x0, y0), math.atan2(dy, dx)


def erase(ctx, density):
    ctx.save()
    ctx.set_operator(cairo.OPERATOR_DEST_OUT)
    ctx.set_source_rgba(0, 0, 0, clamp01(density))
    ctx.fill()
    ctx.restore()


# ================================================================== FLOWER

FL_BASE = (56.0, 83.5)
FL_HEAD = (58.0, 47.5)
FL_C1 = (51.0, 71.0)
FL_C2 = (63.5, 58.0)
N_PETALS = 7
FALLING_PETAL = 1          # this petal leaves the flower in the bird act


def _petal_angle(i):
    return -math.pi / 2 + (i + 0.5) * TAU / N_PETALS


def draw_flower(ink, red, t, opacity=1.0, detached=False):
    """Seed -> sprout -> stem -> leaves -> bud -> bloom. `t` = seconds into the act."""
    if opacity <= 0:
        return
    # ground and seed
    soil = smooth(ramp(t, 0.0, 0.8))
    ellipse(ink, FL_BASE[0], FL_BASE[1] + 3.0, 17.0 * soil, 3.4 * soil)
    fill(ink, 0.40 * opacity)
    seed_in = smooth(ramp(t, 0.1, 0.7))
    wobble = 0.12 * math.sin(t * 7.0) * (1 - ramp(t, 1.0, 1.6))
    seed_scale = seed_in * (1.0 - 0.35 * ramp(t, 1.2, 2.2))
    ellipse(ink, FL_BASE[0], FL_BASE[1] - 0.5, 4.6 * seed_scale, 3.3 * seed_scale, -0.35 + wobble)
    fill(ink, 0.95 * opacity)

    # gentle sway once in bloom
    sway = 0.022 * math.sin((t - 8.0) * 1.7) * ramp(t, 8.0, 9.0)
    ink.save()
    red.save()
    for ctx in (ink, red):
        ctx.translate(*FL_BASE)
        ctx.rotate(sway)
        ctx.translate(-FL_BASE[0], -FL_BASE[1])

    grow = ease_in_out(ramp(t, 1.2, 4.3))
    if grow > 0:
        p0, p1, p2, p3 = bezier_split(FL_BASE, FL_C1, FL_C2, FL_HEAD, grow)
        ink.move_to(*p0)
        ink.curve_to(*p1, *p2, *p3)
        ink.set_line_width(4.2)
        ink.set_line_cap(cairo.LINE_CAP_ROUND)
        ink.set_source_rgba(0, 0, 0, 0.9 * opacity)
        ink.stroke()

    # leaves unfold from the stem
    for k, (u, side, t0) in enumerate(((0.33, -1, 3.0), (0.55, 1, 3.6))):
        s = ease_out_back(ramp(t, t0, t0 + 1.4), 1.2)
        if s <= 0 or grow < u:
            continue
        x, y = bezier_point(FL_BASE, FL_C1, FL_C2, FL_HEAD, u)
        up = -math.pi / 2
        ang = up + side * lerp(0.15, 1.05, ease_out(ramp(t, t0, t0 + 1.6)))
        petal_path(ink, x, y, 17.0 * s, 7.6 * s, ang)
        fill(ink, 0.85 * opacity)

    # bud, then petals opening one by one
    bud = ease_out(ramp(t, 4.0, 5.1))
    bloom0 = 5.1
    if bud > 0:
        hx, hy = FL_HEAD
        for i in range(N_PETALS):
            if detached and i == FALLING_PETAL:
                continue
            t0 = bloom0 + 0.42 * i
            o = ease_out_back(ramp(t, t0, t0 + 1.1), 1.3)
            closed = -math.pi / 2 + (i - 3) * 0.08
            ang = lerp(closed, _petal_angle(i), o)
            length = lerp(8.5 * bud, 16.5, o)
            width = lerp(4.6 * bud, 9.5, o)
            petal_path(red, hx, hy, length, width, ang)
            fill(red, 0.95 * opacity)
        centre = ease_out(ramp(t, bloom0 + 0.3, bloom0 + 1.2))
        if centre > 0:
            ink.arc(hx, hy, 4.8 * centre, 0, TAU)
            fill(ink, 0.95 * opacity)
    ink.restore()
    red.restore()


def flower_petal_tip_state(t_flower):
    """Where the falling petal sits on the (swaying) flower."""
    ang = _petal_angle(FALLING_PETAL)
    return FL_HEAD, ang


def draw_scene_flower(ink, red, t):
    draw_flower(ink, red, t - T_FLOWER[0])


# ==================================================================== BIRD

BIRD_PATH = [
    (52.0, 63.0), (57.0, 56.0), (64.0, 51.0), (70.0, 46.5), (67.5, 41.0),
    (58.0, 40.5), (48.5, 43.5), (44.5, 51.0), (48.0, 58.5), (55.0, 62.0),
    (60.0, 63.5),
]
GROUND_Y = 82.5
BIRD_SCALE = 1.75


def draw_bird(ink, red, x, y, heading, flap, scale=1.0, morph=1.0, density=0.95, breast=0.9):
    """Side-view bird. `flap` in -1..1 (wing tips up..down). `morph` 0..1
    grows wings/head/tail out of a petal-shaped body."""
    ink.save()
    red.save()
    for ctx in (ink, red):
        ctx.translate(x, y)
        ctx.rotate(heading)
        if math.cos(heading) < 0:          # keep the back up when flying left
            ctx.scale(1, -1)
        ctx.scale(scale, scale)
    m = smooth(morph)
    # body
    ellipse(ink, 0, 0, lerp(5.2, 6.6, m), lerp(2.6, 2.5, m))
    fill(ink, density * m)
    # breast keeps the petal's colour
    ellipse(red, lerp(0, 1.8, m), lerp(0, 1.1, m), lerp(5.4, 3.2, m), lerp(2.7, 1.5, m))
    fill(red, breast * lerp(1.0, 0.85, m))
    if m > 0:
        # head and beak
        ink.arc(5.6, -1.0, 2.35 * m, 0, TAU)
        fill(ink, density)
        ink.move_to(7.3, -1.6)
        ink.line_to(7.3 + 3.2 * m, -0.6)
        ink.line_to(7.3, 0.2)
        ink.close_path()
        fill(ink, density)
        # tail
        ink.move_to(-5.0, -0.8)
        ink.line_to(-5.0 - 5.8 * m, -2.9 * m)
        ink.line_to(-5.0 - 5.2 * m, 1.9 * m)
        ink.close_path()
        fill(ink, density)
        # wings: far wing lighter, near wing darker
        for off, dens, phase in ((-1.2, 0.6, 0.35), (0.6, density, 0.0)):
            f = math.sin(math.asin(max(-1.0, min(1.0, flap))) + phase)
            tip_y = lerp(-15.0, 9.0, (f + 1) / 2) * m
            tip_x = lerp(-5.5, -2.0, (f + 1) / 2)
            ink.move_to(3.4, -0.4 + off * 0.3)
            ink.curve_to(2.6, tip_y * 0.5 - 1.0, tip_x + 4.5, tip_y * 0.95, tip_x, tip_y)
            ink.curve_to(tip_x - 3.8, tip_y * 0.6, -5.0, tip_y * 0.12, -4.2, 0.3 + off * 0.3)
            ink.close_path()
            fill(ink, dens * m)
    ink.restore()
    red.restore()


def bird_state(tb):
    """Bird position/heading/flap at `tb` seconds into the bird act."""
    u = ease_in_out(ramp(tb, 2.1, 10.0))
    (x, y), heading = catmull(BIRD_PATH, u)
    rate = 1.25
    amp = 1.0 - 0.55 * smooth(ramp(math.sin(tb * 0.9), 0.4, 1.0))    # glide now and then
    flap = amp * math.sin(TAU * rate * tb)
    return x, y, heading, flap


def draw_scene_bird(ink, red, t):
    tb = t - T_BIRD[0]
    # the flower lingers, one petal comes loose
    fade = 1.0 - smooth(ramp(tb, 0.8, 2.6))
    draw_flower(ink, red, 10.0 + tb, opacity=fade, detached=True)
    # the falling petal
    (hx, hy), ang = flower_petal_tip_state(10.0 + tb)
    start = (hx + 8.0 * math.cos(ang), hy + 8.0 * math.sin(ang))
    fall = ease_in_out(ramp(tb, 0.2, 2.1))
    px = lerp(start[0], BIRD_PATH[0][0], fall) + 4.0 * math.sin(tb * 2.6) * (1 - fall)
    py = lerp(start[1], BIRD_PATH[0][1], fall)
    morph = smooth(ramp(tb, 1.4, 2.6))
    x, y, heading, flap = bird_state(tb)
    if morph < 1:
        # petal: red almond tumbling, turning into the bird
        spin = ang + 1.4 * math.sin(tb * 2.1) * (1 - morph)
        cx = lerp(px, x, morph)
        cy = lerp(py, y, morph)
        rot = lerp(spin, heading, morph)
        petal_path(red, cx - 8.0 * math.cos(rot), cy - 8.0 * math.sin(rot), 16.5, 9.5, rot)
        fill(red, 0.95 * (1 - morph))
    if morph > 0:
        draw_bird(ink, red, x, y, heading, flap * morph, scale=lerp(0.9, BIRD_SCALE, morph), morph=morph)
        # soft shadow on the ground, larger and fainter when the bird is high
        alt = clamp01((GROUND_Y - y) / 45.0)
        ellipse(ink, x - 2.0, GROUND_Y, 14.0 * (1.1 - 0.4 * alt), 2.4 * (1.1 - 0.4 * alt))
        fill(ink, (0.50 - 0.28 * alt) * morph * (1 - ramp(tb, 9.2, 10.0)))


# ===================================================================== SEA

def sea_level(ts):
    return lerp(98.0, 57.0, ease_out(ramp(ts, 0.0, 1.6)))


def wave_y(layer, x, ts, calm=0.0):
    base_off, amp, wl, speed, _ = layer
    a = amp * (1 - calm)
    y = base_off + a * math.sin(TAU * (x - speed * ts) / wl) \
        + 0.35 * a * math.sin(TAU * (1.7 * x + 0.6 * speed * ts) / wl + 1.3)
    return y


# (offset below sea level, amplitude, wavelength, speed, density)
WAVES = [
    (-5.0, 2.0, 26.0, 3.0, 0.30),
    (4.0, 3.8, 34.0, -4.0, 0.58),
    (15.0, 5.2, 42.0, 5.5, 0.90),
]


def roller_bump(x, ts):
    """A big rolling crest that sweeps left to right across the frame."""
    c = lerp(18.0, 100.0, ease_in_out(ramp(ts, 3.2, 8.6)))
    h = 17.0 * math.sin(math.pi * ramp(ts, 3.2, 8.6))
    d = (x - c) / 12.0
    return -h * math.exp(-d * d) * (1.0 if d < 0 else 1.3), c, h


def draw_waves(ink, ts, calm=0.0, rise=None, roll=True):
    lvl = sea_level(ts) if rise is None else rise
    for k, layer in enumerate(WAVES):
        dens = layer[4]
        top = lvl + layer[0] - layer[1]
        grad = cairo.LinearGradient(0, top, 0, top + 13.0)
        grad.add_color_stop_rgba(0.0, 0, 0, 0, dens)
        grad.add_color_stop_rgba(1.0, 0, 0, 0, dens * (0.45 if k < 2 else 0.7))
        ink.move_to(10, 110)
        x = 10.0
        while x <= 96.0:
            y = lvl + wave_y(layer, x, ts, calm)
            if roll and k == 2:
                y += roller_bump(x, ts)[0] * (1 - calm)
            ink.line_to(x, y)
            x += 1.0
        ink.line_to(96, 110)
        ink.close_path()
        ink.set_source(grad)
        ink.fill()
    if roll and calm < 1:
        # the curl on the rolling crest
        _, c, h = roller_bump(0.0, ts)
        if h > 1.0:
            top = lvl + WAVES[2][0] - h
            ink.save()
            ink.translate(c + 1.5, top + 1.2)
            ink.scale(h / 7.5, h / 7.5)
            ink.move_to(-8.0, 3.5)
            ink.curve_to(-4.0, -4.5, 5.5, -5.0, 7.0, 0.5)
            ink.curve_to(7.5, 3.5, 3.5, 4.2, 2.5, 1.5)
            ink.curve_to(4.2, 2.0, 4.8, -0.8, 1.5, -1.8)
            ink.curve_to(-2.5, -2.6, -4.5, 1.5, -5.0, 5.5)
            ink.close_path()
            ink.restore()
            fill(ink, 0.9 * (1 - calm))


def draw_boat(ink, red, x, y, tilt, opacity=1.0, scale=1.0):
    ink.save()
    red.save()
    for ctx in (ink, red):
        ctx.translate(x, y)
        ctx.rotate(tilt)
        ctx.scale(scale, scale)
    ink.move_to(-6.0, -1.2)
    ink.line_to(6.5, -1.2)
    ink.line_to(4.0, 2.4)
    ink.line_to(-4.2, 2.4)
    ink.close_path()
    fill(ink, 0.95 * opacity)
    ink.rectangle(-0.6, -12.0, 1.3, 11.0)
    fill(ink, 0.9 * opacity)
    red.move_to(0.9, -11.5)
    red.line_to(7.2, -2.4)
    red.line_to(0.9, -2.4)
    red.close_path()
    fill(red, 0.95 * opacity)
    ink.restore()
    red.restore()


def boat_state(ts, calm=0.0):
    x = lerp(38.0, 66.0, ease_in_out(ramp(ts, 1.5, 12.0)))
    lvl = sea_level(ts)
    layer = WAVES[1]
    y = lvl + wave_y(layer, x, ts, calm)
    slope = (wave_y(layer, x + 1.0, ts, calm) - wave_y(layer, x - 1.0, ts, calm)) / 2.0
    return x, y - 0.8, math.atan(slope) * 0.9


def draw_scene_sea(ink, red, t):
    ts = t - T_SEA[0]
    # the bird dives in
    if ts < 1.6:
        tb = T_BIRD[1] - T_BIRD[0] + ts
        x0, y0, h0, _ = bird_state(T_BIRD[1] - T_BIRD[0])
        dive = ease_in(ramp(ts, 0.0, 1.15))
        x = x0 + 5.0 * dive
        y = lerp(y0, sea_level(1.1) + 2.0, dive)
        heading = lerp(h0, 1.25, ease_out(ramp(ts, 0.0, 0.6)))
        vis = 1 - ramp(ts, 1.0, 1.25)
        if vis > 0:
            draw_bird(ink, red, x, y, heading, -0.9, scale=lerp(BIRD_SCALE, 1.4, dive), density=0.95 * vis, breast=0.9 * vis)
    # splash
    sp = ramp(ts, 1.05, 2.1)
    if 0 < sp < 1:
        lvl = sea_level(ts) + WAVES[0][0]
        h = 12.0 * math.sin(math.pi * sp)
        for k, dx in enumerate((-4.0, 0.0, 4.5)):
            ellipse(ink, 64.0 + dx * (0.9 + sp), lvl - h * (0.9 + 0.35 * (k == 1)), 2.8, 4.0)
            fill(ink, 0.8 * (1 - sp))
    draw_waves(ink, ts)
    b = ramp(ts, 1.8, 3.0)
    if b > 0:
        bx, by, tilt = boat_state(ts)
        draw_boat(ink, red, bx, by, tilt, opacity=b, scale=1.7)
        # the boat sits in the mid layer: redraw the near wave in front of it
        lvl = sea_level(ts)
        ink.move_to(10, 110)
        x = 10.0
        while x <= 96.0:
            ink.line_to(x, lvl + wave_y(WAVES[2], x, ts) + roller_bump(x, ts)[0])
            x += 1.0
        ink.line_to(96, 110)
        ink.close_path()
        fill(ink, 0.35)


# ==================================================================== CITY

# (left, width, height) - heights above the water line
BUILDINGS = [
    (29.0, 11.0, 22.0), (40.0, 11.5, 33.0), (51.5, 12.0, 40.0),
    (63.5, 10.5, 29.0), (74.0, 11.5, 35.0),
]
# (building index, column 0..1, height fraction, light-up time)
WINDOWS = [
    (2, 0.30, 0.78, 2.6), (1, 0.55, 0.62, 3.3), (4, 0.40, 0.55, 4.0),
    (2, 0.68, 0.46, 4.7), (3, 0.50, 0.62, 5.4), (0, 0.50, 0.50, 6.1),
    (4, 0.62, 0.28, 6.8), (1, 0.30, 0.30, 7.5),
]
WATER_Y = 77.0
MOON_END = (73.0, 40.0)
MOON_R = 6.5


def city_rise(tc, i):
    return ease_out(ramp(tc, 0.5 + 0.13 * i, 2.4 + 0.13 * i))


def moon_state(tc):
    u = ease_in_out(ramp(tc, 2.8, 9.6))
    return (lerp(76.0, MOON_END[0], u), lerp(72.0, MOON_END[1], u)), MOON_R


def draw_night_sky(ink, amount, moon, moon_r, glow=1.0):
    if amount <= 0:
        return
    grad = cairo.LinearGradient(0, 28, 0, WATER_Y)
    grad.add_color_stop_rgba(0.0, 0, 0, 0, 0.50 * amount)
    grad.add_color_stop_rgba(1.0, 0, 0, 0, 0.16 * amount)
    ink.rectangle(0, 0, 100, WATER_Y + 1)
    ink.set_source(grad)
    ink.fill()
    if moon is not None and moon_r > 0:
        mx, my = moon
        # wash the sky darker around the moon and leave the moon as bare
        # paper - the ink painter's way of drawing a moon
        ring = cairo.RadialGradient(mx, my, moon_r * 0.8, mx, my, moon_r * 2.5)
        ring.add_color_stop_rgba(0, 0, 0, 0, 0.42 * glow * amount)
        ring.add_color_stop_rgba(1, 0, 0, 0, 0.0)
        ink.arc(mx, my, moon_r * 2.5, 0, TAU)
        ink.set_source(ring)
        ink.fill()
        ink.save()
        ink.set_operator(cairo.OPERATOR_DEST_OUT)
        ink.arc(mx, my, moon_r, 0, TAU)
        ink.set_source_rgba(0, 0, 0, 1)
        ink.fill()
        ink.restore()


def draw_city(ink, tc, sink=0.0, windows_on=1.0):
    for i, (left, w, h) in enumerate(BUILDINGS):
        r = city_rise(tc, i) * (1 - sink)
        if r <= 0:
            continue
        top = WATER_Y - h * r
        ink.rectangle(left, top, w, WATER_Y - top + 0.5)
        fill(ink, 0.95)
    # windows: lifted ink inside the buildings
    for b, col, hf, t_on in WINDOWS:
        on = smooth(ramp(tc, t_on, t_on + 0.5)) * windows_on
        left, w, h = BUILDINGS[b]
        r = city_rise(tc, b) * (1 - sink)
        if on <= 0 or r < 0.95:
            continue
        top = WATER_Y - h * r
        wx = left + col * w - 1.7
        wy = top + (1 - hf) * h * r - 2.2
        if wy < top + 1.0:
            continue
        ink.rectangle(wx, wy, 3.4, 4.4)
        erase(ink, 0.8 * on)


def draw_scene_city(ink, red, t):
    tc = t - T_CITY[0]
    calm = smooth(ramp(tc, 0.0, 1.8))
    ts = T_SEA[1] - T_SEA[0] + tc
    # the sea settles into a flat water band
    lvl = lerp(sea_level(ts), WATER_Y - WAVES[2][0] + 2.0, smooth(ramp(tc, 0.0, 1.8)))
    moon, mr = moon_state(tc)
    draw_night_sky(ink, smooth(ramp(tc, 1.2, 3.4)), moon, mr)
    draw_waves(ink, ts, calm=calm, rise=lvl, roll=False)
    bo = 1 - smooth(ramp(tc, 0.0, 1.4))
    if bo > 0:
        bx, by, tilt = boat_state(ts, calm)
        draw_boat(ink, red, bx + 6.0 * ramp(tc, 0, 1.4), by, tilt * (1 - calm), opacity=bo, scale=1.7)
    draw_city(ink, tc)


# ===================================================================== EYE

EYE_C = (CX, 57.0)
EYE_W = 25.0          # half width
EYE_H = 12.8          # half height when fully open
IRIS_R = 10.5
PUPIL_R = 4.8


def almond(ctx, cx, cy, hw, hh_top, hh_bot):
    ctx.move_to(cx - hw, cy)
    ctx.curve_to(cx - hw * 0.45, cy - hh_top * 1.33, cx + hw * 0.45, cy - hh_top * 1.33, cx + hw, cy)
    ctx.curve_to(cx + hw * 0.45, cy + hh_bot * 1.33, cx - hw * 0.45, cy + hh_bot * 1.33, cx - hw, cy)
    ctx.close_path()


def eye_open_amount(te):
    """0 closed .. 1 open: opens, blinks twice (look at viewer), stays open."""
    o = ease_out(ramp(te, 1.1, 2.3))
    for tb in (6.15, 6.85):
        if tb <= te < tb + 0.55:
            x = (te - tb) / 0.55
            o *= 1 - (ease_in(x / 0.35) if x < 0.35 else 1 - ease_out((x - 0.35) / 0.65))
    return o


def gaze(te):
    """Iris offset: looks left, right, up, then straight at the viewer."""
    keys = [(2.3, (0.0, 0.0)), (2.9, (-9.0, 0.6)), (3.6, (-9.0, 0.6)), (4.2, (8.5, -0.5)),
            (4.8, (8.5, -0.5)), (5.2, (2.5, -3.2)), (5.5, (2.5, -3.2)), (5.95, (0.0, 0.0))]
    if te <= keys[0][0]:
        return keys[0][1]
    for (t0, p0), (t1, p1) in zip(keys, keys[1:]):
        if te < t1:
            return lerp2(p0, p1, ease_in_out(ramp(te, t0, t1)))
    return keys[-1][1]


def draw_eye(ink, red, openness, look=(0.0, 0.0), socket=1.0, lid=1.0, iris=1.0, centre=EYE_C,
             width=EYE_W):
    cx, cy = centre
    hh = EYE_H * openness
    # soft socket shading around the eye
    if socket > 0:
        ellipse(ink, cx, cy, width * 1.45, EYE_H * 2.1)
        g = cairo.RadialGradient(cx, cy, 2.0, cx, cy, width * 1.45)
        g.add_color_stop_rgba(0.0, 0, 0, 0, 0.55 * socket)
        g.add_color_stop_rgba(0.72, 0, 0, 0, 0.42 * socket)
        g.add_color_stop_rgba(1.0, 0, 0, 0, 0.0)
        ink.set_source(g)
        ink.fill()
    # the eye opening: clear the socket, draw iris and pupil inside
    almond(ink, cx, cy, width, max(hh, 0.01), max(hh * 0.82, 0.01))
    erase(ink, 1.0)
    if hh > 0.2 and iris > 0:
        ix, iy = cx + look[0], cy + look[1]
        for ctx in (ink, red):
            ctx.save()
            almond(ctx, cx, cy, width, hh, hh * 0.82)
            ctx.clip()
        red.arc(ix, iy, IRIS_R, 0, TAU)
        fill(red, 0.95 * iris)
        ink.arc(ix, iy, PUPIL_R * (1.0 + 0.12 * ramp(abs(look[0]) + abs(look[1]), 3, 0)), 0, TAU)
        fill(ink, 0.97 * iris)
        # catch-light, soft and generous so it reads through the dot screen
        ink.arc(ix + 2.4, iy - 2.4, 1.7, 0, TAU)
        erase(ink, 0.75 * iris)
        red.arc(ix + 2.4, iy - 2.4, 1.7, 0, TAU)
        erase(red, 0.75 * iris)
        ink.restore()
        red.restore()
    # upper lid: a heavy brush line, thinner as it closes
    if lid > 0:
        ink.move_to(cx - width - 1.2, cy + 0.4)
        ink.curve_to(cx - width * 0.45, cy - hh * 1.33 - 2.6, cx + width * 0.45, cy - hh * 1.33 - 2.6,
                     cx + width + 1.2, cy + 0.4)
        ink.curve_to(cx + width * 0.45, cy - hh * 1.33 + 1.9, cx - width * 0.45, cy - hh * 1.33 + 1.9,
                     cx - width - 1.2, cy + 0.4)
        ink.close_path()
        fill(ink, 0.95 * lid)
        # lower lid, lighter
        ink.move_to(cx - width + 1.5, cy + 0.6)
        ink.curve_to(cx - width * 0.4, cy + hh * 1.09 + 1.1, cx + width * 0.4, cy + hh * 1.09 + 1.1,
                     cx + width - 1.5, cy + 0.6)
        ink.curve_to(cx + width * 0.4, cy + hh * 1.09 - 0.2, cx - width * 0.4, cy + hh * 1.09 - 0.2,
                     cx - width + 1.5, cy + 0.6)
        ink.close_path()
        fill(ink, 0.55 * lid)


def draw_scene_eye(ink, red, t):
    te = t - T_EYE[0]
    tc = T_CITY[1] - T_CITY[0] + te
    m = smooth(ramp(te, 0.0, 1.3))              # moon -> eye morph
    # the city dissolves while the moon drifts to the centre
    fade = smooth(ramp(te, 0.05, 1.1))
    if fade < 1:
        moon, mr = moon_state(tc)
        ink.push_group()
        draw_night_sky(ink, 1.0, moon, mr)
        draw_waves(ink, T_SEA[1] - T_SEA[0] + tc, calm=1.0,
                   rise=WATER_Y - WAVES[2][0] + 2.0, roll=False)
        draw_city(ink, tc)
        ink.pop_group_to_source()
        ink.paint_with_alpha(1.0 - fade)
    moon, mr = moon_state(tc)
    centre = lerp2(moon, EYE_C, m)
    width = lerp(mr, EYE_W, m)
    openness = eye_open_amount(te)
    if m < 1:
        # the moon: a bright disk stretching into a closed eye
        ellipse(ink, centre[0], centre[1], width, lerp(mr, 0.4, m))
        erase(ink, 1.0)
    draw_eye(ink, red, openness, gaze(te), socket=m, lid=m, iris=ramp(te, 1.2, 1.8),
             centre=centre, width=width)


# ================================================================== RETURN

def draw_scene_return(ink, red, t):
    tr = t - T_RETURN[0]
    openness = 1 - ease_in_out(ramp(tr, 0.0, 1.2))
    draw_eye(ink, red, openness, (0.0, 0.0))


# ================================================================ timeline

SCENES = {
    "flower": draw_scene_flower,
    "bird": draw_scene_bird,
    "sea": draw_scene_sea,
    "city": draw_scene_city,
    "eye": draw_scene_eye,
    "return": draw_scene_return,
}


def draw(ink, red, t, box=None):
    """Compositor entry point: paint the picture for film time `t`."""
    act = act_at(t)
    fn = SCENES.get(act)
    if fn is not None:
        fn(ink, red, t)
    return {"act": act}


def look(t: float) -> compose.Look:
    """Global styling of the frame at film time `t`."""
    if t < T_TITLE[1]:
        clean = 1.0 - smooth(ramp(t, T_TITLE[1] - 0.3, T_TITLE[1] + 0.9))
        return compose.Look(picture=0.0, clean=clean, reveal=ease_in_out(ramp(t, 0.35, 3.2)))
    if t < T_RETURN[0]:
        clean = 1.0 - smooth(ramp(t, T_TITLE[1] - 0.3, T_TITLE[1] + 0.9))
        return compose.Look(picture=smooth(ramp(t, T_FLOWER[0], T_FLOWER[0] + 0.4)), clean=clean)
    if t < T_LINK[0]:
        tr = t - T_RETURN[0]
        return compose.Look(picture=1.0 - smooth(ramp(tr, 0.8, 2.6)), clean=smooth(ramp(tr, 1.0, 2.8)))
    return compose.Look(picture=0.0, clean=1.0)

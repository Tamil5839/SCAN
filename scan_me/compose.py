"""Compose one frame: paper + picture + code dots.

Layers, bottom to top:
  1. PAPER   warm off-white with a static, very low-contrast grain
  2. PICTURE the scene as an ink wash (soft edges, capped darkness), faded
             out near the quiet zone and near light function modules
  3. CODE    a rounded-square dot at the centre of every data module in its
             true colour, and full, solid function patterns on top

Why the picture is a soft wash rather than hard near-black shapes: OpenCV's
QR decoder binarises with a local (Gaussian, blockSize 83, C=2) threshold and
reads each module from the white fraction of its *whole cell*. Every sharp
picture edge therefore prints a ~1 module wide black band on its dark side,
and a bright dot inside a dark area prints a black ring around itself. Soft
edges, a darkest tone well above the dots' black and no bright dots inside
ink keep both OpenCV and zxing (which samples module centres) reading every
module correctly. scanmodel.py predicts both reads; `frame()` uses it as a
safety net and re-renders when a module's margin is thin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cairo
import cv2
import numpy as np

import config
import qrlayer
import scanmodel
from scanmodel import Geometry

_ATLAS_STEPS = 64
_GROW_MAX = 0.28      # safety net may enlarge a dark dot by up to this
_REF_SIZE = 1080.0


def hex_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def luma(rgb) -> float:
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass
class Look:
    """Per-frame styling decided by the storyboard (scenes.py)."""

    picture: float = 1.0        # global picture opacity
    clean: float = 0.0          # 0 = picture dots, 1 = clean code dots
    reveal: float | None = None  # title draw-in wave progress 0..1


@dataclass
class SafetyReport:
    iterations: int = 0
    cv_block_errors: int = 0     # worst RS block, predicted wrong codewords
    zx_block_errors: int = 0
    cv_min_margin: float = 0.0   # smallest signed margin over data modules
    zx_min_margin: float = 0.0
    weak: int = 0                # modules below the margin targets
    risky_codewords: int = 0     # worst RS block: codewords holding a weak module
    lifted: int = 0              # modules where the picture was lifted
    grown: int = 0               # dark dots enlarged beyond their style size


@dataclass
class Picture:
    """Ink and accent densities (0..1) at full frame resolution."""

    ink: np.ndarray
    red: np.ndarray
    meta: dict = field(default_factory=dict)


class Compositor:
    def __init__(self, size: int = None, version: int = 4, ec: str = None):
        self.size = size or config.SIZE_PX
        self.ec = ec or config.EC_LEVEL
        self.lay = qrlayer.layout(version, self.ec)
        self.geo = Geometry.fit(self.size, self.lay.n, config.QUIET_ZONE)
        self.k = self.size / _REF_SIZE          # scale relative to 1080 px

        self.paper = hex_rgb(config.PAPER)
        self.ink = hex_rgb(config.INK)
        self.accent = hex_rgb(config.ACCENT)
        self.dot_dark = hex_rgb(config.DOT_DARK)
        self.dot_light = hex_rgb(config.DOT_LIGHT)
        self.luma_paper = luma(self.paper)
        self.luma_min = luma(self.paper + (self.ink - self.paper) * config.PICTURE_INK_MAX)

        self.paper_rgb = self._paper_texture()
        self.safe = self._safe_mask()
        self.atlas = self._dot_atlas()
        self._overlays = {}
        n = self.lay.n
        r, c = np.mgrid[0:n, 0:n]
        self.wave_delay = (r + c) / (2.0 * (n - 1))

    # ------------------------------------------------------------ geometry

    @property
    def module(self) -> int:
        return self.geo.module

    def scene_to_px(self) -> float:
        """Scene coordinates are 0..100 across the whole frame."""
        return self.size / 100.0

    def canvas_box(self):
        """Region (scene units) where the picture is at full strength."""
        g = self.geo
        lo = g.origin + (6.5 + config.SAFE_FADE_END) * g.module
        hi = g.origin + (g.n - config.SAFE_FADE_END) * g.module
        s = 100.0 / self.size
        return lo * s, lo * s, hi * s, hi * s

    # -------------------------------------------------------------- layers

    def _paper_texture(self) -> np.ndarray:
        rng = np.random.default_rng(config.SEED)
        h = w = self.size
        fine = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 0.8 * self.k + 0.3)
        fine /= fine.std() + 1e-6
        mottle = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 28 * self.k)
        mottle /= mottle.std() + 1e-6
        grain = config.PAPER_GRAIN * (0.5 * fine + 1.0 * mottle)
        return self.paper[None, None, :] + grain[..., None]

    def _safe_mask(self) -> np.ndarray:
        """1 where the picture may be at full strength, 0 near hazards.

        Hazards are the quiet zone and every function module except the
        alignment pattern (its outer ring is dark, which is harmless).
        """
        g, lay = self.geo, self.lay
        # distance to the quiet zone: the picture fades out gently toward the
        # symbol's border (a steep fade prints a black band into the last
        # rows of modules)
        inside = np.zeros((self.size, self.size), np.uint8)
        inside[g.origin:g.origin + g.extent, g.origin:g.origin + g.extent] = 1
        border = cv2.distanceTransform(inside, cv2.DIST_L2, 5) / g.module
        border_fade = smoothstep(config.BORDER_FADE_START, config.BORDER_FADE_END, border)
        hazard = np.zeros((self.size, self.size), np.uint8)
        func = lay.function.copy()
        pos = qrlayer.consts.ALIGNMENT_POS[lay.version - 2] if lay.version > 1 else ()
        for ar in pos:
            for ac in pos:
                if func[ar, ac] and not (ar < 8 and ac < 8) and not (ar < 8 and ac > lay.n - 9) \
                        and not (ar > lay.n - 9 and ac < 8):
                    func[ar - 2:ar + 3, ac - 2:ac + 3] = False
        m = g.module
        big = np.kron(func, np.ones((m, m), np.uint8)).astype(bool)
        hazard[g.origin:g.origin + g.extent, g.origin:g.origin + g.extent][big] = 1
        free = (1 - hazard).astype(np.uint8)
        dist = cv2.distanceTransform(free, cv2.DIST_L2, 5) / m
        safe = smoothstep(config.SAFE_FADE_START, config.SAFE_FADE_END, dist) * border_fade
        # keep pictures out of the thin strips between the finder patterns
        # and the timing lines: they would read as stray patches
        lo = g.origin + (6.5 + config.SAFE_FADE_END) * m
        xs = np.arange(self.size, dtype=np.float32)
        edge = smoothstep(lo - 4.5 * m / 2.6, lo + 0.4 * m, xs)
        return (safe * edge[None, :] * edge[:, None]).astype(np.float32)

    def _dot_atlas(self) -> np.ndarray:
        m = self.module
        atlas = np.zeros((_ATLAS_STEPS + 1, m, m), np.float32)
        for k in range(1, _ATLAS_STEPS + 1):
            s = m * k / _ATLAS_STEPS
            surf = cairo.ImageSurface(cairo.FORMAT_A8, m, m)
            ctx = cairo.Context(surf)
            rounded_rect(ctx, (m - s) / 2, (m - s) / 2, s, s, s * config.DOT_CORNER)
            ctx.set_source_rgba(0, 0, 0, 1)
            ctx.fill()
            surf.flush()
            a = np.ndarray((m, surf.get_stride()), np.uint8, buffer=surf.get_data())[:, :m]
            atlas[k] = a / 255.0
        return atlas

    def overlay(self, mask: int):
        """Function patterns for a given mask (format bits depend on it)."""
        if mask not in self._overlays:
            self._overlays[mask] = self._build_overlay(mask)
        return self._overlays[mask]

    def _build_overlay(self, mask: int):
        lay, g = self.lay, self.geo
        dark = qrlayer.matrix("0", lay.version, lay.ec, mask)   # function modules are payload independent
        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.size, self.size)
        ctx = cairo.Context(surf)
        ctx.set_antialias(cairo.ANTIALIAS_NONE)
        m, o, n = g.module, g.origin, lay.n
        finders = [(0, 0), (0, n - 7), (n - 7, 0)]

        def in_finder(r, c):
            return any(fr <= r < fr + 7 and fc <= c < fc + 7 for fr, fc in finders)

        for r in range(n):
            for c in range(n):
                if not lay.function[r, c] or in_finder(r, c):
                    continue
                col = self.dot_dark if dark[r, c] else self.dot_light
                ctx.set_source_rgb(*(col / 255.0))
                ctx.rectangle(o + c * m, o + r * m, m, m)
                ctx.fill()
        ctx.set_antialias(cairo.ANTIALIAS_DEFAULT)
        rad = config.FINDER_CORNER * m
        for fr, fc in finders:
            x, y = o + fc * m, o + fr * m
            ctx.set_source_rgb(*(self.dot_dark / 255.0))
            rounded_rect(ctx, x, y, 7 * m, 7 * m, rad)
            ctx.fill()
            ctx.set_source_rgb(*(self.dot_light / 255.0))
            ctx.rectangle(x + m, y + m, 5 * m, 5 * m)
            ctx.fill()
            ctx.set_source_rgb(*(self.dot_dark / 255.0))
            ctx.rectangle(x + 2 * m, y + 2 * m, 3 * m, 3 * m)
            ctx.fill()
        surf.flush()
        buf = np.ndarray((self.size, self.size, 4), np.uint8, buffer=surf.get_data())
        alpha = buf[..., 3].astype(np.float32) / 255.0
        rgb = buf[..., [2, 1, 0]].astype(np.float32)          # premultiplied
        o2, e = g.origin, g.extent
        return rgb[o2:o2 + e, o2:o2 + e].copy(), alpha[o2:o2 + e, o2:o2 + e].copy()

    # -------------------------------------------------------------- picture

    def render_picture(self, draw, t: float, look: Look) -> Picture:
        """Rasterise scene densities at half resolution, soften, upsample.

        `draw(ink_ctx, red_ctx, t, canvas)` paints densities (alpha) in scene
        units (0..100 across the frame).
        """
        h = self.size // 2
        surfs = [cairo.ImageSurface(cairo.FORMAT_A8, h, h) for _ in range(2)]
        ctxs = [cairo.Context(s) for s in surfs]
        for ctx in ctxs:
            ctx.scale(h / 100.0, h / 100.0)
        meta = draw(ctxs[0], ctxs[1], t, self.canvas_box()) or {}
        out = []
        sigma = config.PICTURE_SOFTNESS_PX * self.k / 2.0
        for s in surfs:
            s.flush()
            a = np.ndarray((h, s.get_stride()), np.uint8, buffer=s.get_data())[:, :h].astype(np.float32) / 255.0
            if sigma > 0:
                a = cv2.GaussianBlur(a, (0, 0), sigma)
            a = cv2.resize(a, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            out.append(a * self.safe * float(look.picture))
        return Picture(out[0], out[1], meta)

    def blank_picture(self) -> Picture:
        z = np.zeros((self.size, self.size), np.float32)
        return Picture(z, z)

    def colorize(self, pic: Picture, lift_px: np.ndarray | None = None) -> np.ndarray:
        ink = pic.ink
        red = pic.red * (1.0 - ink)          # ink covers the accent, it does not stack on it
        if lift_px is not None:
            keep = 1.0 - lift_px
            ink = ink * keep
            red = red * keep
        rgb = self.paper_rgb + (self.accent - self.paper_rgb) * red[..., None]
        rgb = rgb + (self.ink[None, None, :] - rgb) * (ink * config.PICTURE_INK_MAX)[..., None]
        return rgb

    def cell_darkness(self, rgb: np.ndarray) -> np.ndarray:
        g, n, m = self.geo, self.lay.n, self.geo.module
        area = rgb[g.origin:g.origin + g.extent, g.origin:g.origin + g.extent]
        lum = area @ np.array([0.299, 0.587, 0.114], np.float32)
        cell = lum.reshape(n, m, n, m).mean(axis=(1, 3))
        return np.clip((self.luma_paper - cell) / (self.luma_paper - self.luma_min), 0.0, 1.0)

    # ----------------------------------------------------------------- dots

    def dot_sizes(self, dark: np.ndarray, darkness: np.ndarray, look: Look):
        """Dot side (fraction of module) for every module."""
        fmin, fmax, fclean = config.DOT_FRACTION, config.DOT_FRACTION_MAX, config.CLEAN_DOT_FRACTION
        # dark module: grows with the ink under it (matched = larger)
        dark_size = fmin + (fmax - fmin) * darkness
        if look.clean > 0:
            dark_size = dark_size + (fclean - dark_size) * look.clean
        # light module: a paper-white dot over paper, vanishing over ink (a
        # bright dot inside ink reads as a dark ring to area-sampling decoders)
        light_size = fmin * (1.0 - smoothstep(0.02, 0.12, darkness))
        size = np.where(dark, dark_size, light_size)
        if look.reveal is not None:
            w = 0.22
            a = np.clip((look.reveal * (1 + w) - self.wave_delay) / w, 0.0, 1.0)
            size = fmin + (size - fmin) * ease_out_back(a)
        return np.clip(size, 0.0, 1.0)

    def draw_code(self, rgb: np.ndarray, dark: np.ndarray, sizes: np.ndarray, mask: int) -> np.ndarray:
        g, lay = self.geo, self.lay
        n, m, o, e = lay.n, g.module, g.origin, g.extent
        idx = np.rint(sizes * _ATLAS_STEPS).astype(np.int32)
        idx[lay.function] = 0
        alpha = self.atlas[idx].transpose(0, 2, 1, 3).reshape(e, e)
        colors = np.where(dark[..., None], self.dot_dark, self.dot_light)          # (n, n, 3)
        col_px = np.repeat(np.repeat(colors, m, axis=0), m, axis=1)
        out = rgb.copy()
        area = out[o:o + e, o:o + e]
        area += (col_px - area) * alpha[..., None]
        ov_rgb, ov_a = self.overlay(mask)
        area *= (1.0 - ov_a[..., None])
        area += ov_rgb
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)

    # ---------------------------------------------------------------- frame

    def frame(self, pic: Picture, dark: np.ndarray, mask: int, look: Look,
              safety: bool = True):
        """Render the final RGB frame; returns (rgb uint8, SafetyReport)."""
        lay, g = self.lay, self.geo
        n, m = lay.n, g.module
        data = lay.data
        lift = np.zeros((n, n), np.float32)
        grow = np.zeros((n, n), np.float32)
        report = SafetyReport()
        iterations = config.SAFETY_ITERATIONS if safety else 0
        for it in range(iterations + 1):
            lift_px = None
            if lift.any():
                lift_px = self._lift_pixels(lift)
            rgb_pic = self.colorize(pic, lift_px)
            darkness = self.cell_darkness(rgb_pic)
            sizes = self.dot_sizes(dark, darkness, look)
            sizes = np.where(dark, np.minimum(sizes + grow, 0.96), sizes)
            rgb = self.draw_code(rgb_pic, dark, sizes, mask)
            if not safety:
                break
            report, fix = self.check(rgb, dark)
            report.iterations = it
            report.lifted = int((lift > 0).sum())
            report.grown = int((grow > 0).sum())
            # strengthen what can still be strengthened
            fix &= np.where(dark, (grow < _GROW_MAX) & config.SAFETY_GROW_DOTS, lift < 1.0)
            if not fix.any() or it == iterations:
                break
            grow = np.where(fix & dark, np.minimum(grow + 0.07, _GROW_MAX), grow)
            lift = np.where(fix & ~dark, np.minimum(lift + 0.34, 1.0), lift)
        return rgb, report

    def _lift_pixels(self, lift: np.ndarray) -> np.ndarray:
        g = self.geo
        m = g.module
        px = np.zeros((self.size, self.size), np.float32)
        px[g.origin:g.origin + g.extent, g.origin:g.origin + g.extent] = np.kron(lift, np.ones((m, m), np.float32))
        px = cv2.GaussianBlur(px, (0, 0), 0.55 * m)
        return np.clip(px * 2.2, 0.0, 1.0)

    def check(self, rgb: np.ndarray, dark: np.ndarray):
        """Predict both decoders' reads of a clean frame.

        Returns (report, fix) where `fix` marks the modules to strengthen:
        every module predicted to read wrong, plus the thin-margin modules of
        any RS block whose count of at-risk codewords exceeds the budget.
        """
        lay = self.lay
        gray = scanmodel.to_gray(rgb)
        cv_dark, cv_margin = scanmodel.opencv_read(gray, self.geo)
        zx_dark, zx_centre, zx_worst = scanmodel.zxing_read(gray, self.geo, jitter=max(1, round(2 * self.k)))
        data = lay.data
        # signed margins, positive = read on the correct side
        cv_ok = np.where(dark, -cv_margin, cv_margin)
        zx_ok = np.where(dark, -zx_worst, zx_worst).astype(np.float32)
        wrong = data & ((cv_dark != dark) | (zx_dark != dark))
        weak = data & ((cv_ok < config.MARGIN_OPENCV) | (zx_ok < config.MARGIN_ZXING))
        # how bad each module is: >= 1 means predicted to read wrong
        bad = np.maximum((config.MARGIN_OPENCV - cv_ok) / config.MARGIN_OPENCV,
                         (config.MARGIN_ZXING - zx_ok) / config.MARGIN_ZXING)
        bad = np.where(weak | wrong, np.maximum(bad, 0.0), 0.0)
        fix = np.zeros_like(weak)
        cw = lay.codeword
        for b in range(lay.n_blocks):
            in_b = (lay.block == b) & (bad > 0)
            if not in_b.any():
                continue
            words = np.unique(cw[in_b])
            worst = np.array([bad[in_b & (cw == w)].max() for w in words])
            n_wrong = int((worst >= 1.0).sum())
            need = max(n_wrong - config.SAFE_MAX_ERRORS, len(words) - config.SAFE_MAX_RISKY, 0)
            for w in words[np.argsort(-worst)][:need]:
                fix |= in_b & (cw == w)
        risky = qrlayer.codeword_errors(weak | wrong, np.zeros_like(weak), lay)
        rep = SafetyReport()
        rep.cv_block_errors = int(qrlayer.codeword_errors(cv_dark, dark, lay).max())
        rep.zx_block_errors = int(qrlayer.codeword_errors(zx_dark, dark, lay).max())
        rep.cv_min_margin = float(cv_ok[data].min())
        rep.zx_min_margin = float(zx_ok[data].min())
        rep.weak = int(weak.sum())
        rep.risky_codewords = int(risky.max())
        return rep, fix


def rounded_rect(ctx, x, y, w, h, r):
    r = max(0.0, min(r, w / 2, h / 2))
    if r <= 0:
        ctx.rectangle(x, y, w, h)
        return
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -np.pi / 2, 0)
    ctx.arc(x + w - r, y + h - r, r, 0, np.pi / 2)
    ctx.arc(x + r, y + h - r, r, np.pi / 2, np.pi)
    ctx.arc(x + r, y + r, r, np.pi, 1.5 * np.pi)
    ctx.close_path()


def ease_out_back(x, s=1.4):
    x = np.asarray(x, dtype=np.float32)
    y = 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2
    return np.where(x <= 0, 0.0, np.where(x >= 1, 1.0, y))

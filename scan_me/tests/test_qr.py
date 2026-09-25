"""Tests for the QR layer, the schedule and the compositor.

Run from scan_me/:  python -m pytest -q
"""

import re

import cv2
import numpy as np
import pytest
from segno import consts, encoder

import compose
import config
import qrlayer
import readers
import scanmodel
import schedule
import scenes

SEGS = schedule.build()
PAYLOADS = schedule.payloads(SEGS)
VERSION = qrlayer.min_version(PAYLOADS, config.EC_LEVEL)
LONGEST = max(PAYLOADS, key=lambda p: len(p.encode("utf-8")))


def plain_render(dark, scale=8, quiet=4):
    """Black-on-white rendering of a module matrix."""
    img = np.where(dark, 0, 255).astype(np.uint8)
    img = np.pad(img, quiet, constant_values=255)
    img = np.kron(img, np.ones((scale, scale), np.uint8))
    return np.stack([img] * 3, axis=-1)


# ------------------------------------------------------------------ qrlayer

def test_version_is_smallest_that_fits_longest_payload():
    assert VERSION == qrlayer.min_version([LONGEST], "H")
    capacity = consts.SYMBOL_CAPACITY[VERSION][consts.ERROR_LEVEL_H]
    assert all(len(encoder.prepare_data(p, None, "utf-8")) for p in PAYLOADS)
    # every payload encodes at the film version, the longest would not fit one version lower
    for p in PAYLOADS:
        qrlayer.matrix(p, VERSION, "H", 0)
    if VERSION > 1:
        with pytest.raises(Exception):
            qrlayer.matrix(LONGEST, VERSION - 1, "H", 0)
    assert capacity > 0


def test_default_film_uses_version_4():
    assert VERSION == 4
    assert qrlayer.layout(4).n == 33


@pytest.mark.parametrize("mask", range(8))
def test_all_masks_decode_plainly(mask):
    dark = qrlayer.matrix(LONGEST, VERSION, "H", mask)
    img = plain_render(dark)
    assert readers.zxing(img) == LONGEST
    assert readers.opencv(img) == LONGEST


def test_fixed_version_keeps_grid_identical():
    shapes = {qrlayer.matrix(p, VERSION, "H", 0).shape for p in PAYLOADS}
    assert shapes == {(VERSION * 4 + 17,) * 2}


def test_function_map_matches_standard_counts():
    for v, data_modules in ((2, 359), (4, 807), (5, 1079), (7, 1568)):
        assert int(qrlayer.layout(v).data.sum()) == data_modules


def test_function_patterns_are_payload_and_mask_independent_except_format():
    lay = qrlayer.layout(VERSION)
    a = qrlayer.matrix(PAYLOADS[0], VERSION, "H", 0)
    b = qrlayer.matrix(PAYLOADS[-1], VERSION, "H", 0)
    assert np.array_equal(a[lay.function], b[lay.function])


@pytest.mark.parametrize("payload", [LONGEST, PAYLOADS[1], config.FINAL_URL])
def test_codeword_map_is_exact(payload):
    """De-interleave via the module map and recompute RS: must match what is in the symbol."""
    lay = qrlayer.layout(VERSION)

    class Buf:
        def __init__(self, ints):
            self.ints = ints

        def toints(self):
            return iter(self.ints)

    for mask in range(8):
        dark = qrlayer.matrix(payload, VERSION, "H", mask)
        data, ec = qrlayer.read_codewords(dark, mask, lay)
        _, expected = encoder.make_blocks(consts.ECC[VERSION][consts.ERROR_LEVEL_H],
                                          Buf([w for blk in data for w in blk]))
        assert [list(e) for e in expected] == ec


def test_codeword_errors_counts_distinct_codewords():
    lay = qrlayer.layout(VERSION)
    dark = qrlayer.matrix(LONGEST, VERSION, "H", 0)
    read = dark.copy()
    rc = np.argwhere(lay.codeword == 5)[:3]           # three modules of one codeword
    for r, c in rc:
        read[r, c] = not read[r, c]
    errs = qrlayer.codeword_errors(read, dark, lay)
    assert errs.sum() == 1


# ----------------------------------------------------------------- schedule

def test_schedule_covers_every_frame_once():
    total = round(config.DURATION_SEC * config.FPS)
    assert SEGS[0].start == 0 and SEGS[-1].end == total
    for a, b in zip(SEGS, SEGS[1:]):
        assert a.end == b.start
    assert schedule.frame_count(SEGS) == total


def test_title_words_bonus_and_link_placement():
    fps = config.FPS
    assert schedule.segment_at(SEGS, 0).payload == config.TITLE_TEXT
    assert schedule.segment_at(SEGS, int(config.TITLE_END_SEC * fps) - 1).payload == config.TITLE_TEXT
    assert schedule.segment_at(SEGS, int(config.LINK_START_SEC * fps)).payload == config.FINAL_URL
    assert schedule.segment_at(SEGS, schedule.frame_count(SEGS) - 1).payload == config.FINAL_URL
    hold = round(config.WORD_HOLD_SEC * fps)
    for s in SEGS:
        if s.kind in ("word", "bonus"):
            assert s.end - s.start >= hold
    assert sum(s.kind == "bonus" for s in SEGS) == len(config.BONUS_MESSAGES)


def test_words_are_numbered_in_order_and_loop():
    words = config.SECRET_SENTENCE.split()
    seq = [schedule.parse_word_payload(s.payload) for s in SEGS if s.kind == "word"]
    assert all(p is not None for p in seq)
    for k, (i, n, w) in enumerate(seq):
        assert n == len(words)
        assert i == k % len(words) + 1
        assert w == words[i - 1]
    assert len(seq) > len(words)            # the sentence loops


def test_sentence_reconstructs_from_payloads():
    decoded = [schedule.segment_at(SEGS, f).payload for f in range(schedule.frame_count(SEGS))]
    assert schedule.reconstruct_sentence(decoded) == config.SECRET_SENTENCE


def test_payloads_are_short_plain_text_and_one_url():
    urlish = [p for p in PAYLOADS if re.match(r"^[a-z]+://", p, re.I)]
    assert urlish == [config.FINAL_URL]
    for p in PAYLOADS:
        assert p.isprintable() and len(p.encode("utf-8")) <= 64


# ---------------------------------------------------------------- compositor

@pytest.fixture(scope="module")
def comp():
    return compose.Compositor(version=VERSION)


def _frame(comp, t, payload, mask=0):
    look = scenes.look(t)
    pic = comp.render_picture(scenes.draw, t, look) if look.picture > 0 else comp.blank_picture()
    dark = qrlayer.matrix(payload, VERSION, "H", mask)
    rgb, rep = comp.frame(pic, dark, mask, look)
    return rgb, dark, rep


def test_function_patterns_are_solid_and_standard(comp):
    rgb, dark, _ = _frame(comp, 40.0, PAYLOADS[3], mask=2)
    g, lay = comp.geo, comp.lay
    m = g.module
    inset = 3                                   # skip the rounded finder corners
    for r, c in np.argwhere(lay.function):
        y0, x0 = g.origin + r * m, g.origin + c * m
        cell = rgb[y0 + inset:y0 + m - inset, x0 + inset:x0 + m - inset].reshape(-1, 3).astype(int)
        want = compose.hex_rgb(config.DOT_DARK if dark[r, c] else config.DOT_LIGHT)
        assert np.abs(cell - want).max() <= 1, (r, c)


def test_quiet_zone_is_clean_paper(comp):
    rgb, _, _ = _frame(comp, 30.0, PAYLOADS[2])
    g = comp.geo
    q = np.ones(rgb.shape[:2], bool)
    q[g.origin:g.origin + g.extent, g.origin:g.origin + g.extent] = False
    paper = compose.hex_rgb(config.PAPER)
    dev = np.abs(rgb[q].astype(float) - paper).max()
    assert dev <= 6 * config.PAPER_GRAIN + 1


def test_picture_never_darker_than_its_cap(comp):
    look = scenes.look(47.0)
    pic = comp.render_picture(scenes.draw, 47.0, look)
    rgb = comp.colorize(pic)
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    assert lum.min() >= comp.luma_min - 6 * config.PAPER_GRAIN - 1


@pytest.mark.parametrize("t", [1.0, 12.0, 20.0, 47.5, 57.0])
def test_composed_frames_decode(comp, t):
    payload = schedule.segment_at(SEGS, int(t * config.FPS)).payload
    rgb, _, _ = _frame(comp, t, payload, mask=3)
    assert readers.zxing(rgb) == payload
    assert readers.opencv(rgb) == payload


def test_rendering_is_deterministic(comp):
    a, _, _ = _frame(comp, 21.0, PAYLOADS[1], mask=5)
    b, _, _ = _frame(compose.Compositor(version=VERSION), 21.0, PAYLOADS[1], mask=5)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------- scanmodel

def _zxing_threshold_reference(gray):
    """Straight port of zxing-cpp HybridBinarizer (new algorithm), loops."""
    h, w = gray.shape
    B = 8
    sh, sw = (h + B - 1) // B, (w + B - 1) // B
    thr = np.zeros((sh, sw), int)
    for y in range(sh):
        y0 = min(y * B, h - B)
        for x in range(sw):
            x0 = min(x * B, w - B)
            blk = gray[y0:y0 + B, x0:x0 + B]
            mn, mx = int(blk.min()), int(blk.max())
            thr[y, x] = (mx + mn) // 2 if mx - mn > 24 else 0
    out = np.zeros_like(thr)
    for y in range(sh):
        top = min(max(y, 2), sh - 3)
        for x in range(sw):
            left = min(max(x, 2), sw - 3)
            win = thr[top - 2:top + 3, left - 2:left + 3]
            s = thr[y, x] * 2 + win.sum()
            n = (2 if thr[y, x] > 0 else 0) + (win > 0).sum()
            out[y, x] = s // n if n else 0
    flat = out.ravel()
    last = -1
    for i in range(flat.size):
        if flat[i]:
            if last != i - 1:
                flat[last + 1:i] = flat[i]
            last = i
    flat[last + 1:] = flat[max(last, 0)]
    out = flat.reshape(sh, sw)
    full = np.zeros((h, w), int)
    for y in range(sh):                                     # later blocks overwrite, like ThresholdImage
        y0 = min(y * B, h - B)
        for x in range(sw):
            x0 = min(x * B, w - B)
            full[y0:y0 + B, x0:x0 + B] = out[y, x]
    return full


def test_zxing_model_matches_reference_port():
    rng = np.random.default_rng(1)
    img = np.full((163, 205), 230, np.uint8)                 # not a multiple of 8
    for _ in range(40):
        x, y = rng.integers(0, 195), rng.integers(0, 153)
        img[y:y + rng.integers(3, 20), x:x + rng.integers(3, 20)] = rng.integers(0, 120)
    img[100:163, 0:60] = 200                    # a flat area exercises the gap filling
    assert np.array_equal(scanmodel.zxing_threshold_map(img), _zxing_threshold_reference(img))


def test_opencv_model_agrees_with_opencv_on_plain_code():
    dark = qrlayer.matrix(LONGEST, VERSION, "H", 1)
    geo = scanmodel.Geometry.fit(8 * (dark.shape[0] + 8), dark.shape[0])
    img = plain_render(dark, scale=geo.module)
    read, _ = scanmodel.opencv_read(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), geo)
    assert np.array_equal(read[qrlayer.layout(VERSION).data], dark[qrlayer.layout(VERSION).data])


# ------------------------------------------------------------------ render

def test_repair_smoothing_keeps_needs_and_ramps():
    import render
    seg = SEGS[5]
    frames = list(range(seg.start - 3, seg.start + 20))          # crosses a segment boundary
    n = qrlayer.layout(VERSION).n
    maps = {i: np.zeros((n, n), np.float32) for i in frames}
    maps[seg.start + 10][4, 7] = 1.0                            # one repair, one frame
    maps[seg.start - 2][9, 9] = 1.0                             # previous segment
    out = render.smooth_repairs(SEGS, frames, maps, reach=4)
    for i in frames:
        assert (out[i] >= maps[i] - 1e-6).all()                 # never less than needed
    ramp = [float(out[i][4, 7]) for i in range(seg.start, seg.start + 20)]
    assert max(abs(a - b) for a, b in zip(ramp, ramp[1:])) <= 0.2 + 1e-6
    assert out[seg.start][9, 9] == 0.0                          # not carried across the boundary

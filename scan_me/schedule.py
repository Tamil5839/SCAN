"""Payload schedule: which text every frame of the film encodes.

0:00 - TITLE_END_SEC            TITLE_TEXT
TITLE_END_SEC - LINK_START_SEC  words of SECRET_SENTENCE, one per slot of
                                WORD_HOLD_SEC, "3/8 - THE", looping; bonus
                                messages take over a few slots
LINK_START_SEC - end            FINAL_URL
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

import config


@dataclass
class Segment:
    index: int
    kind: str            # "title" | "word" | "bonus" | "link"
    payload: str
    start: int           # first frame (inclusive)
    end: int             # last frame (exclusive)
    word: int = 0        # 1-based word position for kind == "word"
    mask: int = -1       # chosen QR mask, filled in by render.py

    @property
    def frames(self) -> range:
        return range(self.start, self.end)

    def seconds(self, fps: int) -> tuple:
        return self.start / fps, self.end / fps


def words(sentence: str = None):
    return (sentence or config.SECRET_SENTENCE).split()


def word_payload(i: int, n: int, word: str, template: str = None) -> str:
    return (template or config.WORD_TEMPLATE).format(i=i, n=n, word=word)


def build(cfg=config) -> list:
    fps = cfg.FPS
    total = round(cfg.DURATION_SEC * fps)
    title_end = round(cfg.TITLE_END_SEC * fps)
    link_start = round(cfg.LINK_START_SEC * fps)
    hold = round(cfg.WORD_HOLD_SEC * fps)
    if not (0 < title_end < link_start < total):
        raise ValueError("need 0 < TITLE_END_SEC < LINK_START_SEC < DURATION_SEC")

    n_slots = max(1, (link_start - title_end) // hold)
    starts = [title_end + k * hold for k in range(n_slots)]
    ends = starts[1:] + [link_start]          # last slot absorbs any remainder

    # place bonus messages on the nearest free slot
    bonus_at = {}
    for text, sec in cfg.BONUS_MESSAGES:
        k = int(round((sec * fps - title_end) / hold))
        k = min(max(k, 0), n_slots - 1)
        while k in bonus_at and k < n_slots - 1:
            k += 1
        if k in bonus_at:
            raise ValueError("too many bonus messages for the film length")
        bonus_at[k] = text

    segs = [Segment(0, "title", cfg.TITLE_TEXT, 0, title_end)]
    ws = words(cfg.SECRET_SENTENCE)
    w = 0
    for k in range(n_slots):
        if k in bonus_at:
            segs.append(Segment(len(segs), "bonus", bonus_at[k], starts[k], ends[k]))
        else:
            i = w % len(ws)
            segs.append(Segment(len(segs), "word",
                                word_payload(i + 1, len(ws), ws[i], cfg.WORD_TEMPLATE),
                                starts[k], ends[k], word=i + 1))
            w += 1
    segs.append(Segment(len(segs), "link", cfg.FINAL_URL, link_start, total))
    return segs


def frame_count(segs) -> int:
    return segs[-1].end


def segment_at(segs, frame: int) -> Segment:
    lo, hi = 0, len(segs) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if segs[mid].start <= frame:
            lo = mid
        else:
            hi = mid - 1
    return segs[lo]


def payloads(segs):
    """Distinct payloads in order of first appearance."""
    seen = []
    for s in segs:
        if s.payload not in seen:
            seen.append(s.payload)
    return seen


def parse_word_payload(payload: str, template: str = None):
    """Inverse of word_payload: returns (i, n, word) or None."""
    template = template or config.WORD_TEMPLATE
    pattern = re.escape(template)
    for key, rx in (("i", r"(?P<i>\d+)"), ("n", r"(?P<n>\d+)"), ("word", r"(?P<word>.+?)")):
        pattern = pattern.replace(re.escape("{" + key + "}"), rx)
    m = re.fullmatch(pattern, payload)
    if not m:
        return None
    return int(m["i"]), int(m["n"]), m["word"]


def reconstruct_sentence(decoded_payloads) -> str:
    """Rebuild the hidden sentence from payloads as a viewer would scan them."""
    found, total = {}, 0
    for p in decoded_payloads:
        parsed = parse_word_payload(p)
        if parsed:
            i, n, word = parsed
            found.setdefault(i, word)
            total = max(total, n)
    return " ".join(found.get(i, "?") for i in range(1, total + 1))


def to_json(segs, version: int, fps: int = None) -> dict:
    fps = fps or config.FPS
    return {
        "fps": fps,
        "frames": frame_count(segs),
        "qr_version": version,
        "ec_level": config.EC_LEVEL,
        "secret_sentence": config.SECRET_SENTENCE,
        "final_url": config.FINAL_URL,
        "segments": [
            dict(asdict(s),
                 start_time=round(s.start / fps, 3),
                 end_time=round(s.end / fps, 3))
            for s in segs
        ],
    }


def write_json(path: str, segs, version: int) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_json(segs, version), f, ensure_ascii=False, indent=2)
        f.write("\n")

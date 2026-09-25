"""Render SCAN ME.

    python render.py                 full film -> output/scan_me.mp4 (+ x-test copy)
    python render.py --preview       every 3rd frame at 540 px -> output/preview.mp4
    python render.py --frame 420     one frame -> output/frame_00420.png
    python render.py --start 14 --end 24 --preview   preview one act

Deterministic: the only randomness (paper grain) is seeded in config.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from dataclasses import asdict

import cv2
import numpy as np

import compose
import config
import qrlayer
import schedule
import scenes

HERE = os.path.dirname(os.path.abspath(__file__))


def out_path(*parts) -> str:
    base = config.OUTPUT_DIR if os.path.isabs(config.OUTPUT_DIR) else os.path.join(HERE, config.OUTPUT_DIR)
    return os.path.join(base, *parts)


def film_version(segs) -> int:
    return qrlayer.min_version(schedule.payloads(segs), config.EC_LEVEL)


# ------------------------------------------------------------- mask choice

def _segment_darkness(comp: compose.Compositor, seg, fps, step=3):
    """Mean picture darkness per module over a segment (0 paper .. 1 ink)."""
    acc = np.zeros((comp.lay.n, comp.lay.n), np.float64)
    frames = list(range(seg.start, seg.end, step)) or [seg.start]
    for f in frames:
        t = f / fps
        look = scenes.look(t)
        pic = comp.render_picture(scenes.draw, t, look) if look.picture > 0 else comp.blank_picture()
        acc += comp.cell_darkness(comp.colorize(pic))
    return acc / len(frames), frames


def choose_mask(comp: compose.Compositor, seg, fps, version):
    """Pick the mask whose modules best agree with the picture.

    Agreement (as specified): sum over data modules of
        (+1 for a dark module, -1 for a light one) * (darkness - 0.5)
    i.e. dark modules on ink and light modules on paper score. Among masks
    within 2% of the best agreement, the one the scanner models rate safest
    on a few sample frames wins (it needs the fewest safety fixes).
    """
    lay = comp.lay
    dark_avg, frames = _segment_darkness(comp, seg, fps)
    mats = qrlayer.all_mask_matrices(seg.payload, version, config.EC_LEVEL)
    data = lay.data
    agree = np.array([float(((2.0 * m - 1.0) * (dark_avg - 0.5))[data].sum()) for m in mats])
    spread = max(agree.max() - agree.min(), 1e-6)
    candidates = [m for m in range(8) if agree[m] >= agree.max() - 0.02 * spread - 1e-9]
    if len(candidates) == 1 and dark_avg.max() < 0.05:
        return candidates[0], agree.tolist(), {}
    samples = frames[:: max(1, len(frames) // 4)][:4]
    risk = {}
    for m in (candidates if len(candidates) > 1 else range(8)):
        score = 0
        for f in samples:
            t = f / fps
            look = scenes.look(t)
            pic = comp.render_picture(scenes.draw, t, look) if look.picture > 0 else comp.blank_picture()
            rgb, _ = comp.frame(pic, mats[m], m, look, safety=False)
            rep, _ = comp.check(rgb, mats[m])
            score += rep.risky_codewords + 2 * (rep.cv_block_errors + rep.zx_block_errors)
        risk[m] = score / len(samples)
    pool = candidates if len(candidates) > 1 else list(range(8))
    best = min(pool, key=lambda m: (risk[m], -agree[m]))
    return best, agree.tolist(), risk


def _choose_worker(args):
    seg_dict, fps, version = args
    comp = _worker_comp(version, config.SIZE_PX)
    seg = schedule.Segment(**seg_dict)
    mask, agree, risk = choose_mask(comp, seg, fps, version)
    return seg.index, mask, agree, risk


def select_masks(segs, version, jobs):
    fps = config.FPS
    tasks = [(asdict(s), fps, version) for s in segs]
    t0 = time.time()
    with mp.Pool(jobs) as pool:
        for idx, mask, agree, risk in pool.imap_unordered(_choose_worker, tasks):
            segs[idx].mask = mask
    print(f"masks chosen for {len(segs)} segments in {time.time() - t0:.0f}s: "
          + " ".join(str(s.mask) for s in segs))
    return segs


# ------------------------------------------------------------------ frames

_COMP = {}
_MATS = {}


def _worker_comp(version, size):
    key = (version, size)
    if key not in _COMP:
        _COMP[key] = compose.Compositor(size=size, version=version)
    return _COMP[key]


def render_frame(i, segs, version, size=None, safety=True):
    """Render film frame `i`; returns (rgb, SafetyReport, segment, act)."""
    size = size or config.SIZE_PX
    comp = _worker_comp(version, size)
    fps = config.FPS
    t = i / fps
    seg = schedule.segment_at(segs, i)
    key = (seg.payload, seg.mask)
    if key not in _MATS:
        _MATS[key] = qrlayer.matrix(seg.payload, version, config.EC_LEVEL, seg.mask)
    dark = _MATS[key]
    look = scenes.look(t)
    pic = comp.render_picture(scenes.draw, t, look) if look.picture > 0 else comp.blank_picture()
    rgb, rep = comp.frame(pic, dark, seg.mask, look, safety=safety)
    return rgb, rep, seg, scenes.act_at(t)


def _frame_worker(args):
    i, segs, version, size, safety, png_dir = args
    rgb, rep, seg, act = render_frame(i, segs, version, size, safety)
    if png_dir:
        cv2.imwrite(os.path.join(png_dir, f"frame_{i:05d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return i, rgb.tobytes(), rep, seg.index, act


def ffmpeg_writer(path, size, fps, crf):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{size}x{size}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def make_xtest(src, dst):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
           "-vf", f"scale={config.XTEST_SIZE}:{config.XTEST_SIZE}",
           "-c:v", "libx264", "-crf", str(config.XTEST_CRF), "-pix_fmt", "yuv420p", dst]
    subprocess.run(cmd, check=True)


def render_video(segs, version, frames, path, size, fps, jobs, safety=True, png_dir=None, log_path=None):
    if png_dir:
        os.makedirs(png_dir, exist_ok=True)
    proc = ffmpeg_writer(path, size, fps, config.VIDEO_CRF)
    tasks = [(i, segs, version, size, safety, png_dir) for i in frames]
    rows = []
    t0 = time.time()
    with mp.Pool(jobs) as pool:
        for n, (i, buf, rep, seg_idx, act) in enumerate(pool.imap(_frame_worker, tasks, chunksize=2)):
            proc.stdin.write(buf)
            seg = segs[seg_idx]
            rows.append(dict(frame=i, time=round(i / config.FPS, 3), act=act, segment=seg_idx,
                             payload=seg.payload, mask=seg.mask, **asdict(rep)))
            if (n + 1) % 60 == 0 or n + 1 == len(tasks):
                el = time.time() - t0
                print(f"  {n + 1}/{len(tasks)} frames  {el:.0f}s  ({el / (n + 1) * 1000:.0f} ms/frame)", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    if log_path and rows:
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


def load_schedule_with_masks(path=None):
    """Re-use masks from messages.json when the schedule is unchanged."""
    segs = schedule.build()
    path = path or out_path("messages.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        old = data.get("segments", [])
        if len(old) == len(segs) and all(o["payload"] == s.payload and o["start"] == s.start
                                         and o["end"] == s.end for o, s in zip(old, segs)) \
                and data.get("qr_version") == film_version(segs):
            for o, s in zip(old, segs):
                s.mask = o.get("mask", -1)
    return segs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview", action="store_true", help="every 3rd frame at 540 px, no safety net")
    ap.add_argument("--frame", type=int, help="render a single frame to PNG")
    ap.add_argument("--start", type=float, default=0.0, help="start time (s)")
    ap.add_argument("--end", type=float, default=None, help="end time (s)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--no-safety", action="store_true", help="skip the per-frame scanner safety net")
    ap.add_argument("--reselect-masks", action="store_true", help="ignore masks cached in messages.json")
    ap.add_argument("--no-png", action="store_true", help="do not keep lossless frames for verify.py")
    args = ap.parse_args(argv)

    os.makedirs(out_path(), exist_ok=True)
    segs = schedule.build() if args.reselect_masks else load_schedule_with_masks()
    version = film_version(segs)
    lay = qrlayer.layout(version, config.EC_LEVEL)
    print(f"QR version {version} ({lay.n}x{lay.n} modules), level {config.EC_LEVEL}, "
          f"{len(segs)} segments, {schedule.frame_count(segs)} frames")

    if any(s.mask < 0 for s in segs):
        select_masks(segs, version, args.jobs)
    schedule.write_json(out_path("messages.json"), segs, version)

    fps = config.FPS
    total = schedule.frame_count(segs)
    first = max(0, int(round(args.start * fps)))
    last = total if args.end is None else min(total, int(round(args.end * fps)))

    if args.frame is not None:
        rgb, rep, seg, act = render_frame(args.frame, segs, version, safety=not args.no_safety)
        p = out_path(f"frame_{args.frame:05d}.png")
        cv2.imwrite(p, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        import readers
        print(f"frame {args.frame} t={args.frame / fps:.2f}s act={act} payload={seg.payload!r} mask={seg.mask}")
        print(f"  {rep}")
        print(f"  zxing: {readers.zxing(rgb)!r}   opencv: {readers.opencv(rgb)!r}")
        print(f"  -> {p}")
        return 0

    if args.preview:
        size = config.SIZE_PX // 2
        frames = list(range(first, last, 3))
        path = out_path("preview.mp4")
        print(f"preview: {len(frames)} frames at {size}px -> {path}")
        render_video(segs, version, frames, path, size, fps / 3, args.jobs, safety=False)
        return 0

    frames = list(range(first, last))
    path = out_path("scan_me.mp4")
    png_dir = None if args.no_png else out_path("frames")
    print(f"rendering {len(frames)} frames -> {path}")
    rows = render_video(segs, version, frames, path, config.SIZE_PX, fps, args.jobs,
                        safety=not args.no_safety, png_dir=png_dir, log_path=out_path("render_log.csv"))
    fixed = sum(1 for r in rows if r["lifted"] or r["grown"])
    print(f"safety net touched {fixed}/{len(rows)} frames; "
          f"worst predicted codeword errors per block: opencv {max(r['cv_block_errors'] for r in rows)}, "
          f"zxing {max(r['zx_block_errors'] for r in rows)}")
    if first == 0 and last == total:
        xt = out_path("scan_me_x_test.mp4")
        make_xtest(path, xt)
        print(f"x-test copy -> {xt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

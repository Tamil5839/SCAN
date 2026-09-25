"""Verify SCAN ME: every frame must decode.

    python verify.py                 Pass A, B and C on the rendered film
    python verify.py --passes A      only some passes
    python verify.py --tune          find the smallest DOT_FRACTION that passes

Pass A  clean frames (lossless PNGs written by render.py): zxing-cpp AND
        OpenCV must both return exactly the expected payload.
Pass B  degraded copies of every clean frame (downscale 540, JPEG 40, blur
        1.5 px, rotate 4 deg, perspective, gamma 0.8 / 1.25): zxing-cpp must
        decode each one.
Pass C  every frame of scan_me.mp4 and of the 720 px CRF 28 x-test copy,
        decoded straight from the video: zxing-cpp must decode each one.
        OpenCV's result is recorded too.

Writes output/verification_report.csv (one row per frame).
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

import config
import readers
import render
import schedule
import scenes

DEG_NAMES = list(readers.DEGRADATIONS)


# ------------------------------------------------------------ pass A and B

def _ab_worker(args):
    i, path, expected, do_a, do_b, with_zbar = args
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:                      # a missing frame fails, it is never skipped
        row = {"error": f"missing {os.path.basename(path)}"}
        if do_a:
            row["A_pass"] = False
        if do_b:
            row["B_pass"] = False
        return i, row
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    row = {}
    if do_a:
        row["A_zxing"] = readers.zxing(img)
        row["A_opencv"] = readers.opencv(img)
        if with_zbar:
            z = readers.zbar(img)
            row["A_zbar"] = "" if z is None else z
        row["A_pass"] = row["A_zxing"] == expected and row["A_opencv"] == expected
    if do_b:
        ok = True
        for name, fn in readers.DEGRADATIONS.items():
            got = readers.zxing(fn(img))
            row[f"B_{name}_zxing"] = got
            ok &= got == expected
        row["B_pass"] = ok
    return i, row


# ------------------------------------------------------------------ pass C

def _c_worker(args):
    i, buf, size, with_opencv = args
    img = np.frombuffer(buf, np.uint8).reshape(size, size, 3)
    z = readers.zxing(img)
    o = readers.opencv(img) if with_opencv else None
    return i, z, o


def video_frames(path):
    """Yield (index, rgb bytes, size) for every frame of a square video."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", path],
                           capture_output=True, text=True, check=True)
    w, h = map(int, probe.stdout.strip().split(","))
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                            stdout=subprocess.PIPE)
    n = w * h * 3
    i = 0
    while True:
        buf = proc.stdout.read(n)
        if len(buf) < n:
            break
        yield i, buf, w
        i += 1
    proc.wait()


def pass_c(video, segs, jobs, with_opencv=True):
    results = {}
    tasks = ((i, buf, size, with_opencv) for i, buf, size in video_frames(video))
    with mp.Pool(jobs) as pool:
        for i, z, o in pool.imap_unordered(_c_worker, tasks, chunksize=4):
            results[i] = (z, o)
    return results


# ------------------------------------------------------------------ report

def run(passes="ABC", jobs=None, frames_dir=None, stride=1, videos=None, report_path=None, quiet=False):
    jobs = jobs or max(1, os.cpu_count() or 2)
    segs = render.load_schedule_with_masks()
    total = schedule.frame_count(segs)
    fps = config.FPS
    frames_dir = frames_dir or render.out_path("frames")
    frames = list(range(0, total, stride))
    rows = {i: {"frame": i, "time": round(i / fps, 3), "act": scenes.act_at(i / fps),
                "segment": schedule.segment_at(segs, i).index,
                "kind": schedule.segment_at(segs, i).kind,
                "expected": schedule.segment_at(segs, i).payload} for i in frames}
    with_zbar = readers.pyzbar is not None
    log = {}
    if os.path.exists(render.out_path("render_log.csv")):
        with open(render.out_path("render_log.csv")) as f:
            for r in csv.DictReader(f):
                log[int(r["frame"])] = r

    def say(*a):
        if not quiet:
            print(*a, flush=True)

    t0 = time.time()
    if "A" in passes or "B" in passes:
        tasks = [(i, os.path.join(frames_dir, f"frame_{i:05d}.png"), rows[i]["expected"],
                  "A" in passes, "B" in passes, with_zbar) for i in frames]
        with mp.Pool(jobs) as pool:
            for n, (i, row) in enumerate(pool.imap_unordered(_ab_worker, tasks, chunksize=2)):
                rows[i].update(row)
                if not quiet and (n + 1) % 300 == 0:
                    print(f"  A/B {n + 1}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
        say(f"pass A/B done in {time.time() - t0:.0f}s")

    if "C" in passes:
        videos = videos or {"main": render.out_path("scan_me.mp4"), "xtest": render.out_path("scan_me_x_test.mp4")}
        for tag, path in videos.items():
            t1 = time.time()
            res = pass_c(path, segs, jobs)
            n_frames = len(res)
            for i in frames:
                z, o = res.get(i, ("", ""))
                rows[i][f"C_{tag}_zxing"] = z
                rows[i][f"C_{tag}_opencv"] = o
                rows[i][f"C_{tag}_pass"] = z == rows[i]["expected"]
            say(f"pass C [{tag}] {n_frames} frames decoded from {os.path.basename(path)} in {time.time() - t1:.0f}s")
            if n_frames != total:
                say(f"  WARNING: video has {n_frames} frames, schedule has {total}")
        for i in frames:
            rows[i]["C_pass"] = all(rows[i].get(f"C_{t}_pass", False) for t in videos)

    for i in frames:
        if i in log:
            r = log[i]
            rows[i]["pred_opencv_block_errors"] = r["cv_block_errors"]
            rows[i]["pred_zxing_block_errors"] = r["zx_block_errors"]
            rows[i]["safety_lifted"] = r["lifted"]
            rows[i]["safety_grown"] = r["grown"]

    ordered = [rows[i] for i in frames]
    full = stride == 1 and set(passes) >= set("ABC")
    report_path = report_path or render.out_path("verification_report.csv" if full
                                                 else "verification_report_partial.csv")
    keys = []
    for r in ordered:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(ordered)
    return ordered, segs


def summarize(ordered, segs, passes):
    n = len(ordered)
    ok = True
    lines = []

    def rate(key):
        return sum(bool(r.get(key)) for r in ordered), n

    if "A" in passes:
        a, na = rate("A_pass")
        za = sum(r.get("A_zxing") == r["expected"] for r in ordered)
        oa = sum(r.get("A_opencv") == r["expected"] for r in ordered)
        lines.append(f"Pass A (clean frames)     {a}/{na}  [zxing {za}/{n}, opencv {oa}/{n}]")
        if "A_zbar" in ordered[0]:
            zb = sum(r.get("A_zbar") == r["expected"] for r in ordered)
            lines.append(f"  (info) pyzbar            {zb}/{n}")
        ok &= a == na
    if "B" in passes:
        b, nb = rate("B_pass")
        per = ", ".join(f"{d} {sum(r.get(f'B_{d}_zxing') == r['expected'] for r in ordered)}" for d in DEG_NAMES)
        lines.append(f"Pass B (degraded, zxing)  {b}/{nb}  [{per}]")
        ok &= b == nb
    if "C" in passes:
        c, nc = rate("C_pass")
        extra = []
        for tag in ("main", "xtest"):
            if f"C_{tag}_pass" in ordered[0]:
                zz = sum(bool(r[f"C_{tag}_pass"]) for r in ordered)
                oo = sum(r.get(f"C_{tag}_opencv") == r["expected"] for r in ordered)
                extra.append(f"{tag}: zxing {zz}/{n}, opencv {oo}/{n}")
        lines.append(f"Pass C (encoded video)    {c}/{nc}  [{'; '.join(extra)}]")
        ok &= c == nc
    fails = [r["frame"] for r in ordered if not all(r.get(k, True) for k in ("A_pass", "B_pass", "C_pass"))]
    if fails:
        lines.append(f"failing frames ({len(fails)}): {fails[:40]}{' ...' if len(fails) > 40 else ''}")
    return ok, lines


def reconstruction_lines(ordered, segs, stride=1):
    """What a viewer gets by scanning the film in order (from the main video if available)."""
    key = "C_main_zxing" if "C_main_zxing" in ordered[0] else "A_zxing"
    decoded = [r.get(key, "") for r in ordered]
    sentence = schedule.reconstruct_sentence(decoded)
    last = decoded[-1] if decoded else ""
    note = "" if stride == 1 else f"  (every {stride}th frame only: words may be missed)"
    return [
        f"hidden sentence rebuilt from decoded frames: {sentence!r}{note}",
        f"  matches SECRET_SENTENCE: {sentence == config.SECRET_SENTENCE}",
        f"final frame decodes to: {last!r}  (FINAL_URL match: {last == config.FINAL_URL})",
    ]


MANUAL_CHECKLIST = """
Manual test checklist (for a human):
  [ ] Play output/scan_me.mp4 full screen on a laptop. Pause at 10 random
      moments and scan each with an iPhone camera and an Android camera (or
      Google Lens). Plain-text payloads show as text (some camera apps put
      it in a search/copy sheet rather than a link banner) - all should
      still show the text. The last 5 seconds should offer to open
      the FINAL_URL.
  [ ] Scan once during each act: title, flower, bird, sea, city, eye, link.
  [ ] Collect the words in order ("1/8 - YOU", "2/8 - FOUND", ...) and check
      the sentence reads correctly.
  [ ] Upload a private test copy to X (or another social network), then scan
      from the phone while it plays in the feed, and again while paused.
  [ ] Watch the film once without scanning: scenes readable at phone size,
      no flashing beyond the gentle dot shimmer at word changes.
"""


# -------------------------------------------------------------------- tune

def tune(candidates, stride, jobs):
    """Render every `stride`-th frame at each DOT_FRACTION (safety net may
    lift the picture but may not enlarge dots) and run passes A, B and a
    compressed-video check; report the smallest value that passes 100%."""
    results = []
    for f in candidates:
        config.DOT_FRACTION = f
        config.SAFETY_GROW_DOTS = False
        tmp = tempfile.mkdtemp(prefix=f"tune_{f:.2f}_", dir=render.out_path())
        segs = render.load_schedule_with_masks()
        version = render.film_version(segs)
        total = schedule.frame_count(segs)
        frames = list(range(0, total, stride))
        t0 = time.time()
        video = os.path.join(tmp, "tune.mp4")
        render.render_video(segs, version, frames, video, config.SIZE_PX, config.FPS, jobs,
                            safety=True, png_dir=tmp)
        xt = os.path.join(tmp, "tune_x_test.mp4")
        render.make_xtest(video, xt)
        ordered, _ = run("AB", jobs, frames_dir=tmp, stride=stride, quiet=True,
                         report_path=os.path.join(tmp, "report.csv"))
        ok_a = sum(bool(r.get("A_pass")) for r in ordered)
        ok_b = sum(bool(r.get("B_pass")) for r in ordered)
        ok_c = 0
        for path in (video, xt):                 # video frame k is film frame frames[k]
            dec = {}
            with mp.Pool(jobs) as pool:
                tasks = ((k, buf, size, False) for k, buf, size in video_frames(path))
                for k, z, _ in pool.imap_unordered(_c_worker, tasks, chunksize=4):
                    dec[k] = z
            ok_c += sum(dec.get(k) == schedule.segment_at(segs, fr).payload for k, fr in enumerate(frames))
        shutil.rmtree(tmp, ignore_errors=True)
        n = len(frames)
        passed = ok_a == n and ok_b == n and ok_c == 2 * n
        results.append((f, passed, ok_a, ok_b, ok_c, n))
        print(f"DOT_FRACTION={f:.2f}: A {ok_a}/{n}  B {ok_b}/{n}  C {ok_c}/{2 * n}  "
              f"-> {'PASS' if passed else 'fail'}  ({time.time() - t0:.0f}s)", flush=True)
    passing = [r[0] for r in results if r[1]]
    if passing:
        print(f"smallest DOT_FRACTION passing 100% (every {stride}th frame): {min(passing):.2f}")
    else:
        print("no candidate passed")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--passes", default="ABC")
    ap.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 2))
    ap.add_argument("--stride", type=int, default=1, help="check every n-th frame (A/B)")
    ap.add_argument("--tune", action="store_true", help="search the smallest passing DOT_FRACTION")
    ap.add_argument("--candidates", default="0.52,0.54,0.56,0.58,0.60")
    args = ap.parse_args(argv)

    if args.tune:
        tune([float(x) for x in args.candidates.split(",")], max(args.stride, 5), args.jobs)
        return 0

    passes = args.passes.upper()
    t0 = time.time()
    ordered, segs = run(passes, args.jobs, stride=args.stride)
    ok, lines = summarize(ordered, segs, passes)
    print()
    print("\n".join(lines))
    print("\n".join(reconstruction_lines(ordered, segs, args.stride)))
    name = "verification_report.csv" if args.stride == 1 and set(passes) >= set("ABC") \
        else "verification_report_partial.csv"
    print(f"report -> {render.out_path(name)}  ({time.time() - t0:.0f}s)")
    print("ALL PASSES 100%" if ok else "NOT FINISHED: some frames fail")
    print(MANUAL_CHECKLIST)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

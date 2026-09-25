"""Fast models of how the two reference decoders read a frame.

Both models use the *true* module geometry (we know exactly where every
module was drawn), so they predict a best-case read of a clean frame. They
are used to steer rendering and to explain failures; the real decoders in
verify.py remain the judge.

OpenCV (QRCodeDetector, objdetect/src/qrcode.cpp)
    adaptiveThreshold(gray, GAUSSIAN_C, blockSize=83, C=2), then each module
    is the white-pixel *fraction of its whole cell*, compared against the
    white fraction of the whole symbol. Area based: a small centre dot is
    outvoted by whatever surrounds it.

zxing-cpp (HybridBinarizer / LocalAverage, "new algorithm")
    Per 8x8 block threshold = (max+min)/2 when max-min > 24, smoothed over
    5x5 blocks; each module is the single pixel at its centre. Centre based.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Geometry:
    """Where the symbol sits in a square frame."""

    size: int          # frame side in px
    n: int             # modules per side (no quiet zone)
    module: int        # module side in px
    origin: int        # px coordinate of module (0, 0)'s top-left corner

    @classmethod
    def fit(cls, size: int, n: int, quiet: int = 4) -> "Geometry":
        total = n + 2 * quiet
        module = size // total
        offset = (size - module * total) // 2
        return cls(size=size, n=n, module=module, origin=offset + quiet * module)

    @property
    def extent(self) -> int:
        return self.n * self.module

    def centers(self) -> np.ndarray:
        return self.origin + np.arange(self.n) * self.module + self.module // 2


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


# --------------------------------------------------------------------- OpenCV

def opencv_binary(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 83, 2)


def opencv_cells(gray: np.ndarray, geo: Geometry):
    """Return (white fraction per module, global white fraction)."""
    b = opencv_binary(gray)
    o, e, n, m = geo.origin, geo.extent, geo.n, geo.module
    area = b[o:o + e, o:o + e].astype(np.float32) * (1.0 / 255.0)
    w = area.reshape(n, m, n, m).mean(axis=(1, 3))
    return w, float(area.mean())


def opencv_read(gray: np.ndarray, geo: Geometry):
    """Predicted dark/light read and a signed margin (>0 = reads light)."""
    w, wbar = opencv_cells(gray, geo)
    return w < wbar, w - wbar


# ---------------------------------------------------------------------- zxing

_B = 8          # block size
_R = 2          # smoothing radius in blocks (5x5 window)
_MIN_RANGE = 24


def _box5(a: np.ndarray) -> np.ndarray:
    """Sum over 5x5 windows centred at [2:-2, 2:-2] (valid region only)."""
    c = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    k = 2 * _R + 1
    return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]


def zxing_threshold_map(gray: np.ndarray) -> np.ndarray:
    """Per-pixel threshold of zxing-cpp's LocalAverage binarizer (black if <=)."""
    h, w = gray.shape
    sh, sw = (h + _B - 1) // _B, (w + _B - 1) // _B
    ys = np.minimum(np.arange(sh) * _B, h - _B)
    xs = np.minimum(np.arange(sw) * _B, w - _B)
    blocks = gray[ys[:, None, None, None] + np.arange(_B)[None, None, :, None],
                  xs[None, :, None, None] + np.arange(_B)[None, None, None, :]].astype(np.int32)
    mn = blocks.min(axis=(2, 3))
    mx = blocks.max(axis=(2, 3))
    thr = np.where(mx - mn > _MIN_RANGE, (mx + mn) // 2, 0)

    s = _box5(thr)
    cnt = _box5((thr > 0).astype(np.int32))
    iy = np.clip(np.arange(sh), _R, sh - _R - 1) - _R
    ix = np.clip(np.arange(sw), _R, sw - _R - 1) - _R
    total = thr * 2 + s[np.ix_(iy, ix)]
    num = (thr > 0) * 2 + cnt[np.ix_(iy, ix)]
    out = np.where(num > 0, total // np.maximum(num, 1), 0).ravel()

    # zero gaps take the next non-zero value (row-major), trailing zeros the last
    nz = np.flatnonzero(out)
    if nz.size:
        nxt = np.searchsorted(nz, np.arange(out.size))
        nxt = np.minimum(nxt, nz.size - 1)
        out = out[nz[nxt]]
    out = out.reshape(sh, sw)
    # each block thresholds its 8x8 pixels; when the size is not a multiple
    # of 8 the last block is shifted inward and overwrites the overlap
    py, px = np.arange(h), np.arange(w)
    by = np.where(py >= h - _B, sh - 1, py // _B)
    bx = np.where(px >= w - _B, sw - 1, px // _B)
    return out[np.ix_(by, bx)]


def zxing_read(gray: np.ndarray, geo: Geometry, jitter: int = 0):
    """Predicted dark/light read at the exact module centres, the signed
    margin there (grey levels, >0 = light), and the worst-case margin over a
    (2*jitter+1)^2 px window modelling imprecise grid sampling.
    """
    thr = zxing_threshold_map(gray)
    diff = gray.astype(np.int32) - thr
    c = geo.centers()
    centre = diff[np.ix_(c, c)]
    read_dark = centre <= 0
    if jitter <= 0:
        return read_dark, centre, centre
    lo = hi = centre
    for dy in range(-jitter, jitter + 1):
        for dx in range(-jitter, jitter + 1):
            v = diff[np.ix_(c + dy, c + dx)]
            lo = np.minimum(lo, v)
            hi = np.maximum(hi, v)
    worst = np.where(read_dark, hi, lo)
    return read_dark, centre, worst

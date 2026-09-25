"""Real decoders and the degradations used by verify.py.

Decoders take an RGB uint8 image and return the decoded text, or "" when
nothing (or no QR code) was read.
"""

from __future__ import annotations

import cv2
import numpy as np
import zxingcpp

try:  # optional third decoder
    from pyzbar import pyzbar
except Exception:  # pragma: no cover - libzbar missing
    pyzbar = None

_QR = zxingcpp.BarcodeFormat.QRCode


def zxing(img: np.ndarray) -> str:
    res = zxingcpp.read_barcodes(img, formats=_QR)
    return res[0].text if res else ""


_cv_detector = None


def opencv(img: np.ndarray) -> str:
    global _cv_detector
    if _cv_detector is None:
        _cv_detector = cv2.QRCodeDetector()
    try:
        text, _, _ = _cv_detector.detectAndDecode(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    except cv2.error:
        return ""
    return text or ""


def zbar(img: np.ndarray) -> str | None:
    """None when pyzbar/libzbar is unavailable."""
    if pyzbar is None:
        return None
    res = pyzbar.decode(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), symbols=[pyzbar.ZBarSymbol.QRCODE])
    return res[0].data.decode("utf-8", "replace") if res else ""


DECODERS = {"zxing": zxing, "opencv": opencv}


# ---------------------------------------------------------------- degradations

def _paper_bgr(img):
    c = img[2, 2]
    return (int(c[0]), int(c[1]), int(c[2]))


def downscale_540(img):
    h, w = img.shape[:2]
    small = cv2.resize(img, (540, 540), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def jpeg_40(img):
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 40])
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def blur_1_5(img):
    return cv2.GaussianBlur(img, (0, 0), 1.5)


def rotate_4(img):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 4.0, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=_paper_bgr(img))


def perspective(img):
    """A phone held slightly below and to the left of the screen."""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w * 0.07, h * 0.05], [w * 0.95, h * 0.01], [w * 0.99, h * 0.99], [w * 0.02, h * 0.95]])
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=_paper_bgr(img))


def _gamma(img, g):
    lut = (np.power(np.arange(256) / 255.0, g) * 255.0 + 0.5).astype(np.uint8)
    return lut[img]


def gamma_0_8(img):
    return _gamma(img, 0.8)


def gamma_1_25(img):
    return _gamma(img, 1.25)


DEGRADATIONS = {
    "down540": downscale_540,
    "jpeg40": jpeg_40,
    "blur1.5": blur_1_5,
    "rot4": rotate_4,
    "persp": perspective,
    "gamma0.8": gamma_0_8,
    "gamma1.25": gamma_1_25,
}

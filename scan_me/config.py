"""SCAN ME - every value here is meant to be edited.

Content, timing and look of the film. Change the text, re-run
`python render.py` and then `python verify.py`; the QR version, the payload
schedule and the mask choice are all recomputed from these values.
"""

# ----------------------------------------------------------------- content

SECRET_SENTENCE = "YOU FOUND THE MESSAGE HIDDEN INSIDE THIS FILM"
FINAL_URL = "https://example.com"   # user replaces this
TITLE_TEXT = "Pause anywhere. Scan me."
WORD_HOLD_SEC = 1.5    # each word's payload stays for this long
EC_LEVEL = "H"         # highest error correction (~30%)
FPS = 30
SIZE_PX = 1080

# How a word payload is written: {i} = position, {n} = number of words.
# ASCII on purpose: a middle dot "·" decodes fine in zxing and OpenCV, but
# zbar-based scanners show it as "繚" (it guesses Shift_JIS). Plain ASCII
# also lets the words use the compact alphanumeric QR mode.
WORD_TEMPLATE = "{i}/{n} - {word}"

# Short fun payloads, each held for WORD_HOLD_SEC at (roughly) the given
# second. Times snap to the word grid; they sit on the scene transitions.
BONUS_MESSAGES = [
    ("You paused at the right moment.", 13.0),   # petal becomes a bird
    ("This bird is made of data.", 23.5),        # bird dives into the sea
    ("The eye is watching you scan.", 44.5),     # moon becomes an eye
    ("Scan the last frame.", 50.5),
]

# ------------------------------------------------------------------ timing

DURATION_SEC = 60.0
TITLE_END_SEC = 4.0     # 0:00-0:04 title, payload TITLE_TEXT
LINK_START_SEC = 55.0   # 0:55-1:00 clean code, payload FINAL_URL

# ------------------------------------------------------------------- look

PAPER = "#F4F1EA"
INK = "#15151A"
ACCENT = "#B8342F"      # picture layer only, never on code dots
DOT_DARK = "#111114"
DOT_LIGHT = "#F7F5EF"

QUIET_ZONE = 4          # modules of clean paper around the code

# Dot side as a fraction of the module side.
#   DOT_FRACTION      smallest dot: a dark module over light paper, i.e. the
#                     dot that fights the picture. Tuned by `verify.py --tune`.
#   DOT_FRACTION_MAX  a dark module over dark ink may grow up to this.
#   CLEAN_DOT_FRACTION  dots of the picture-free title and link code.
DOT_FRACTION = 0.62
DOT_FRACTION_MAX = 0.70
CLEAN_DOT_FRACTION = 0.86
DOT_CORNER = 0.28       # corner radius as a fraction of the dot side
FINDER_CORNER = 0.22    # outer corner rounding of finder patterns (modules)

# Picture layer. Scanners binarise against the local surroundings, so the
# picture is a soft ink wash: its darkest tone stays well above the code's
# dark dots and its edges are soft. The sharp dot screen supplies the crisp
# detail.
PICTURE_INK_MAX = 0.76      # 1.0 would be solid INK; 0.76 ~ luma 78
PICTURE_SOFTNESS_PX = 11.0  # gaussian sigma of the picture, at 1080 px
# Picture fades to paper near the quiet zone and near light function
# modules (finder separators, timing, format). Distances in modules.
SAFE_FADE_START = 0.35
SAFE_FADE_END = 1.6
BORDER_FADE_START = 0.3   # fade toward the quiet zone, modules
BORDER_FADE_END = 3.0

PAPER_GRAIN = 1.0       # grain amplitude in grey levels (keep tiny)
SEED = 20260925         # every random choice is seeded

# ------------------------------------------------------- scanner safety net
# Each frame is checked against models of both reference decoders
# (scanmodel.py). Modules predicted to read wrong, or with thin margins,
# get their dot enlarged or the picture lifted around them, and the frame
# is re-rendered. Margins are what is left for blur, JPEG, video coding.
SAFETY_ITERATIONS = 4
MARGIN_OPENCV = 0.04        # white-fraction margin per module
MARGIN_ZXING = 16           # grey-level margin at the module centre (2 px jitter)
# Level H corrects 8 wrong codewords per Reed-Solomon block. Per block, the
# net only repairs the worst codewords until at most SAFE_MAX_ERRORS are
# predicted wrong and at most SAFE_MAX_RISKY are wrong or thin; the rest of
# the budget is left for blur, JPEG and video coding.
SAFE_MAX_ERRORS = 2
SAFE_MAX_RISKY = 4
SAFETY_GROW_DOTS = True     # may enlarge a thin dark dot (off while tuning DOT_FRACTION)

# ------------------------------------------------------------------ output

OUTPUT_DIR = "output"
VIDEO_CRF = 12
XTEST_SIZE = 720
XTEST_CRF = 28

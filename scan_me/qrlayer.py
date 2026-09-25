"""QR layer: one fixed version for the whole film, level-H matrices with an
explicit mask, and the module maps the compositor needs.

Everything here is pure data: boolean module matrices, which modules are
function patterns, and which Reed-Solomon block every data module belongs to
(used to count codeword errors per block when predicting how a scanner will
read a frame).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import segno
from segno import consts

_EC = {
    "L": consts.ERROR_LEVEL_L,
    "M": consts.ERROR_LEVEL_M,
    "Q": consts.ERROR_LEVEL_Q,
    "H": consts.ERROR_LEVEL_H,
}

# segno's verbose module types that carry data (everything else is a
# function pattern: finders, separators, timing, alignment, format, version
# and the dark module).
_DATA_TYPES = (consts.TYPE_DATA_DARK, consts.TYPE_DATA_LIGHT)


def make_code(payload: str, version: int, ec: str, mask: int) -> segno.QRCode:
    """Encode `payload` with a forced version, error level and mask.

    Text is encoded as UTF-8 without ECI: every mainstream reader decodes
    UTF-8 byte mode correctly, while the QR default (ISO-8859-1) makes
    zxing-family readers guess Shift_JIS for a lone high byte.
    """
    return segno.make(payload, error=ec.lower(), version=version, mask=mask,
                      encoding="utf-8", micro=False, boost_error=False)


def min_version(payloads, ec: str = "H") -> int:
    """Smallest version that fits every payload at error level `ec`."""
    best = 1
    for p in payloads:
        q = segno.make(p, error=ec.lower(), encoding="utf-8", micro=False, boost_error=False)
        best = max(best, q.version)
    return best


def matrix(payload: str, version: int, ec: str, mask: int) -> np.ndarray:
    """N x N bool array, True = dark module (no quiet zone)."""
    code = make_code(payload, version, ec, mask)
    return np.array([[bool(v) for v in row] for row in code.matrix], dtype=bool)


def all_mask_matrices(payload: str, version: int, ec: str) -> np.ndarray:
    """(8, N, N) bool array, one matrix per mask pattern."""
    return np.stack([matrix(payload, version, ec, m) for m in range(8)])


@dataclass(frozen=True)
class Layout:
    """Geometry of the fixed-version symbol that does not depend on payload."""

    version: int
    ec: str
    n: int                      # modules per side
    function: np.ndarray        # (N, N) bool, True = function pattern module
    block: np.ndarray           # (N, N) int, RS block index of data modules, -1 otherwise
    codeword: np.ndarray        # (N, N) int, interleaved codeword index, -1 otherwise
    n_blocks: int
    ec_per_block: tuple         # EC codewords per block
    correctable: tuple          # max codeword errors correctable per block

    @property
    def data(self) -> np.ndarray:
        return ~self.function


def _placement_order(function: np.ndarray):
    """Module coordinates in the standard zig-zag data placement order."""
    n = function.shape[0]
    order = []
    col = n - 1
    upward = True
    while col > 0:
        if col == 6:
            col -= 1
        rows = range(n - 1, -1, -1) if upward else range(n)
        for r in rows:
            for c in (col, col - 1):
                if not function[r, c]:
                    order.append((r, c))
        upward = not upward
        col -= 2
    return order


def _block_structure(version: int, ec: str):
    """List of (data_len, ec_len) per block in standard block order."""
    blocks = []
    for info in consts.ECC[version][_EC[ec]]:
        for _ in range(info.num_blocks):
            blocks.append((info.num_data, info.num_total - info.num_data))
    return blocks


def _interleaved_owner(version: int, ec: str):
    """For each interleaved codeword index -> (block index, is_ec, index in block)."""
    blocks = _block_structure(version, ec)
    owners = []
    max_data = max(d for d, _ in blocks)
    for i in range(max_data):
        for b, (d, _) in enumerate(blocks):
            if i < d:
                owners.append((b, False, i))
    ec_len = blocks[0][1]
    for i in range(ec_len):
        for b in range(len(blocks)):
            owners.append((b, True, i))
    return owners


def function_map(version: int) -> np.ndarray:
    """Function-pattern modules per ISO/IEC 18004, computed from first principles.

    (segno's verbose module labels tag one data module as format info, so the
    map is derived from the standard rather than read back from segno.)
    """
    n = 17 + 4 * version
    f = np.zeros((n, n), dtype=bool)
    # finder patterns + separators
    f[0:8, 0:8] = f[0:8, n - 8:n] = f[n - 8:n, 0:8] = True
    # format information (+ the dark module at (n-8, 8))
    f[8, 0:9] = f[0:9, 8] = True
    f[8, n - 8:n] = True
    f[n - 8:n, 8] = True
    # timing patterns
    f[6, :] = f[:, 6] = True
    # alignment patterns
    pos = consts.ALIGNMENT_POS[version - 2] if version > 1 else ()  # table starts at version 2
    corners = {(pos[0], pos[0]), (pos[0], pos[-1]), (pos[-1], pos[0])} if pos else set()
    for r in pos:
        for c in pos:
            if (r, c) in corners:    # would overlap a finder pattern: not placed
                continue
            f[r - 2:r + 3, c - 2:c + 3] = True
    # version information
    if version >= 7:
        f[0:6, n - 11:n - 8] = f[n - 11:n - 8, 0:6] = True
    return f


@lru_cache(maxsize=None)
def layout(version: int, ec: str = "H") -> Layout:
    function = function_map(version)
    n = function.shape[0]
    owners = _interleaved_owner(version, ec)
    block = np.full((n, n), -1, dtype=np.int16)
    codeword = np.full((n, n), -1, dtype=np.int16)
    for k, (r, c) in enumerate(_placement_order(function)):
        cw = k // 8
        if cw < len(owners):          # the rest are remainder bits
            codeword[r, c] = cw
            block[r, c] = owners[cw][0]
    structure = _block_structure(version, ec)
    ec_per_block = tuple(e for _, e in structure)
    return Layout(version=version, ec=ec, n=n, function=function, block=block,
                  codeword=codeword, n_blocks=len(structure), ec_per_block=ec_per_block,
                  correctable=tuple(e // 2 for e in ec_per_block))


def read_codewords(dark: np.ndarray, mask: int, lay: Layout):
    """Recover the de-interleaved (data, ec) byte blocks from a matrix.

    Used by the tests to prove the module->codeword map is exact.
    """
    n = lay.n
    r, c = np.mgrid[0:n, 0:n]
    patt = MASK_FUNCTIONS[mask](r, c)
    bits = dark ^ (patt & lay.data)
    order = _placement_order(lay.function)
    owners = _interleaved_owner(lay.version, lay.ec)
    stream = [int(bits[rc]) for rc in order][: 8 * len(owners)]
    words = [int("".join(map(str, stream[i:i + 8])), 2) for i in range(0, len(stream), 8)]
    structure = _block_structure(lay.version, lay.ec)
    data = [[0] * d for d, _ in structure]
    ecw = [[0] * e for _, e in structure]
    for w, (b, is_ec, i) in zip(words, owners):
        (ecw if is_ec else data)[b][i] = w
    return data, ecw


MASK_FUNCTIONS = (
    lambda i, j: (i + j) % 2 == 0,
    lambda i, j: i % 2 == 0,
    lambda i, j: j % 3 == 0,
    lambda i, j: (i + j) % 3 == 0,
    lambda i, j: (i // 2 + j // 3) % 2 == 0,
    lambda i, j: (i * j) % 2 + (i * j) % 3 == 0,
    lambda i, j: ((i * j) % 2 + (i * j) % 3) % 2 == 0,
    lambda i, j: ((i + j) % 2 + (i * j) % 3) % 2 == 0,
)


def codeword_errors(read_dark: np.ndarray, true_dark: np.ndarray, lay: Layout) -> np.ndarray:
    """Number of distinct wrong codewords per RS block, given a predicted read."""
    wrong = (read_dark != true_dark) & (lay.codeword >= 0)
    errs = np.zeros(lay.n_blocks, dtype=int)
    for b in range(lay.n_blocks):
        errs[b] = np.unique(lay.codeword[wrong & (lay.block == b)]).size
    return errs

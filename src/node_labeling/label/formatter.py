from typing import List, Tuple
from functools import lru_cache

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}"
})

from node_labeling.utils.config import Config
from node_labeling.utils.constants import *
from node_labeling.models.label_type import LabelType

def measure_ink_mm(text: str) -> Tuple[float, float]:
    '''
    Measure the MBR of the formatted label.

    Parameters
    ----------
    text : str
        formatted LaTeX label

    Returns
    -------
    width, height : Tuple[float, float]
        size of the ink MBR
    '''
    fig = plt.figure(dpi=DPI)
    ax = fig.add_subplot(111)
    t = ax.text(0.5, 0.5, text, usetex=True)
    
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    
    px_per_mm = DPI / 25.4
    bbox = t.get_tightbbox(renderer)
    w_mm = bbox.width / px_per_mm
    h_mm = bbox.height / px_per_mm
    plt.close(fig)

    return w_mm, h_mm

def format_label_text(cfg: Config, plain_text: str, label_type: LabelType) -> str:
    '''
    Format plain_text into at most k_rows rows and return a LaTeX string.
 
    Parameters
    ----------
    cfg : Config
        configuration
    plain_text : str
        raw label text
    label_type : LabelType
        general, extent, or intent

    Returns
    -------
    formatted_label : str
        LaTeX formatted plain text
    '''
    if not plain_text:
        return plain_text
 
    words = _split_words(plain_text)
    n = len(words)
 
    if cfg.k_rows == 1 or len(plain_text) <= cfg.max_row_chars or n <= 1:
        return _format_row(cfg, str(plain_text), label_type)
 
    splits = _best_splits_cached(tuple(len(w) for w in words), cfg.k_rows)
 
    rows = []
    prev = 0
    for s in splits:
        rows.append(str(' '.join(words[prev:s])))
        prev = s
    rows.append(str(' '.join(words[prev:])))
 
    formatted = [_format_row(cfg, r, label_type) for r in rows]
    joined    = r' \\[-1pt] '.join(formatted)
    return rf'\begin{{tabular}}{{@{{}}c@{{}}}}{joined}\end{{tabular}}'

def _format_row(cfg: Config, row: str, label_type: LabelType) -> str:
    '''
    Format row based on label type.

    Parameters
    ----------
    cfg : Config
        configuration
    row : str
        plain text of a single row
    label_type : LabelType
        general, extent, or intent

    Returns
    -------
    formatted_row : str
        LaTeX formatted plain text of a single row
    '''
    parts  = MATH_RE.split(row)
    result = []
    for part in parts:
        if part.startswith('$') and part.endswith('$'):
            result.append(part)          
        elif part.strip():
            result.append(rf'{label_type.latex}{{{part}}}')
    return rf'{{{cfg.font_size} {"".join(result)}}}'

def _split_words(text: str) -> List[str]:
    '''
    Split into words respecting LaTeX/math boundaries.

    Parameters
    ----------
    text : str
        plain text

    Returns
    -------
    tokens : List[str]
        tokens corresponding to words
    '''
    if not any(c in text for c in LATEX_CHARS):
        return text.split()
 
    tokens: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
 
        if c in ' \t\n':
            i += 1
            continue
 
        if c == '$':
            j = text.find('$', i + 1)
            if j == -1:
                j = n - 1
            tokens.append(text[i : j + 1])
            i = j + 1
            continue
 
        if c == '\\':
            j = i + 1
            while j < n and text[j].isalpha():
                j += 1
            if j == i + 1 and j < n:
                j += 1
            if j < n and text[j] == '{':
                depth, k = 1, j + 1
                while k < n and depth:
                    if   text[k] == '{': depth += 1
                    elif text[k] == '}': depth -= 1
                    k += 1
                j = k
            tokens.append(text[i : j])
            i = j
            continue
 
        if c == '{':
            depth, j = 1, i + 1
            while j < n and depth:
                if   text[j] == '{': depth += 1
                elif text[j] == '}': depth -= 1
                j += 1
            tokens.append(text[i : j])
            i = j
            continue
 
        j = i
        while j < n and text[j] not in ' \t\n\\{$':
            j += 1
        tokens.append(text[i : j])
        i = j
 
    return tokens or text.split()

@lru_cache(maxsize=4096)
def _best_splits_cached(word_lens: Tuple[int, ...], k: int) -> Tuple[int, ...]:
    '''
    Cached wrapper around the DP.

    Parameters
    ----------
    word_lens : Tuple[int, ...]
        number of chars in each word

    Returns
    -------
    splits: Tuple[int]
        split indices
    '''
    return tuple(_dp_splits(word_lens, k))

def _dp_splits(word_lens: Tuple[int, ...], k: int) -> List[int]:
    '''
    DP partitioning: minimize sum of squared row lengths to minimize variance.
    Complexity: O(k * n^2)

    Parameters
    ----------
    word_lens : Tuple[int, ...]
        number of chars in each word

    Returns
    -------
    splits: List[int]
        split indices
    '''
    n = len(word_lens)
    k = min(k, n)
 
    cum = [0] * (n + 1)
    for i, wl in enumerate(word_lens):
        cum[i + 1] = cum[i] + wl + (1 if i else 0)
 
    seg_sq = [[0] * (n + 1) for _ in range(n)]
    for i in range(n):
        base = cum[i]
        off = 1 if i else 0
        for j in range(i + 1, n + 1):
            raw = cum[j] - base - off
            seg_sq[i][j] = raw * raw
 
    INF = float('inf')
    stride = n + 1
    dp = [INF] * ((k + 1) * stride)
    par = [-1] * ((k + 1) * stride)
    dp[0] = 0.0
 
    for r in range(1, k + 1):
        r_off = r * stride
        r1_off = (r - 1) * stride
        sq_r = seg_sq
        for j in range(r, n + 1):
            best = INF
            bi = -1
            for i in range(r - 1, j):
                prev = dp[r1_off + i]
                if prev == INF:
                    continue
                cost = prev + sq_r[i][j]
                if cost < best:
                    best = cost
                    bi = i
            dp[r_off + j] = best
            par[r_off + j] = bi
 
    best_r = k
    best_cost = dp[k * stride + n]
    for r in range(1, k):
        c = dp[r * stride + n]
        if c < best_cost:
            best_cost = c
            best_r = r
 
    splits: list[int] = [0] * (best_r - 1)
    j = n
    for r in range(best_r, 1, -1):
        i = par[r * stride + j]
        splits[r - 2] = i
        j = i
 
    return splits
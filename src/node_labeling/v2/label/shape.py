from functools import lru_cache
from typing import Tuple

import matplotlib
from matplotlib import dviread
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import get_font
from matplotlib.ft2font import LoadFlags
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.textpath import TextToPath
from shapely import affinity, box, make_valid, unary_union
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

import node_labeling.label.formatter
from node_labeling.utils.constants import SHAPE_TOLERANCE_MM

PT_TO_MM: float = 25.4 / 72.0
GLYPH_SCALE: float = TextToPath.FONT_SCALE

def measure_label_shape(tex: str) -> BaseGeometry:
    '''
    Measure the outline of the letters of a formatted label.

    The outlines come from the same TeX output that is drawn: the dvi is
    compiled at the text size matplotlib draws with, every glyph is loaded
    from its font and placed at its dvi position. Holes (e, o, ...) are filled,
    and the union is simplified by `SHAPE_TOLERANCE_MM`.

    Parameters
    ----------
    tex : str
        formatted LaTeX label

    Returns
    -------
    ink : BaseGeometry
        Polygon or MultiPolygon in mm, relative to the point where the text is
        drawn with `ha='center', va='center'`
    '''
    return _measure(tex, float(matplotlib.rcParams['font.size']))

@lru_cache(maxsize=None)
def _measure(tex: str, fontsize: float) -> BaseGeometry:
    '''
    Cached body of `measure_label_shape`.

    Parameters
    ----------
    tex : str
        formatted LaTeX label
    fontsize : float
        size of the matplotlib text in pt

    Returns
    -------
    ink : BaseGeometry
        glyph union in mm, relative to the drawing point
    '''
    ox, oy = _baseline_origin(tex, fontsize)
    with dviread.Dvi(TexManager.make_dvi(tex, fontsize), 72) as dvi:
        page, = dvi

    parts = []
    for text in page.text:
        if text.font.subfont:
            raise NotImplementedError('Indexing TTC fonts is not supported')
        glyph = _glyph(text.font.resolve_path(), text.index)
        if glyph.is_empty:
            continue
        s = text.font_size / GLYPH_SCALE * PT_TO_MM
        parts.append(affinity.affine_transform(
            glyph, [s, 0, 0, s, (text.x + ox) * PT_TO_MM, (text.y + oy) * PT_TO_MM]
        ))
    for x, y, h, w in page.boxes:
        parts.append(box((x + ox) * PT_TO_MM, (y + oy) * PT_TO_MM, (x + ox + w) * PT_TO_MM, (y + oy + h) * PT_TO_MM))

    if not parts:
        return Polygon()
    ink = _fill_holes(unary_union(parts))
    return _fill_holes(ink.simplify(SHAPE_TOLERANCE_MM, preserve_topology=True))

@lru_cache(maxsize=None)
def _glyph(font_path: str, index: int) -> BaseGeometry:
    '''
    Filled outline of one glyph, unhinted, at `GLYPH_SCALE` pt.

    Parameters
    ----------
    font_path : str
        font file
    index : int
        FreeType glyph index

    Returns
    -------
    glyph : BaseGeometry
        union of all contours (so holes are filled), origin at the glyph origin
    '''
    font = get_font(font_path)
    font.clear()
    font.set_size(GLYPH_SCALE, 72)
    font.load_glyph(index, flags=LoadFlags.NO_HINTING)
    verts, codes = font.get_path()
    contours = Path(verts, codes).to_polygons()
    return unary_union([make_valid(Polygon(c)) for c in contours if len(c) >= 3])

def _baseline_origin(tex: str, fontsize: float) -> Tuple[float, float]:
    '''
    Offset of the TeX baseline origin from the drawing point, as matplotlib
    lays out a text with `ha='center', va='center'`.

    Parameters
    ----------
    tex : str
        formatted LaTeX label
    fontsize : float
        size of the matplotlib text in pt

    Returns
    -------
    offset : Tuple[float, float]
        (x, y) in pt
    '''
    fig = Figure(dpi=72)
    renderer = FigureCanvasAgg(fig).get_renderer()
    text = fig.text(0, 0, tex, ha='center', va='center', usetex=True, fontsize=fontsize)
    _, info, _ = text._get_layout(renderer)
    (_, _, (x, y)), = info
    return x, y

def _fill_holes(geom: BaseGeometry) -> BaseGeometry:
    '''
    Drop the interior rings of every polygon.

    Parameters
    ----------
    geom : BaseGeometry
        Polygon or MultiPolygon

    Returns
    -------
    filled : BaseGeometry
        the same polygons without holes
    '''
    def exteriors(g: BaseGeometry):
        polys = [g] if g.geom_type == 'Polygon' else [p for p in getattr(g, 'geoms', []) if p.geom_type == 'Polygon']
        return [Polygon(p.exterior) for p in polys if not p.is_empty]

    filled = exteriors(unary_union(exteriors(geom)))
    return filled[0] if len(filled) == 1 else MultiPolygon(filled)

def font_size_pt(font_cmd: str) -> float:
    '''
    Size of the font that a LaTeX size command selects, as TeX typesets it.

    Parameters
    ----------
    font_cmd : str
        LaTeX size command, e.g. `\\footnotesize`

    Returns
    -------
    size : float
        font size in TeX pt (1/72.27 in)
    '''
    return _font_size_pt(font_cmd, float(matplotlib.rcParams['font.size']))

@lru_cache(maxsize=None)
def _font_size_pt(font_cmd: str, fontsize: float) -> float:
    '''
    Cached body of `font_size_pt`.

    Parameters
    ----------
    font_cmd : str
        LaTeX size command
    fontsize : float
        size of the matplotlib text in pt

    Returns
    -------
    size : float
        font size in TeX pt
    '''
    with dviread.Dvi(TexManager.make_dvi(rf'{{{font_cmd} x}}', fontsize), 72) as dvi:
        page, = dvi
    return float(page.text[0].font_size) * 72.27 / 72.0

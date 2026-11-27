import math
import numpy as np
import shapely

from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Tuple
from shapely import affinity
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points

from node_labeling.utils.config import Config
from node_labeling.utils.constants import (
    BINDER_DOT_SIZE, DOT_GAP_MM, EDGE_CLEARANCE_MM, LABEL_DISTANCE, MM_PER_UNIT, PADDING_FONT_PT, PADDING_MM, SHAPE_TOLERANCE_MM
)
from node_labeling.v2.label.shape import measure_label_shape, font_size_pt

@dataclass(frozen=True, eq=False)
class LabelShape:
    '''
    Geometry of one label text, relative to the point where the text is drawn
    (normalized units).

    Attributes
    ----------
    ink : BaseGeometry
        union of the glyph outlines, holes filled
    pad : Polygon
        ink grown by the padding with round joins: every outline point is
        `padding` away from the ink, so a node on the outline keeps that
        distance to the letters
    padding : float
        padding (normalized units)
    '''
    ink: BaseGeometry
    pad: Polygon
    padding: float

    @classmethod
    def from_tex(cls, tex: str, cfg: Config) -> 'LabelShape':
        '''
        Measure a formatted label and build its shape (cached per text and padding).

        Parameters
        ----------
        tex : str
            formatted LaTeX label
        cfg : Config
            configuration (font size, padding)

        Returns
        -------
        shape : LabelShape
            ink and pad shape
        '''
        return _label_shape(tex, padding_mm(cfg))

    @classmethod
    def from_ink(cls, ink: BaseGeometry, padding: float) -> 'LabelShape':
        '''
        Build the shape around a given ink geometry.

        Parameters
        ----------
        ink : BaseGeometry
            ink geometry (normalized units)
        padding : float
            padding (normalized units)

        Returns
        -------
        shape : LabelShape
            ink and pad shape
        '''
        return cls(ink=ink, pad=_grow(ink, padding), padding=padding)

    def exp(self, node: Tuple[float, float]) -> Polygon:
        '''
        Pad grown by one padding away from the node, so nothing sits right behind
        the label and its node assignment stays unambiguous.

        The direction comes from the outline normal at the node, i.e. from the
        nearest ink point to the node. With a = that direction, the pad is swept
        by the rectangle spanned by `padding * a / max(|ax|, |ay|)`: one padding
        on the far side at side positions, on both far sides at corner positions
        (as v1 does for its boxes).

        Parameters
        ----------
        node : Tuple[float, float]
            node position, relative to the drawing point

        Returns
        -------
        exp : Polygon
            expanded shape, relative to the drawing point
        '''
        ink_pt, _ = nearest_points(self.ink, Point(node))
        away = np.array([ink_pt.x - node[0], ink_pt.y - node[1]])
        norm = np.linalg.norm(away)
        if norm < 1e-12:
            c = self.pad.centroid
            away = np.array([c.x - node[0], c.y - node[1]])
            norm = np.linalg.norm(away)
        if norm < 1e-12:
            return self.overflow_exp
        reach = self.padding * away / np.max(np.abs(away))
        return _sweep_rect(self.pad, reach[0], reach[1])

    @cached_property
    def spacer(self) -> Polygon:
        '''
        Ink grown by half the minimal distance between two labels
        (`LABEL_DISTANCE` paddings): two labels keep that distance iff their
        spacers do not overlap. Clearly more than the distance between the rows
        of one label, so two stacked labels do not read as one; less than 2
        paddings, which round pads could not meet at a shared node.

        Returns
        -------
        spacer : Polygon
            grown ink, relative to the drawing point
        '''
        return _grow(self.ink, LABEL_DISTANCE * self.padding / 2.0)

    @cached_property
    def dot_ring(self) -> Polygon:
        '''
        Outline on which the binder dot of an overflow label sits: the ink grown
        by the dot radius + `DOT_GAP_MM` (at most the padding), so the dot is
        beside the real letters.

        Returns
        -------
        ring : Polygon
            grown ink, relative to the drawing point
        '''
        dot_mm = math.sqrt(BINDER_DOT_SIZE) / 2.0 * 25.4 / 72.0
        return _grow(self.ink, min((dot_mm + DOT_GAP_MM) / MM_PER_UNIT, self.padding))

    @cached_property
    def body(self) -> BaseGeometry:
        '''
        The label as a solid block for lines passing by (binders): the letters with
        the gaps between letters, words and rows closed, grown by the edge
        clearance (`EDGE_CLEARANCE_MM`, scaled like the padding). Made by shrinking
        the pad (whose gaps are closed) back by padding − clearance.

        Returns
        -------
        body : BaseGeometry
            solid label block, relative to the drawing point
        '''
        clearance = self.padding * EDGE_CLEARANCE_MM / PADDING_MM
        body = self.pad.buffer(-(self.padding - clearance))
        return body if not body.is_empty else self.ink.buffer(clearance)

    @cached_property
    def pad_hull(self) -> Polygon:
        '''
        Returns
        -------
        hull : Polygon
            convex hull of the pad (cheap proxy for soft forces)
        '''
        return self.pad.convex_hull

    @cached_property
    def overflow_exp(self) -> Polygon:
        '''
        Expanded shape of an overflow label (attached by a binder): the same
        margin on every side, ink grown by √2 · padding (as v1's overflow boxes).

        Returns
        -------
        exp : Polygon
            expanded shape, relative to the drawing point
        '''
        return _grow(self.ink, math.sqrt(2) * self.padding)

def padding_mm(cfg: Config) -> float:
    '''
    Label padding in mm: `cfg.padding_mm`, or `PADDING_MM` scaled with the font size.

    Parameters
    ----------
    cfg : Config
        configuration

    Returns
    -------
    padding : float
        padding in mm
    '''
    if cfg.padding_mm is not None:
        return cfg.padding_mm
    return PADDING_MM * font_size_pt(cfg.font_size) / PADDING_FONT_PT

@lru_cache(maxsize=None)
def _label_shape(tex: str, pad_mm: float) -> LabelShape:
    '''
    Cached body of `LabelShape.from_tex`.

    Parameters
    ----------
    tex : str
        formatted LaTeX label
    pad_mm : float
        padding in mm

    Returns
    -------
    shape : LabelShape
        ink and pad shape (normalized units)
    '''
    ink = affinity.scale(measure_label_shape(tex), 1.0 / MM_PER_UNIT, 1.0 / MM_PER_UNIT, origin=(0, 0))
    return LabelShape.from_ink(ink, pad_mm / MM_PER_UNIT)

def _grow(ink: BaseGeometry, distance: float) -> Polygon:
    '''
    One hole-free polygon around the ink at the given distance (round joins).

    The padding is larger than letter and row gaps, so the buffer is a single
    polygon. If not, gaps are closed (buffer out and back in) until it is.

    Parameters
    ----------
    ink : BaseGeometry
        ink geometry
    distance : float
        buffer distance

    Returns
    -------
    grown : Polygon
        simplified outline, holes dropped
    '''
    grown = ink.buffer(distance)
    close = distance
    while grown.geom_type != 'Polygon' and close < 100 * distance:
        grown = ink.buffer(distance + close).buffer(-close)
        close *= 2
    if grown.geom_type != 'Polygon':
        grown = grown.convex_hull
    grown = Polygon(grown.exterior).simplify(SHAPE_TOLERANCE_MM / MM_PER_UNIT)
    return Polygon(grown.exterior)

def _sweep_rect(poly: Polygon, rx: float, ry: float) -> Polygon:
    '''
    Minkowski sum of a polygon and the axis-parallel rectangle spanned by the
    origin and (rx, ry), i.e. the area the polygon covers while moving anywhere
    in that rectangle.

    Every point of the sum outside the polygon lies on an edge moved by a vector
    of the rectangle, so the sum is the polygon plus one hull (edge ⊕ rectangle)
    for each edge that faces a rectangle corner.

    Parameters
    ----------
    poly : Polygon
        hole-free polygon
    rx : float
        x-extent of the rectangle (signed)
    ry : float
        y-extent of the rectangle (signed)

    Returns
    -------
    swept : Polygon
        swept polygon
    '''
    if math.hypot(rx, ry) < 1e-12:
        return poly
    ring = np.asarray(shapely.orient_polygons(poly).exterior.coords)
    a, b = ring[:-1], ring[1:]
    corners = np.array([(0.0, 0.0), (rx, 0.0), (rx, ry), (0.0, ry)])
    normal = np.column_stack([b[:, 1] - a[:, 1], a[:, 0] - b[:, 0]])
    front = (normal @ corners.T).max(axis=1) > 1e-15
    a, b = a[front], b[front]
    pts = np.concatenate([a[:, None, :] + corners, b[:, None, :] + corners], axis=1)
    hulls = shapely.convex_hull(shapely.multipoints(pts))
    hulls = hulls[shapely.get_type_id(hulls) == 3]
    swept = shapely.union_all([poly, *hulls])
    if swept.geom_type != 'Polygon':
        swept = max(getattr(swept, 'geoms', [swept]), key=lambda g: g.area)
    return Polygon(swept.exterior)

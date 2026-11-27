import numpy as np
import shapely

from dataclasses import dataclass
from typing import Tuple

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import substring

from node_labeling.models.label_type import LabelType
from node_labeling.utils.constants import MM_PER_UNIT, SHAPE_TOLERANCE_MM
from node_labeling.v2.models.label_shape import LabelShape

@dataclass(frozen=True, eq=False)
class Slider:
    '''
    All drawing points of a label at which its node lies on the pad outline,
    restricted to the arc that the label type allows (normalized units).

    A drawing point p puts the node on the outline point q (relative to the
    drawing point) if p = node - q, so the slider is the outline arc reflected
    through the node. It is parameterized by the arc length t of the outline,
    t in [0, length]. With a node radius r the outline is the pad grown by r,
    so the node center keeps padding + r from the letters.

    Attributes
    ----------
    node : Tuple[float, float]
        node position
    arc : LineString
        outline arc, relative to the drawing point
    closed : bool
        whether the arc is the whole outline (general labels)
    center_y : float
        height of the label's center line, relative to the drawing point
    '''
    node: Tuple[float, float]
    arc: LineString
    closed: bool
    center_y: float

    @classmethod
    def build(cls, node: Tuple[float, float], shape: LabelShape, label_type: LabelType, node_radius: float = 0.0) -> 'Slider':
        '''
        Parameters
        ----------
        node : Tuple[float, float]
            node position
        shape : LabelShape
            label shape
        label_type : LabelType
            general (whole outline), intent (label above the node) or extent (below)
        node_radius : float
            distance the node center keeps to the pad outline

        Returns
        -------
        slider : Slider
            slider of the label around its node
        '''
        center_y = center_line(shape)
        outline = shape.pad if node_radius <= 0 else _grow_outline(shape.pad, node_radius)
        arc, closed = slider_arc(outline, label_type, center_y)
        return cls(node=(float(node[0]), float(node[1])), arc=arc, closed=closed, center_y=center_y)

    @property
    def length(self) -> float:
        '''
        Returns
        -------
        length : float
            arc length of the slider
        '''
        return self.arc.length

    def local(self, t: np.ndarray) -> np.ndarray:
        '''
        Parameters
        ----------
        t : np.ndarray
            arc lengths

        Returns
        -------
        q : np.ndarray
            (n, 2) outline points, relative to the drawing point
        '''
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return shapely.get_coordinates(shapely.line_interpolate_point(self.arc, t))

    def positions(self, t: np.ndarray) -> np.ndarray:
        '''
        Parameters
        ----------
        t : np.ndarray
            arc lengths

        Returns
        -------
        positions : np.ndarray
            (n, 2) drawing points that put the node on the outline at t
        '''
        return np.asarray(self.node) - self.local(t)

    def position(self, t: float) -> Tuple[float, float]:
        '''
        Parameters
        ----------
        t : float
            arc length

        Returns
        -------
        position : Tuple[float, float]
            drawing point that puts the node on the outline at t
        '''
        x, y = self.positions(t)[0]
        return float(x), float(y)

    def nearest_t(self, position: Tuple[float, float]) -> float:
        '''
        Parameters
        ----------
        position : Tuple[float, float]
            any drawing point

        Returns
        -------
        t : float
            arc length of the slider position closest to it
        '''
        return float(self.arc.project(Point(self.node[0] - position[0], self.node[1] - position[1])))

def _grow_outline(pad: Polygon, r: float) -> Polygon:
    '''
    Parameters
    ----------
    pad : Polygon
        pad outline
    r : float
        distance

    Returns
    -------
    grown : Polygon
        pad grown by r (round joins), simplified, without holes
    '''
    grown = pad.buffer(r).simplify(SHAPE_TOLERANCE_MM / MM_PER_UNIT)
    return Polygon(grown.exterior)

def center_line(shape: LabelShape) -> float:
    '''
    Height of the label's center line: the middle of the vertical extent of its
    letters (relative to the drawing point).

    Parameters
    ----------
    shape : LabelShape
        label shape

    Returns
    -------
    center_y : float
        center line height
    '''
    _, y0, _, y1 = shape.ink.bounds
    return (y0 + y1) / 2.0

def slider_arc(pad: Polygon, label_type: LabelType, center_y: float) -> Tuple[LineString, bool]:
    '''
    Part of the pad outline on which the node may lie.

    - general: the whole outline
    - intent (label above the node): the part at or below the center line, i.e.
      the lower outline plus the lower half of both sides, so the label center
      never goes below the node
    - extent (label below the node): the part at or above the center line

    Multi-row outlines can cross the center line more than twice (steps between
    rows of different length). The arc runs between the outermost crossings,
    through the lowest (intent) or highest (extent) outline point.

    Parameters
    ----------
    pad : Polygon
        pad outline, relative to the drawing point
    label_type : LabelType
        label type
    center_y : float
        center line height

    Returns
    -------
    arc : LineString
        counter-clockwise outline arc
    closed : bool
        whether the arc is the whole outline
    '''
    ring = LineString(shapely.orient_polygons(pad).exterior.coords)
    if label_type == LabelType.GENERAL:
        return ring, True

    coords = np.asarray(ring.coords)
    seg_len = np.hypot(*np.diff(coords, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    above = coords[:, 1] > center_y
    crossings = []
    for i in np.flatnonzero(above[:-1] != above[1:]):
        (x0, y0), (x1, y1) = coords[i], coords[i + 1]
        f = (center_y - y0) / (y1 - y0)
        crossings.append((x0 + f * (x1 - x0), s[i] + f * seg_len[i]))
    if len(crossings) < 2:
        return ring, True

    s_left = min(crossings)[1]
    s_right = max(crossings)[1]
    extreme = np.argmin(coords[:-1, 1]) if label_type == LabelType.INTENT else np.argmax(coords[:-1, 1])
    s_ext = s[extreme]
    total = ring.length

    def forward(a: float, b: float) -> LineString:
        if a <= b:
            return substring(ring, a, b)
        head = np.asarray(substring(ring, a, total).coords)
        tail = np.asarray(substring(ring, 0.0, b).coords)
        return LineString(np.vstack([head, tail[1:]]))

    def on_forward(a: float, b: float, x: float) -> bool:
        return (x - a) % total <= (b - a) % total

    if on_forward(s_left, s_right, s_ext):
        return forward(s_left, s_right), False
    return forward(s_right, s_left), False

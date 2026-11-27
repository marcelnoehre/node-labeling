import math
import numpy as np

from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Tuple

from matplotlib.path import Path
from shapely.geometry import LineString

from node_labeling.utils.constants import MM_PER_UNIT, SHAPE_TOLERANCE_MM

Point2 = Tuple[float, float]

@dataclass(frozen=True, eq=False)
class Binder:
    '''
    Line from a node to the attachment point of its overflow label (normalized
    units): straight, or a cubic Bézier curve (T8). Immutable.

    Attributes
    ----------
    start : Tuple[float, float]
        node position
    end : Tuple[float, float]
        attachment point (center of the binder dot)
    controls : Optional[Tuple[Tuple[float, float], Tuple[float, float]]]
        the two cubic control points, None for a straight binder
    '''
    start: Point2
    end: Point2
    controls: Optional[Tuple[Point2, Point2]] = None

    @property
    def straight(self) -> bool:
        '''
        Returns
        -------
        straight : bool
            whether the binder is a straight segment
        '''
        return self.controls is None

    @property
    def kind(self) -> str:
        '''
        Returns
        -------
        kind : str
            'straight', 'bend' (both control points on the same side of the
            chord) or 's' (opposite sides)
        '''
        if self.straight:
            return 'straight'
        sides = [_side(self.start, self.end, c) for c in self.controls]
        return 's' if sides[0] * sides[1] < 0 else 'bend'

    @property
    def bend(self) -> float:
        '''
        Returns
        -------
        bend : float
            largest angle between the chord and the end tangents, in degrees
        '''
        if self.straight:
            return 0.0
        chord = np.subtract(self.end, self.start)
        a = _angle(chord, np.subtract(self.controls[0], self.start))
        b = _angle(chord, np.subtract(self.end, self.controls[1]))
        return max(a, b)

    @property
    def start_tangent(self) -> np.ndarray:
        '''
        Returns
        -------
        tangent : np.ndarray
            unit direction in which the binder leaves the node
        '''
        target = self.end if self.straight else self.controls[0]
        d = np.subtract(target, self.start)
        n = np.linalg.norm(d)
        return d / n if n > 0 else d

    @cached_property
    def line(self) -> LineString:
        '''
        Returns
        -------
        line : LineString
            the binder for geometry tests; curves are flattened so that no
            point is more than `SHAPE_TOLERANCE_MM` off the curve
        '''
        if self.straight:
            return LineString([self.start, self.end])
        p0, p1, p2, p3 = (np.asarray(p, dtype=float) for p in (self.start, *self.controls, self.end))
        m = max(np.linalg.norm(p0 - 2 * p1 + p2), np.linalg.norm(p1 - 2 * p2 + p3))
        tol = SHAPE_TOLERANCE_MM / MM_PER_UNIT
        n = max(1, math.ceil(math.sqrt(3 * m / (4 * tol))))
        t = np.linspace(0.0, 1.0, n + 1)[:, None]
        pts = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3
        return LineString(pts)

    @property
    def length(self) -> float:
        '''
        Returns
        -------
        length : float
            arc length
        '''
        return self.line.length

    def path(self) -> Path:
        '''
        Returns
        -------
        path : Path
            matplotlib path (a real curve for Bézier binders)
        '''
        if self.straight:
            return Path([self.start, self.end], [Path.MOVETO, Path.LINETO])
        return Path([self.start, *self.controls, self.end], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])

def _side(a: Point2, b: Point2, p: Point2) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

def _angle(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(u, v) / (nu * nv))))))

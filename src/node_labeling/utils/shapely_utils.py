from typing import List

from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

def as_points(geom: BaseGeometry) -> List[Point]:
    '''
    Flatten a shapely intersection geometry into the Points it is made of.

    Two edges can be collinear and overlap along a shared sub-segment
    (e.g. one edge's endpoints both lie on another edge), in which case
    `LineString.intersection` returns a LineString/GeometryCollection
    instead of a Point. This reduces any such geometry to the finite set
    of points that bound it.

    Parameters
    ----------
    geom : BaseGeometry
        Result of `edge.intersection(other_edge)`

    Returns
    -------
    List[Point]
        Constituent points of the intersection
    '''
    if geom.is_empty:
        return []
    if geom.geom_type == 'Point':
        return [geom]
    if geom.geom_type == 'MultiPoint':
        return list(geom.geoms)
    if geom.geom_type in ('LineString', 'LinearRing'):
        coords = list(geom.coords)
        return [Point(coords[0]), Point(coords[-1])]
    if geom.geom_type == 'MultiLineString' or geom.geom_type == 'GeometryCollection':
        points = []
        for part in geom.geoms:
            points.extend(as_points(part))
        return points
    return []

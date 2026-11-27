from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import math
import networkx as nx

from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry

from node_labeling.utils.shapely_utils import as_points
from node_labeling.utils.constants import NODE_RADIUS, MM_PER_UNIT, BINDER_DOT_SIZE, BINDER_DOT_GAP_MM, ROW_BREAK
from node_labeling.models.anchor import Anchor, AnchorType

if TYPE_CHECKING:
    from node_labeling.v2.models.label_candidate import LabelCandidate

def pad_overlap(a: LabelCandidate, b: LabelCandidate) -> bool:
    '''
    True iff the padded boxes of two candidates intersect (touching is not a conflict).

    Parameters
    ----------
    a : LabelCandidate
        first label candidate 
    b : LabelCandidate
        second label candidate

    Returns
    -------
    conflict : bool
        wether the ink boxes overlap
    '''
    buffer = 0.05
    (ax0, ay0), _, (ax1, ay1), _ = a.pad_bbox_corners
    (bx0, by0), _, (bx1, by1), _ = b.pad_bbox_corners
    
    return (ax0 + buffer < bx1 and ax1 - buffer > bx0 and ay0 + buffer < by1 and ay1 - buffer > by0)

def bbox_corners(cx, cy, hx, hy):
    '''
    Derive the corner positions based on the center and the half-size
    
    Parameters
    ----------
    cx : float
        x-coordinate of the center
    cy : float
        y-coordinate of the center
    hx : float
        half-width
    hy : float
        half-height

    Returns
    -------
    bbox_corners : Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]
        bbox corners based on the center and half-size
    '''
    return (
        (cx - hx, cy - hy), # BL
        (cx + hx, cy - hy), # BR
        (cx + hx, cy + hy), # TR
        (cx - hx, cy + hy), # TL
    )

def label_rows(text: str) -> int:
    '''
    Number of rows of a formatted label.

    Parameters
    ----------
    text : str
        formatted LaTeX label

    Returns
    -------
    rows : int
        number of rows
    '''
    return len(text.split(ROW_BREAK))

def label_padding(ink_h: float, rows: int) -> float:
    '''
    Padding between the ink box and the padded box of a label.
    The formula is defined in mm, input and output are normalized units.

    Parameters
    ----------
    ink_h : float
        height of the ink box (normalized units)
    rows : int
        number of rows of the label

    Returns
    -------
    padding : float
        padding (normalized units)
    '''
    ink_h_mm = ink_h * MM_PER_UNIT
    padding_mm = (ink_h_mm / rows - 0.1 * rows) * 0.5
    return max(0.0, padding_mm) / MM_PER_UNIT

def overflow_boxes(cx: float, cy: float, half_iw: float, half_ih: float, padding: float):
    '''
    Ink, padded and expanded box of an overflow label centered at (cx, cy).
    Overflow labels are attached by a binder, so the padding is the same on every side.

    Parameters
    ----------
    cx : float
        x-coordinate of the ink box center
    cy : float
        y-coordinate of the ink box center
    half_iw : float
        half-width of the ink box
    half_ih : float
        half-height of the ink box
    padding : float
        label padding (see `label_padding`)

    Returns
    -------
    ink, pad, exp : Tuple[corners, corners, corners]
        bbox corners (BL, BR, TR, TL) of the three boxes
    '''
    exp_margin = padding * math.sqrt(2)
    return (
        bbox_corners(cx, cy, half_iw, half_ih),
        bbox_corners(cx, cy, half_iw + padding, half_ih + padding),
        bbox_corners(cx, cy, half_iw + exp_margin, half_ih + exp_margin),
    )

def binder_anchor(anchor_type: AnchorType, cx: float, cy: float, half_iw: float, half_ih: float, padding: float) -> Anchor:
    '''
    Final binder anchor of an overflow label: just outside the ink box, so the binder dot sits close to
    the text without covering it. The dot center keeps the dot radius plus `BINDER_DOT_GAP_MM` to the ink box,
    corner dots measured to the ink corner (diagonally). It never leaves the pad box, which stays the
    reserved space of the label.

    Parameters
    ----------
    anchor_type : AnchorType
        side or corner of the label the binder attaches to
    cx : float
        x-coordinate of the ink box center
    cy : float
        y-coordinate of the ink box center
    half_iw : float
        half-width of the ink box
    half_ih : float
        half-height of the ink box
    padding : float
        label padding (see `label_padding`)

    Returns
    -------
    anchor : Anchor
        anchor on the ink box grown by the dot offset
    '''
    dot_radius_mm = math.sqrt(BINDER_DOT_SIZE) / 2.0 * 25.4 / 72.0
    offset = min((dot_radius_mm + BINDER_DOT_GAP_MM) / MM_PER_UNIT, padding)
    if anchor_type in (AnchorType.TL, AnchorType.TR, AnchorType.BL, AnchorType.BR):
        offset /= math.sqrt(2)
    return Anchor(anchor_type, cx, cy, *bbox_corners(cx, cy, half_iw + offset, half_ih + offset))

def label_wh(corners: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]) -> Tuple[float, float]:
    '''
    Bbox dimensions.

    Parameters
    ----------
    corners : Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]
        bbox corners
    
    Returns
    -------
    width, height : Tuple[float, float]
        size of the given label
    '''
    bl, br, _, tl = corners
    return abs(br[0] - bl[0]), abs(tl[1] - bl[1])


def bbox_polygon(cx: float, cy: float, w: float, h: float) -> Polygon:
    '''
    Create a polygon for a given bbox bbox centered at (cx, cy) with width w and height h.

    Parameters
    ----------
    cx : float 
        x-coordinate of center
    cy : float 
        y-coordinate of center
    w : float
        desired width 
    h : float
        desired height

    Returns
    -------
    bbox_poly : Polygon
        bbox polygon derived from center and size 
    '''
    hw, hh = w / 2, h / 2
    return Polygon([
        (cx - hw, cy - hh), (cx + hw, cy - hh),
        (cx + hw, cy + hh), (cx - hw, cy + hh),
    ])

def binding_line_valid(
    G: nx.Graph,
    lid: int,
    anchor: Anchor,
    label_candidates: Dict[int, List[LabelCandidate]],
    overflow_candidates: Dict[int, LabelCandidate],
    placed_overflow: List[int],
    edges: Optional[List[Tuple[int, int]]] = None,
    soft: bool = False,
    max_edge_crossings: int = -1
) -> bool:
    '''
    Check wether a binding line is valid.

    Parameters
    ----------
    G : nx.Graph
        graph containing positions
    lid : int
        label the binding line is attached to
    anchor : Anchor
        chosen anchor
    label_candidates : Dict[int, List[LabelCandidate]]
        already placed labels
    overflow_candidates : Dict[int, LabelCandidate]
        overflow candidates to be placed
    placed_overflow : List[Dict[int, Tuple[float, float], LineString]]
        already placed overflow candidates
    edges : Optional[List[Tuple[int, int]]] = None
        forbidden edges of the graph
    soft : bool = False
        soft mode is less restrictive
    max_edge_crossings : int = -1
        restrict max number of edge crossings

    Returns
    valid : bool
        wether the binding line is valid
    '''
    ol = overflow_candidates[lid]
    binding_line = LineString([anchor.pos, G.nodes[ol.node_id]['pos']])

    for o_lid in placed_overflow:
        if o_lid == lid:
            continue

        o_ol = overflow_candidates[o_lid]
        o_cx, o_cy = G.nodes[o_ol.node_id]['pos']
        w, h = label_wh(overflow_candidates[o_lid].exp_bbox_corners)

        # binder intersects already placed overflow label
        if binding_line.intersects(bbox_polygon(*o_ol.center, w, h)):
            return False
        
        # binder conflicts with another binder
        if not soft:
            if binding_line.crosses(LineString([o_ol.anchor.pos, (o_cx, o_cy)])):
                return False
    
    for candidates in label_candidates.values():
        for cand in candidates:
            ink_poly = cand.ink_poly
            pad_poly = cand.pad_poly

            if soft:
                # binder intersects ink bbox of a label
                if binding_line.intersects(ink_poly):
                    return False
            else:
                # binder intersects padded bbox of a label
                if binding_line.intersects(pad_poly):
                    return False

    for nid, data in G.nodes(data=True):
        if nid == ol.node_id:
            continue
        if not isinstance(nid, int):
            continue
        
        # binder intersects with another node
        if binding_line.distance(Point(data['pos'])) < NODE_RADIUS:
            return False
    
    for a, b in edges or []:
        # binder to close to an forbidden
        if Point(anchor.pos).distance(LineString([G.nodes[a]['pos'], G.nodes[b]['pos']])) < NODE_RADIUS:
            return False

        
    if max_edge_crossings >= 0:
        # restrict the number of edge crossings 
        crossings = 0
        for u, v in G.edges():
            if binding_line.crosses(LineString([G.nodes[u]['pos'], G.nodes[v]['pos']])):
                crossings += 1
                
            if crossings > max_edge_crossings:
                return False

    # valid - no conflict found
    return True

def angle_from_centroid(centroid: Tuple[float, float], point: Tuple[float, float]) -> float:
    '''
    Angle of a point from the centroid of the drawing.

    Parameters
    ----------
    centroid : Tuple[float, float]
        centroid of the drawing
    point : Tuple[float, float]
        point to get the angle for

    Returns
    -------
    angle : float
        angle from centroid to the point
    '''
    return math.atan2(point[1] - centroid[1], point[0] - centroid[0])

def angular_gap_between(a1: float, a2: float) -> float:
    '''
    Calculates the counter-clockwise angular gap between two angles.

    Parameters
    ----------
    a1 : float
        starting angle in radians.
    a2 : float
        ending angle in radians.

    Returns
    -------
    gap : float
        positive angular distance from a1 to a2 in the range [0, 2π).
    '''
    return (a2 - a1) % (2 * math.pi)
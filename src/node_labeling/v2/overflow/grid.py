import math
import numpy as np
import shapely

from typing import Any, Dict, List, Tuple

from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry

from node_labeling.models.label_type import LabelType
from node_labeling.utils.config import Config
from node_labeling.utils.constants import OUTER_MARGIN
from node_labeling.v2.models.candidate import Candidate
from node_labeling.v2.overflow.binders import Layout, alternatives, attach_many, binder_valid
from node_labeling.v2.utils.geometry import angle_from_centroid, angular_gap_between
from node_labeling.v2.utils.shapes import translate_many

ALTERNATIVE_BINDERS: int = 8

def _generate_candidates(
        lid: int,
        layout: Layout,
        centroid: Tuple[float, float],
        alpha_polygon: Polygon,
        placed_union: BaseGeometry,
        overflow: Dict[int, Candidate],
        cfg: Config
) -> List[Tuple[Candidate, float]]:
    '''
    Grid positions outside the diagram for one overflow label, with their cost
    (v1's grid on shapes).

    - grid of drawing points every `cfg.grid_step` around the alpha shape
    - window: the label center is at least `min_label_dist` from the boundary,
      and the side of the exp shape facing the boundary at most `max_label_dist`
      (the shape's support towards the boundary, not a box)
    - the exp shape stays outside the alpha shape, off placed labels and other nodes
    - binder: the shortest one to the dot ring; if it is invalid (soft rules),
      the next shortest of `ALTERNATIVE_BINDERS` others
    - cost: half plane, angle to the node, binder length, alignment, distance of
      the attachment to the boundary (weights from `cfg`)

    Parameters
    ----------
    lid : int
        overflow label
    layout : Layout
        fixed context
    centroid : Tuple[float, float]
        centroid of the drawing
    alpha_polygon : Polygon
        alpha shape (prepared)
    placed_union : BaseGeometry
        union of the exp shapes of the labels placed at their nodes (prepared)
    overflow : Dict[int, Candidate]
        overflow labels (those with a binder are placed)
    cfg : Config
        configuration

    Returns
    -------
    scored : List[Tuple[Candidate, float]]
        best `cfg.top_k` positions with binder, and their cost
    '''
    ol = overflow[lid]
    shape = ol.shape
    exp = shape.overflow_exp
    ex0, ey0, ex1, ey1 = exp.bounds
    V = np.asarray(exp.exterior.coords)
    node = np.asarray(ol.node_pos)

    bx0, by0, bx1, by1 = alpha_polygon.bounds
    margin = max(ex1 - ex0, ey1 - ey0) + OUTER_MARGIN
    xs = np.arange(bx0 - margin - ex0, bx1 + margin - ex1, cfg.grid_step)
    ys = np.arange(by0 - margin - ey0, by1 + margin - ey1, cfg.grid_step)
    P = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)

    ext = np.asarray(alpha_polygon.exterior.coords)
    a, d = ext[:-1], np.diff(ext, axis=0)
    l2 = np.where((d ** 2).sum(axis=1) > 0, (d ** 2).sum(axis=1), 1.0)
    t = np.clip(((P[:, None, :] - a[None]) * d[None]).sum(axis=2) / l2[None], 0.0, 1.0)
    closest = a[None] + t[:, :, None] * d[None]
    dist2 = ((P[:, None, :] - closest) ** 2).sum(axis=2)
    k = dist2.argmin(axis=1)
    rows = np.arange(len(P))
    min_dist = np.sqrt(dist2[rows, k])
    away = (P - closest[rows, k]) / np.where(min_dist > 0, min_dist, 1.0)[:, None]
    toward = (V @ (-away.T)).max(axis=0)
    P = P[(min_dist >= cfg.min_label_dist) & (min_dist - toward <= cfg.max_label_dist)]
    if not len(P):
        return []

    obst = layout.obstacles
    others = obst.node_xy[np.array([n != ol.node_id for n in obst.node_ids], dtype=bool)]
    if len(others):
        rel_x = others[None, :, 0] - P[:, None, 0]
        rel_y = others[None, :, 1] - P[:, None, 1]
        inside = shapely.contains_xy(exp, rel_x, rel_y) | shapely.intersects_xy(exp, rel_x, rel_y)
        P = P[~inside.any(axis=1)]
    if not len(P):
        return []
    exps = translate_many(exp, P)
    P = P[~(shapely.intersects(exps, alpha_polygon) | shapely.intersects(exps, placed_union))]
    if not len(P):
        return []

    cands = [ol.moved(p) for p in P]
    binders = attach_many(cands)
    alpha_ext = alpha_polygon.exterior
    ix0, iy0, ix1, iy1 = shape.ink.bounds
    ink_center = np.array([(ix0 + ix1) / 2.0, (iy0 + iy1) / 2.0])
    node_angle = math.atan2(node[1] - centroid[1], node[0] - centroid[0])

    scored: List[Tuple[Candidate, float]] = []
    for c, b in zip(cands, binders):
        if not binder_valid(layout, lid, c, b, overflow, soft=True):
            b = next((alt for alt in alternatives(c)[1:ALTERNATIVE_BINDERS + 1]
                      if binder_valid(layout, lid, c, alt, overflow, soft=True)), None)
            if b is None:
                continue
        p = np.asarray(c.position)
        center = p + ink_center
        to_node = node - center
        to_end = np.asarray(b.end) - center
        n1, n2 = np.linalg.norm(to_node), np.linalg.norm(to_end)
        align = float(np.dot(to_node, to_end) / (n1 * n2)) if n1 > 0 and n2 > 0 else -1.0

        _, pad_y0, _, pad_y1 = c.pad.bounds
        c_type = 0.0
        if ol.label_type == LabelType.INTENT and (b.end[1] < node[1] or pad_y0 < node[1]):
            c_type = cfg.w_type
        elif ol.label_type == LabelType.EXTENT and (b.end[1] > node[1] or pad_y1 > node[1]):
            c_type = cfg.w_type
        cell_angle = math.atan2(p[1] - centroid[1], p[0] - centroid[0])
        angle_offset = abs((cell_angle - node_angle + math.pi) % (2 * math.pi) - math.pi)
        cost = (angle_offset * cfg.w_angle + b.length * cfg.w_binder + (1.0 - align) * cfg.w_align
                + alpha_ext.distance(Point(b.end)) * cfg.w_boundary + c_type)
        scored.append((c.moved(c.position, b), cost))

    scored.sort(key=lambda x: (x[1], x[0].position[0], x[0].position[1]))
    return scored[:cfg.top_k]

def grid_overflow_candidates(
        layout: Layout,
        overflow: Dict[int, Candidate],
        gaps: List[Dict[str, Any]],
        centroid: Tuple[float, float],
        alpha_polygon: Polygon,
        placed_union: BaseGeometry,
        cfg: Config
) -> Dict[int, List[Tuple[Candidate, float]]]:
    '''
    Grid positions for every unplaced overflow label, in the order of the
    boundary gaps they are assigned to (sequential: the vectorized grid is
    faster than v1's thread pool, whose Python loops contend for the GIL).

    Parameters
    ----------
    layout : Layout
        fixed context
    overflow : Dict[int, Candidate]
        overflow labels
    gaps : List[Dict[str, Any]]
        gaps between the nodes on the boundary walk of the alpha shape
    centroid : Tuple[float, float]
        centroid of the drawing
    alpha_polygon : Polygon
        alpha shape
    placed_union : BaseGeometry
        union of the exp shapes of the labels placed at their nodes
    cfg : Config
        configuration

    Returns
    -------
    grid_candidates : Dict[int, List[Tuple[Candidate, float]]]
        scored positions per label
    '''
    order: List[int] = []
    for gap in gaps:
        assigned = sorted(gap['assigned'], key=lambda nid: angular_gap_between(gap['a_left'], angle_from_centroid(centroid, layout.node_pos(nid))))
        for nid in assigned:
            order += sorted(lid for lid, ol in overflow.items() if ol.node_id == nid and ol.binder is None and lid not in order)
    for lid in order:
        shapely.prepare(overflow[lid].shape.overflow_exp)
    return {lid: _generate_candidates(lid, layout, centroid, alpha_polygon, placed_union, overflow, cfg) for lid in order}

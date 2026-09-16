import numpy as np
import networkx as nx

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any
from shapely import unary_union
from shapely.geometry import Polygon
from shapely.prepared import prep

from node_labeling.utils.config import Config
from node_labeling.models.label_candidate import LabelCandidate
from node_labeling.models.label_type import LabelType
from node_labeling.models.anchor import AnchorType
from node_labeling.utils.geometry import *
from node_labeling.utils.constants import *

def _generate_candidates(
        lid: int,
        G: nx.Graph,
        centroid: Tuple[float, float],
        alpha_polygon: Polygon,
        placed_union: Polygon,
        placed_overflow: Dict[int, LabelCandidate],
        label_candidates: Dict[int, List[LabelCandidate]],
        overflow_candidates: Dict[int, LabelCandidate],
        cfg: Config
) -> List[Tuple[LabelCandidate, float]]:
    '''
    Generate all valid grid candidates for a given overflow label.

    Parameters
    ----------
    lid : int
        overflow label to generate candidates for
    G : nx.Graph
        graph containing positions
    centroid : Tuple[float, float]
        centroid of the drawing
    alpha_polygon : Polygon
        alpha shape
    placed_union : Polygon
        combined polygon of all already placed labels
    placed_overflow : Dict[int, LabelCandidate]
        already placed bounded overflow labels
    label_candidates : Dict[int, List[LabelCandidate]]
        already placed label candidates
    overflow_candidates : Dict[int, LabelCandidate]
        all overflow candidates
    cfg: Config
        configuration

    Returns
    -------
    scored : List[Tuple[LabelCandidate, float]]
        all grid candidates with their assigned cost
    '''
    ol = overflow_candidates[lid]
    iw, ih = label_wh(ol.ink_bbox_corners)
    pw, ph = label_wh(ol.pad_bbox_corners)
    ew, eh = label_wh(ol.exp_bbox_corners)
    hw, hh = ew / 2, eh / 2
    node_pos = G.nodes[ol.node_id]['pos']
    node_pt = np.array(node_pos, dtype=float)
    other_node_pos = np.array([data['pos'] for nid, data in G.nodes(data=True) if isinstance(nid, int) and nid != ol.node_id], dtype=float)
    alpha_bounds = alpha_polygon.bounds
    alpha_exterior = alpha_polygon.exterior
    prepared_alpha = prep(alpha_polygon)
    prepared_placed_union = prep(placed_union)

    # grid
    margin = max(ew, eh) + OUTER_MARGIN
    gx0 = alpha_bounds[0] - margin
    gy0 = alpha_bounds[1] - margin
    gx1 = alpha_bounds[2] + margin
    gy1 = alpha_bounds[3] + margin
    xs = np.arange(gx0 + hw, gx1, cfg.grid_step)
    ys = np.arange(gy0 + hh, gy1, cfg.grid_step)
    gx, gy = np.meshgrid(xs, ys)
    cx_all = gx.ravel()
    cy_all = gy.ravel()
    ext_coords = np.array(alpha_exterior.coords)
    seg_a = ext_coords[:-1]
    seg_b = ext_coords[1:]
    seg_d = seg_b - seg_a
    seg_l2 = (seg_d ** 2).sum(axis=1)
    pts_3d = np.stack([cx_all, cy_all], axis=1)[:, None, :]
    sa_3d = seg_a[None, :, :]
    sd_3d = seg_d[None, :, :]
    sl2_3d = seg_l2[None, :]
    t = np.clip(((pts_3d - sa_3d) * sd_3d).sum(axis=2) / np.where(sl2_3d > 0, sl2_3d, 1.0), 0.0, 1.0)
    closest = sa_3d + t[:, :, None] * sd_3d
    dist2 = ((pts_3d - closest) ** 2).sum(axis=2)
    min_dist = np.sqrt(dist2.min(axis=1))

    keep = (min_dist >= cfg.min_label_dist) & (min_dist <= cfg.max_label_dist)
    ex0 = cx_all - hw;  ey0 = cy_all - hh
    ex1 = cx_all + hw;  ey1 = cy_all + hh
    overlaps_poly_aabb = ~(
        (ex1 < alpha_bounds[0]) | (ex0 > alpha_bounds[2]) |
        (ey1 < alpha_bounds[1]) | (ey0 > alpha_bounds[3])
    )
    if len(other_node_pos):
        ox = other_node_pos[:, 0]
        oy = other_node_pos[:, 1]
        node_inside = (
            (ox[None, :] >= ex0[:, None]) & (ox[None, :] <= ex1[:, None]) &
            (oy[None, :] >= ey0[:, None]) & (oy[None, :] <= ey1[:, None])
        ).any(axis=1)
        keep &= ~node_inside

    candidate_indices = np.where(keep)[0] # pruned search space

    scored: List[Tuple[LabelCandidate, float]] = []
    for i in candidate_indices:
        ox_i, oy_i = float(cx_all[i]), float(cy_all[i])
        exp_bbox = bbox_polygon(ox_i, oy_i, ew, eh)

        if overlaps_poly_aabb[i] and prepared_alpha.intersects(exp_bbox):
            continue # must be in graph exterior

        if prepared_placed_union.intersects(exp_bbox):
            continue # overlaps already placed label

        pad_bbox = bbox_polygon(ox_i, oy_i, pw, ph)
        pad_corners = bbox_corners(ox_i, oy_i, pw/2, ph/2)
        ink = bbox_polygon(ox_i, oy_i, iw, ih)
        center = np.array([ox_i, oy_i])

        # anchor selection
        vec = node_pt - center
        dist = float(np.linalg.norm(vec))
        unit = vec / dist if dist > 0 else vec
        anchors = [Anchor(at, ox_i, oy_i, *pad_corners) for at in AnchorType if at != AnchorType.O]
        pts = np.array([a.pos for a in anchors])
        va = pts - center
        na = np.linalg.norm(va, axis=1)
        safe = na > 0
        aligns = np.where(safe, (va / np.where(safe[:,None], na[:,None], 1.0)) @ unit, -1.0)

        chosen_anchor: Anchor = None
        chosen_align: float = None

        for anchor_i in np.argsort(-aligns):
            pt = pts[anchor_i]
            line = LineString([pt.tolist(), node_pos])

            if ink.intersects(line):
                continue # intersects own ink

            if not binding_line_valid(G, lid, anchors[anchor_i], label_candidates, overflow_candidates, placed_overflow, [], True):
                continue # invalid binding line

            chosen_anchor = anchors[anchor_i]
            chosen_align = float(aligns[anchor_i])
            break

        if chosen_anchor is None:
            continue # no valid anchor found

        angle_to_cell = math.atan2(oy_i - centroid[1], ox_i - centroid[0])
        node_angle = math.atan2(node_pos[1] - centroid[1], node_pos[0] - centroid[0])
        angle_offset = abs((angle_to_cell - node_angle + math.pi) % (2 * math.pi) - math.pi)
        anchor_shapely = Point(chosen_anchor.pos)
        dist_to_boundary = alpha_exterior.distance(anchor_shapely)
        _, min_y, _, max_y = pad_bbox.bounds

        # cost
        c_type = 0.0
        if ol.label_type == LabelType.INTENT:
            if (anchor_shapely.y < node_pos[1]) or (min_y < node_pos[1]):
                c_type = cfg.w_type
        elif ol.label_type == LabelType.EXTENT:
            if (anchor_shapely.y > node_pos[1]) or (max_y > node_pos[1]):
                c_type = cfg.w_type

        c_angle = angle_offset * cfg.w_angle
        c_binder = LineString([chosen_anchor.pos, node_pos]).length * cfg.w_binder
        c_align = (1.0 - chosen_align) * cfg.w_align
        c_boundary = dist_to_boundary * cfg.w_boundary
        cost = c_angle + c_binder + c_align + c_boundary + c_type
        scored.append((LabelCandidate(
            ol.node_id, chosen_anchor, ol.label_type, bbox_corners(ox_i, oy_i, iw/2, ih/2), 
            pad_corners, bbox_corners(ox_i, oy_i, ew/2, eh/2), (ox_i, oy_i), ol.text
        ), cost))

    scored.sort(key=lambda x: (x[1], x[0].center[0], x[0].center[1]))
    return scored[:cfg.top_k]

def _generate_grid_candidates(args: Dict[str, Any]) -> Tuple[int, List[Tuple[LabelCandidate, float]]]:
    '''
    Worker function for parallel label candidate generation.

    Parameters
    ----------
    args : Dict[str, Any]
        dictionary passing the context

    Returns
    -------
    lid, grid_candidates : Tuple[int, List[Tuple[LabelCandidate, float]]]
        label id and all valid grid candidates 
    '''
    grid_candidates = _generate_candidates(
        args['lid'], args['G'], args['centroid'], args['alpha_polygon'], args['placed_union'], 
        args['placed_overflow'], args['label_candidates'], args['overflow_candidates'], args['cfg'])

    return args['lid'], grid_candidates

def grid_overflow_candidates(
        G: nx.Graph,
        label_candidates: Dict[int, List[LabelCandidate]],
        overflow_candidates: Dict[int, LabelCandidate],
        gaps: Dict[str, Any],
        centroid: Tuple[float, float],
        alpha_polygon: Polygon,
        placed_union: Polygon,
        placed_overflow: Dict[int, LabelCandidate],
        cfg: Config
    ) -> Dict[int, List[Tuple[LabelCandidate, float]]]:
    '''
    Generate a set of valid label candidates for each unplaced overflow label.

    Parameters
    ----------
    G : nx.Graph
        graph containing positions
    label_candidates : Dict[int, List[LabelCandidate]]
        already placed label candidates
    overflow_candidates : Dict[int, LabelCandidate]
        all overflow candidates
    gaps : Dict[str, Any]
        gaps between the nodes on the boundary walk of the alpha shape
    centroid : Tuple[float, float]
        centroid of the drawing
    alpha_polygon : Polygon
        alpha shape
    placed_union : Polygon
        combined polygon of all already placed labels
    placed_overflow : Dict[int, LabelCandidate]
        already placed bounded overflow labels
    cfg: Config
        configuration

    Returns
    -------
    grid_candidates : Dict[int, List[Tuple[LabelCandidate, float]]]
        valid grid candidates with their cost for each unplaced overflow candidate
    '''
    grid_candidates: Dict[int, List[Tuple[LabelCandidate, float]]] = {}

    tasks: List[Tuple[LabelCandidate, float]] = []
    for gap in gaps:
        if not gap['assigned']:
            continue

        assigned = sorted(
            gap['assigned'], 
            key=lambda nid: angular_gap_between(gap['a_left'], angle_from_centroid(centroid, G.nodes[nid]['pos']))
        )
        
        for nid in assigned:
            matches = [
                (l_id, ol) for l_id, ol in overflow_candidates.items() 
                if ol.node_id == nid
            ]
            matches.sort(key=lambda x: x[0])
            for lid, ol in matches:
                tasks.append({
                    'lid': lid,
                    'G': G,
                    'centroid': centroid,
                    'alpha_polygon': alpha_polygon,
                    'placed_union': placed_union,
                    'placed_overflow': placed_overflow,
                    "label_candidates":   label_candidates,
                    "overflow_candidates": overflow_candidates,
                    'cfg': cfg
                })

    with ThreadPoolExecutor() as pool:
        results = pool.map(_generate_grid_candidates, tasks)
        for g_lid, candidates in results:
            grid_candidates[g_lid] = candidates

    return grid_candidates
import time
import networkx as nx

from typing import Dict, List
from shapely import unary_union
from shapely.geometry import Polygon

from node_labeling.utils.config import Config
from node_labeling.models.label_candidate import LabelCandidate
from node_labeling.models.anchor import AnchorType
from node_labeling.utils.geometry import *
from node_labeling.overflow.grid import *
from node_labeling.overflow.hungarian import *

def _adjust_anchors(
        G: nx.Graph, 
        label_candidates: Dict[int, List[LabelCandidate]], 
        overflow_candidates: Dict[int, LabelCandidate], 
        unbounded_overflow_labels: List[int],
        alpha_polygon: Polygon
    ) -> Dict[int, LabelCandidate]:
    '''
    Translate label to improve anchoring.

    Parameters
    ----------
    G : nx.Graph
        graph containing positions
    label_candidates: Dict[int, List[LabelCandidate]], 
        already placed candidates
    overflow_candidates: Dict[int, LabelCandidate], 
        placed overflow candidates
    unbounded_ooverflow_candidatesverflow_labels: List[int],
        overflow labels that are placed in the graph's exterior
    alpha_polygon: Polygon
        polygon of the drawing

    Returns
    -------
    overflow_candidates: Dict[int, LabelCandidate]
        refined overflow candidates
    '''
    sorted_unbounded_overflow_labels = sorted(
        unbounded_overflow_labels, 
        key=lambda lid: Point(overflow_candidates[lid].anchor.pos).distance(alpha_polygon)
    )

    for lid in sorted_unbounded_overflow_labels:
        ol = overflow_candidates[lid]
        cx, cy = ol.center
        anchor_candidates = [
            Anchor(at, *ol.center, *ol.pad_bbox_corners)
            for at in AnchorType if at != AnchorType.O
        ]

        label_center = np.array([cx, cy])
        node_pos = G.nodes[ol.node_id]['pos']
        node_pt = np.array(node_pos)
        vec = node_pt - label_center
        dist_to_node = np.linalg.norm(vec)
        unit = vec / dist_to_node if dist_to_node > 0 else vec
        scored_anchors: List[Tuple[float, Anchor]] = []
        for anchor in anchor_candidates:
            va = np.array(anchor.pos) - label_center
            na = np.linalg.norm(va)
            align = float(np.dot(unit, va / na)) if na > 0 else -1.0
            scored_anchors.append((align, anchor))

        scored_anchors.sort(key=lambda x: x[0], reverse=True)

        for _, anchor in scored_anchors:
            if ol.anchor.anchor_type == anchor.anchor_type:
                break

            anchor_offset_from_center = np.array(anchor.pos) - label_center
            fixed_anchor_pos = np.array(ol.anchor.pos)
            new_center = fixed_anchor_pos - anchor_offset_from_center
            pw, ph = label_wh(ol.pad_bbox_corners)
            new_corners = bbox_corners(*new_center, pw/2, ph/2)
            new_anchor = Anchor(anchor.anchor_type, *ol.anchor.pos, *new_corners)

            tmp_ol = copy.deepcopy(ol)
            tmp_ol.update_position(*new_center, new_anchor)

            tmp_pad_poly = Polygon(tmp_ol.pad_bbox_corners)
            tmp_exp_poly = Polygon(tmp_ol.exp_bbox_corners)

            if tmp_exp_poly.intersects(alpha_polygon):
                continue # overlaps graph

            if any(tmp_exp_poly.intersects(Polygon(l[0].pad_bbox_corners)) for l in label_candidates.values() if l):
                continue # overlaps with existing label
    
            if any(tmp_exp_poly.intersects(Polygon(oc.exp_bbox_corners)) for oc in overflow_candidates.values() if oc.node_id != tmp_ol.node_id):
                continue # overlaps another overflow label
 
            if any(
                tmp_pad_poly.intersects(LineString([G.nodes[oc.node_id]['pos'], oc.anchor.pos])) 
                for oc in overflow_candidates.values() if oc.node_id != tmp_ol.node_id
            ):
                continue # overlaps with a binder of another overflow label

            overflow_candidates[lid] = tmp_ol
            break

    return overflow_candidates

def unbounded_overflow_labels(
        G: nx.Graph,
        label_candidates: Dict[int, List[LabelCandidate]],
        overflow_candidates: Dict[int, LabelCandidate],
        alpha_shape: List[int],
        cfg: Config
    ) -> Tuple[Dict[int, List[Tuple[LabelCandidate, float]]], Dict[int, LabelCandidate], float, float]:
    '''
    Place remaining overflow labels in the graph exterior.

    Parameters
    ----------
    G : nx.Graph
        graph containing positions
    label_candidates : Dict[int, List[LabelCandidate]]
        already placed label candidates
    overflow_candidates : Dict[int, LabelCandidate]
        all overflow candidates
    alpha_shape : List[int]
        boundary walk of the alpha shape
    cfg: Config
        configuration

    Returns
    -------
    grid_candidates : Dict[int, List[Tuple[LabelCandidate, float]]]
        valid grid candidates
    overflow_candidates : Dict[int, LabelCandidate]
        updated overflow candidates
    grid_duration : float
        runtime for building the grid
    hungarian_duration : float
        runtime of the hungarian solver
    '''
    alpha_pos = [G.nodes[nid]['pos'] for nid in alpha_shape]
    alpha_N = len(alpha_shape)
    alpha_polygon = Polygon(alpha_pos)
    centroid = (alpha_polygon.centroid.x, alpha_polygon.centroid.y)
    placed_union = unary_union([
        Polygon(candidate.exp_bbox_corners)
        for candidates in label_candidates.values()
        for candidate in candidates
    ]) if label_candidates else Polygon()
    node_angles = sorted(
        [(angle_from_centroid(centroid, G.nodes[nid]['pos']), nid) for nid in alpha_shape],
        key=lambda x: x[0]
    )
    gaps = []
    for i in range(alpha_N):
        a_left,  node_left  = node_angles[i]
        a_right, node_right = node_angles[(i + 1) % alpha_N]
        gaps.append({
            'a_left': a_left, 
            'a_right': a_right,
            'gap_size': angular_gap_between(a_left, a_right),
            'node_left': node_left, 
            'node_right': node_right,
            'assigned': []
        })

    # assign unplaced overflow labels to their natural gap
    unplaced_overflow = {
        lid: ol 
        for lid, ol in overflow_candidates.items() 
        if ol.anchor.anchor_type == AnchorType.O
    }
    placed_overflow = {
        lid: ol 
        for lid, ol in overflow_candidates.items() 
        if ol.anchor.anchor_type != AnchorType.O
    }
    for ol in unplaced_overflow.values():
        outward_angle = angle_from_centroid(centroid, ol.center)
        best_gap = max(
            (g for g in gaps if angular_gap_between(g['a_left'], outward_angle) <= g['gap_size']),
            key=lambda g: g['gap_size'],
            default=max(gaps, key=lambda g: g['gap_size'])
        )
        best_gap['assigned'].append(ol.node_id)

    start_time = time.perf_counter()
    grid_candidates = grid_overflow_candidates(G, label_candidates, overflow_candidates, gaps, centroid, alpha_polygon, placed_union, placed_overflow, cfg)
    grid_duration = (time.perf_counter() - start_time) * 1000
    
    start_time = time.perf_counter()
    assignment = hungarian_solver(G, sorted(grid_candidates.keys()), grid_candidates, cfg)
    for lid, chosen in assignment.items():
        overflow_candidates[lid] = chosen[0]
    hungarian_duration = (time.perf_counter() - start_time) * 1000

    return grid_candidates, _adjust_anchors(G, label_candidates, overflow_candidates, unplaced_overflow.keys(), alpha_polygon), grid_duration, hungarian_duration

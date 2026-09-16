import copy
import numpy as np

from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment

from node_labeling.utils.config import Config
from node_labeling.utils.geometry import *
from node_labeling.models.label_candidate import LabelCandidate

def _build_cost_matrix_with_penalties(
    label_ids: List[int],
    cand_by_i: List[List[Tuple[LabelCandidate, float]]],
    top_k: int,
    penalty_table: Dict[Tuple[int, int], float],
    cfg: Config
) -> np.ndarray:
    '''
    Build the cost matrix for all pairs of label candidates, combining the
    static base cost and penalty.

    Parameters
    ----------
    label_ids : List[int]
        unplaced labels
    cand_by_i : List[List[Tuple[LabelCandidate, float]]]
        each candidate with an assigned cost given by label id
    top_k : int
        top k candidates to keep
    penalty_table : Dict[Tuple[int, int], float]
        penalty table iteratively build from conflicts
    cfg: Config
        configuration

    Returns
    -------
    cost_matrix : np.ndarray
        cost for all pairs of label candidates
    '''
    n = len(label_ids)
    matrix = np.full((n, n * top_k), cfg.w_miss)
    for i in range(n):
        for k, cand_i in enumerate(cand_by_i[i]):
            base_cost = cand_i[1]
            injected = penalty_table.get((i, k), 0.0)
            matrix[i, i * top_k + k] = base_cost + injected

    return matrix

def _find_conflicting_pairs(
    G: nx.Graph,
    label_ids: List[int],
    assignment: Dict[int, Optional[Tuple[LabelCandidate, float]]],
    cfg: Config
) -> List[Tuple[int, int]]:
    '''
    Find every (lid_a, lid_b) pair where the assigned candidates overlap.
    
    Parameters
    ----------
    G : nx.Graph
        graph containing positions
    label_ids : List[int]
        unplaced labels
    assignment : Dict[int, Optional[Tuple[LabelCandidate, float]]]
        candidate assigned for each label id
    cfg : Config
        configuration

    Returns
    -------
    conflicts : List[Tuple[int, int]]
        conflicting pairs in the current assignment
    '''
    conflicts = []
    ids_with_cand = [lid for lid in label_ids if assignment.get(lid) is not None]
    for a in range(len(ids_with_cand)):
        for b in range(a + 1, len(ids_with_cand)):
            lid_a = ids_with_cand[a]
            lid_b = ids_with_cand[b]
            if _conflict_cost(G, assignment[lid_a], assignment[lid_b], cfg) > 0.0:
                conflicts.append((lid_a, lid_b))

    return conflicts

def _conflict_cost(G : nx.Graph, a: Tuple[LabelCandidate, float], b: Tuple[LabelCandidate, float], cfg: Config) -> float:
    '''
    Conflict cost for two label candidates.

    Parameters
    ----------
    G : nx.Graph
        graph containing positions
    a: Tuple[LabelCandidate, float]
        label candidate a 
    b: Tuple[LabelCandidate, float]
        label candidate b
    cfg: Config
        configuration
    
    Returns
    -------
    cost : float
        conflict cost between candidates a and b
    '''
    pad_poly_a = Polygon(a[0].pad_bbox_corners)
    exp_poly_a = Polygon(a[0].exp_bbox_corners)
    anchor_a = Point(a[0].anchor.pos)
    binder_a = LineString([a[0].anchor.pos, G.nodes[a[0].node_id]['pos']])
    pad_poly_b = Polygon(b[0].pad_bbox_corners)
    exp_poly_b = Polygon(b[0].exp_bbox_corners)
    anchor_b = Point(b[0].anchor.pos)
    binder_b = LineString([b[0].anchor.pos, G.nodes[b[0].node_id]['pos']])
    cost = 0.0

    if pad_poly_a.intersects(pad_poly_b):
        overlap_area = pad_poly_a.intersection(pad_poly_b).area
        cost += cfg.w_tight_overlap + cfg.w_overlap * (overlap_area / 200.0)

    if exp_poly_a.intersects(exp_poly_b):
        enc_area = exp_poly_a.intersection(exp_poly_b).area
        cost += cfg.w_padding * (enc_area / 400.0)

    if binder_a.intersects(pad_poly_b):
        cost += cfg.w_binder_intersect
    if binder_b.intersects(pad_poly_a):
        cost += cfg.w_binder_intersect

    if binder_a.crosses(binder_b):
        cost += cfg.w_binder_cross

    if anchor_a.distance(binder_b) < 0.1:
        cost += cfg.w_binder_intersect
    
    if anchor_b.distance(binder_a) < 0.1:
        cost += cfg.w_binder_intersect

    return cost

def _greedy_assignment(
    label_ids: List[int],
    candidates: Dict[int, List[Tuple[LabelCandidate, float]]],
) -> Dict[int, Optional[Tuple[LabelCandidate, float]]]:
    '''
    Greedy fallback if the hungarian solver is too expensive.

    Parameters
    ----------
    label_ids : List[int]
        unplaced labels
    candidates : Dict[int, List[Tuple[LabelCandidate, float]]]
        candidates with an assigned cost
    
    Returns
    -------
    assignment : Dict[int, Optional[Tuple[LabelCandidate, float]]]
        chosen grid candidates by label id
    '''
    assignment: Dict[int, Optional[dict]] = {}
    placed_bboxes: List = []
    order = sorted(label_ids, key=lambda lid: len(candidates.get(lid, [])))
    for lid in order:
        chosen = None
        for cand in candidates.get(lid, []):
            cand_poly = Polygon(bbox_corners(cand[0].pad_bbox_corners))
            if not any(cand_poly.intersects(pb) for pb in placed_bboxes):
                chosen = cand
                break

        if chosen is None and candidates.get(lid):
            chosen = candidates[lid][0]
        
        if chosen:
            chosen_poly = Polygon(bbox_corners(chosen[0].pad_bbox_corners))
            placed_bboxes.append(chosen_poly)
        
        assignment[lid] = chosen

    return assignment

def _repair_overlaps(
        G: nx.Graph,
        label_ids: List[int], 
        candidates: Dict[int, List[Tuple[LabelCandidate, float]]], 
        assignment: Dict[int, Optional[Tuple[LabelCandidate, float]]], 
        cfg: Config
    ) -> Dict[int, Optional[Tuple[LabelCandidate, float]]]:
    '''
    Repair overlaps locally if they dont add much extra cost.

    Parameters
    ---------- 
    G : nx.Graph
        graph containing positions
    label_ids : List[int]
        unplaced labels
    candidates : Dict[int, List[Tuple[LabelCandidate, float]]]
        candidates with an assigned cost
    assignment : Dict[int, Optional[Tuple[LabelCandidate, float]]]
        chosen grid candidates by label id
    cfg : Config
        configuration

    Returns
    -------
    assignment : Dict[int, Optional[Tuple[LabelCandidate, float]]]
        refined adjustment
    '''
    for _ in range(cfg.max_hungarian_fixes):
        made_progress = False
        for lid_a in label_ids:
            ca = assignment.get(lid_a)
            if not ca:
                continue

            for lid_b in label_ids:
                if lid_a == lid_b or not assignment.get(lid_b): 
                    continue
                cb = assignment[lid_b]
                
                if _conflict_cost(G, ca, cb, cfg) > 0:
                    best_alt = ca
                    # Wrap the generator in list()
                    min_conflict = sum(_conflict_cost(G, ca, assignment[o], cfg) for o in label_ids if o != lid_a and assignment.get(o))

                    for alt in candidates.get(lid_a, []):
                        new_conflict = sum(_conflict_cost(G, alt, assignment[o], cfg) for o in label_ids if o != lid_a and assignment.get(o))
                        
                        if new_conflict < min_conflict:
                            min_conflict = new_conflict
                            best_alt = alt
                            made_progress = True
                    
                    assignment[lid_a] = best_alt
        
        if not made_progress: 
            break

    return assignment

def hungarian_solver(
    G: nx.Graph,
    label_ids: List[int],
    candidates: Dict[int, List[Tuple[LabelCandidate, float]]],
    cfg: Config
) -> Dict[int, Optional[Tuple[LabelCandidate, float]]]:
    '''
    Iterative hungarian solver for an initial global assignment.

    Parameters
    ----------
    label_ids : List[int]
        unplaced labels
    candidates : Dict[int, List[Tuple[LabelCandidate, float]]]
        candidates with an assigned cost
    cfg : Config
        configuration

    Returns
    -------
    assignment : Dict[int, Optional[Tuple[LabelCandidate, float]]]
        chosen grid candidates by label id
    '''
    if not label_ids:
        return {}
    
    try:
        if len(label_ids) > 200:
            raise ImportError("too large")

        cand_by_i = [candidates.get(lid, []) for lid in label_ids]
        penalty_table: Dict[Tuple[int, int], float] = {}
        best_assignment: Dict[int, Optional[Dict]] = {}
        min_actual_cost = float('inf')

        for iteration in range(cfg.max_hungarian_iterations):
            matrix = _build_cost_matrix_with_penalties(label_ids, cand_by_i, cfg.top_k, penalty_table, cfg)
            row_i, col_i = linear_sum_assignment(matrix)
            current_assignment: Dict[int, Tuple[LabelCandidate, float]] = {}
            current_chosen_idx: Dict[int, int] = {}
            
            current_base_sum = 0.0
            for i, col in zip(row_i, col_i):
                lid = label_ids[i]
                k = col - i * cfg.top_k
                cands = cand_by_i[i]

                if 0 <= k < len(cands) and matrix[i, col] < cfg.w_miss:
                    current_assignment[lid] = cands[k]
                    current_chosen_idx[lid] = k
                    current_base_sum += cands[k][1]
                else:
                    current_assignment[lid] = None
                    current_chosen_idx[lid] = -1
                    current_base_sum += cfg.w_miss

            conflict_pairs = _find_conflicting_pairs(G, label_ids, current_assignment, cfg)
            current_conflict_sum = sum(
                _conflict_cost(G, current_assignment[la], current_assignment[lb], cfg)
                for la, lb in conflict_pairs
            )

            total_actual_cost = current_base_sum + current_conflict_sum
            
            if total_actual_cost < min_actual_cost:
                min_actual_cost = total_actual_cost
                best_assignment = copy.copy(current_assignment)

            if not conflict_pairs:
                break

            for lid_a, lid_b in conflict_pairs:
                a, b = label_ids.index(lid_a), label_ids.index(lid_b)
                ka, kb = current_chosen_idx[lid_a], current_chosen_idx[lid_b]
                
                pair_cost = _conflict_cost(G, current_assignment[lid_a], current_assignment[lid_b], cfg)
                scale = cfg.iter_penalty_multiplier ** iteration
                
                if ka >= 0:
                    penalty_table[(a, ka)] = penalty_table.get((a, ka), 0.0) + (pair_cost * scale)
                if kb >= 0:
                    penalty_table[(b, kb)] = penalty_table.get((b, kb), 0.0) + (pair_cost * scale)

        assignment = best_assignment

    except ImportError:
        assignment = _greedy_assignment(label_ids, candidates)

    return _repair_overlaps(G, label_ids, candidates, assignment, cfg)

import time
import warnings
import shapely

from dataclasses import replace
from typing import Dict, List, Tuple

from shapely import unary_union
from shapely.geometry import Polygon

from node_labeling.utils.config import Config
from node_labeling.v2.models.candidate import Candidate
from node_labeling.v2.overflow.binders import Layout
from node_labeling.v2.overflow.grid import _generate_candidates, grid_overflow_candidates
from node_labeling.v2.overflow.hungarian import hungarian_solver
from node_labeling.v2.utils.geometry import angle_from_centroid, angular_gap_between

def unbounded_overflow_labels(
        layout: Layout,
        overflow: Dict[int, Candidate],
        alpha_shape: List[int],
        cfg: Config
) -> Tuple[Dict[int, List[Tuple[Candidate, float]]], Dict[int, Candidate], float, float]:
    '''
    Place the remaining overflow labels outside the diagram: grid positions per
    label, then the iterative Hungarian assignment. The binder attachment comes
    with each position (nearest point of the dot ring), so v1's corner swap
    (`_adjust_anchors`) is gone.

    Parameters
    ----------
    layout : Layout
        fixed context
    overflow : Dict[int, Candidate]
        overflow labels, unplaced ones without binder
    alpha_shape : List[int]
        boundary walk of the alpha shape
    cfg : Config
        configuration

    Returns
    -------
    grid_candidates : Dict[int, List[Tuple[Candidate, float]]]
        scored grid positions per label
    overflow : Dict[int, Candidate]
        overflow labels, placed where possible
    grid_duration : float
        runtime of the grid in ms
    hungarian_duration : float
        runtime of the assignment in ms
    '''
    alpha_polygon = Polygon([layout.node_pos(n) for n in alpha_shape])
    shapely.prepare(alpha_polygon)
    centroid = (alpha_polygon.centroid.x, alpha_polygon.centroid.y)
    placed_union = unary_union([c.exp for c in layout.direct]) if layout.direct else Polygon()
    shapely.prepare(placed_union)

    node_angles = sorted((angle_from_centroid(centroid, layout.node_pos(n)), n) for n in alpha_shape)
    gaps = []
    for i in range(len(node_angles)):
        a_left, _ = node_angles[i]
        a_right, _ = node_angles[(i + 1) % len(node_angles)]
        gaps.append({'a_left': a_left, 'a_right': a_right, 'gap_size': angular_gap_between(a_left, a_right), 'assigned': []})

    unplaced = [lid for lid, ol in overflow.items() if ol.binder is None]
    for lid in unplaced:
        ol = overflow[lid]
        outward = angle_from_centroid(centroid, ol.node_pos)
        gap = max((g for g in gaps if angular_gap_between(g['a_left'], outward) <= g['gap_size']),
                  key=lambda g: g['gap_size'], default=max(gaps, key=lambda g: g['gap_size']))
        if ol.node_id not in gap['assigned']:
            gap['assigned'].append(ol.node_id)

    start = time.perf_counter()
    grid = grid_overflow_candidates(layout, overflow, gaps, centroid, alpha_polygon, placed_union, cfg)
    for lid in [lid for lid, cands in grid.items() if not cands]:
        relaxed = cfg
        for _ in range(3):
            relaxed = replace(relaxed, max_label_dist=relaxed.max_label_dist * 2)
            grid[lid] = _generate_candidates(lid, layout, centroid, alpha_polygon, placed_union, overflow, relaxed)
            if grid[lid]:
                break
    grid_duration = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    assignment = hungarian_solver(sorted(grid), grid, cfg)
    for lid, chosen in assignment.items():
        if chosen is None:
            warnings.warn(f'Label {lid} ({overflow[lid].text!r}) could not be placed and is drawn at its node')
            continue
        overflow[lid] = chosen[0]
    hungarian_duration = (time.perf_counter() - start) * 1000
    return grid, overflow, grid_duration, hungarian_duration

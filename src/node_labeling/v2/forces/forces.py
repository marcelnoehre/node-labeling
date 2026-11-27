import numpy as np
import shapely

from typing import Dict, List
from shapely import unary_union
from shapely.geometry import Polygon, Point
from shapely.ops import nearest_points

from node_labeling.models.label_type import LabelType
from node_labeling.utils.config import Config
from node_labeling.utils.shapely_utils import as_points
from node_labeling.v2.models.candidate import Candidate
from node_labeling.v2.overflow.binders import Layout, attach, binder_valid

def cross2d(a: np.ndarray, b: np.ndarray) -> float:
    '''
    Z-component of the 3D cross product of two 2D vectors.
    '''
    return a[0] * b[1] - a[1] * b[0]

def optimize_overflow_labels(
    layout: Layout,
    overflow: Dict[int, Candidate],
    unbounded_overflow_labels: List[int],
    alpha_shape: List[int],
    cfg: Config
) -> Dict[int, Candidate]:
    '''
    Refines overflow label positions using force-directed optimization
    calibrated for small-scale coordinate systems (v1's forces on shapes; the
    binder attachment is re-picked after every move, see `attach`).

    Soft forces use convex hulls of the pads; the guards use the exact shapes.

    Parameters
    ----------
    layout : Layout
        fixed context
    overflow : Dict[int, Candidate]
        overflow labels
    unbounded_overflow_labels : List[int]
        overflow labels that are placed in the graph's exterior
    alpha_shape : List[int]
        boundary walk of the drawing
    cfg : Config
        configuration

    Returns
    -------
    overflow : Dict[int, Candidate]
        refined overflow labels
    '''
    alpha_polygon = Polygon([layout.node_pos(n) for n in alpha_shape])
    drawing_centroid = np.array([alpha_polygon.centroid.x, alpha_polygon.centroid.y])
    placed_union = unary_union([c.pad for c in layout.direct] + [alpha_polygon])
    placed_ink_union = unary_union([c.ink for c in layout.direct] + [alpha_polygon])
    shapely.prepare(placed_union)
    shapely.prepare(placed_ink_union)
    ink_polys = [c.ink for c in layout.direct]
    ink_hulls = [ink.convex_hull for ink in ink_polys]
    alpha_nodes = [Point(layout.node_pos(n)).buffer(0.15) for n in alpha_shape]

    movable = [lid for lid in unbounded_overflow_labels if overflow[lid].binder is not None]

    for i in range(cfg.force_iterations):
        pending: Dict[int, Candidate] = {}
        movable.sort(key=lambda lid: Point(overflow[lid].position).distance(placed_union))

        for lid in movable:
            ol = overflow[lid]
            node_pos = np.array(ol.node_pos)
            current_center = np.array(ol.position)
            hull = ol.pad_hull
            ex0, ey0, ex1, ey1 = ol.exp.bounds
            ew, eh = ex1 - ex0, ey1 - ey0
            force_vector = np.array([0.0, 0.0])
            radial_vec = current_center - drawing_centroid
            radial_unit = radial_vec / (np.linalg.norm(radial_vec) + 1e-6)
            tangent_vec = np.array([-radial_unit[1], radial_unit[0]])

            # repulsion from labels at their nodes and boundary nodes
            threshold = max(ew, eh)
            for ink in ink_hulls + alpha_nodes:
                p1, p2 = nearest_points(ink, hull)
                dist_to_ink = np.hypot(p2.x - p1.x, p2.y - p1.y)
                if dist_to_ink < threshold:
                    vec_to_ink = np.array([ink.centroid.x, ink.centroid.y]) - drawing_centroid
                    side = cross2d(radial_unit, vec_to_ink / (np.linalg.norm(vec_to_ink) + 1e-6))
                    slide_dir = tangent_vec if side < 0 else -tangent_vec
                    force_vector += slide_dir * (threshold - dist_to_ink) * cfg.w_inner_proximity

            # tangential shift away from other overflow labels
            threshold = max(ew, eh) * 0.5
            for o_lid in movable:
                if lid == o_lid:
                    continue
                o_ol = overflow[o_lid]
                p1, p2 = nearest_points(hull, o_ol.pad_hull)
                dist = np.hypot(p2.x - p1.x, p2.y - p1.y)
                if dist < threshold:
                    vec_to_other = np.array(o_ol.position) - drawing_centroid
                    side = cross2d(radial_unit, vec_to_other / (np.linalg.norm(vec_to_other) + 1e-6))
                    slide_dir = tangent_vec if side < 0 else -tangent_vec
                    force_vector += slide_dir * (threshold - dist) * cfg.w_global_proximity

            # spring pulling the binder short
            anchor_pos = np.array(ol.binder.end)
            binder = ol.binder.line
            crossings = as_points(alpha_polygon.boundary.intersection(binder)) if binder.length > 0 else []
            if crossings:
                anchor_pt = Point(anchor_pos)
                intersection_pos = np.array(min(crossings, key=lambda p: p.distance(anchor_pt)).coords[0])
            else:
                intersection_pos = node_pos
            gap_vec = anchor_pos - intersection_pos
            dist = np.linalg.norm(gap_vec)
            displacement = dist - cfg.min_binder_length + 0.1
            force_vector -= gap_vec / (dist + 1e-6) * np.sign(displacement) * displacement ** 2 * cfg.w_spring

            # binder repulsion
            binder_threshold = 0.2
            obstacles = [(p, p.centroid) for p in ink_hulls + alpha_nodes]
            obstacles += [(o.pad_hull, o.pad_hull.centroid) for o_lid, o in overflow.items() if o_lid != lid and o.binder is not None]
            for obstacle, obstacle_centroid in obstacles:
                dist_binder = binder.distance(obstacle)
                if dist_binder < binder_threshold:
                    vec_to_obs = np.array([obstacle_centroid.x, obstacle_centroid.y]) - drawing_centroid
                    side = cross2d(radial_unit, vec_to_obs / (np.linalg.norm(vec_to_obs) + 1e-6))
                    slide_dir = tangent_vec if side < 0 else -tangent_vec
                    force_vector += slide_dir * (binder_threshold - dist_binder) * cfg.w_binder_dodge

            # half plane
            if ol.label_type in (LabelType.INTENT, LabelType.EXTENT):
                _, min_y, _, max_y = ol.pad.bounds
                intent = ol.label_type == LabelType.INTENT
                if (intent and min_y < node_pos[1]) or (not intent and max_y > node_pos[1]):
                    side = cross2d(radial_unit, np.array([0.0, 1.0]))
                    sign = 1.0 if intent else -1.0
                    slide_dir = tangent_vec if side * sign > 0 else -tangent_vec
                    infringement = (node_pos[1] - min_y) if intent else (max_y - node_pos[1])
                    force_vector += slide_dir * (infringement + 0.1) * cfg.w_half_plane

            cooling = 1.0 - (i / cfg.force_iterations)
            step_size = cfg.force_step_size * cooling
            force_mag = np.linalg.norm(force_vector)
            delta = force_vector * step_size
            if force_mag > 1.0:
                delta = force_vector / force_mag * step_size
            if np.linalg.norm(delta) < cfg.force_tolerance:
                continue

            proposed = ol.moved(current_center + delta)
            proposed = proposed.moved(proposed.position, attach(proposed))
            if not _valid(layout, lid, ol, proposed, overflow, pending, movable, alpha_polygon,
                          drawing_centroid, placed_union, placed_ink_union, ink_polys, cfg):
                continue
            pending[lid] = proposed

        if not pending:
            return overflow
        overflow.update(pending)

    return overflow

def _valid(
        layout: Layout,
        lid: int,
        current: Candidate,
        proposed: Candidate,
        overflow: Dict[int, Candidate],
        pending: Dict[int, Candidate],
        movable: List[int],
        alpha_polygon: Polygon,
        drawing_centroid: np.ndarray,
        placed_union,
        placed_ink_union,
        ink_polys: list,
        cfg: Config
) -> bool:
    '''
    Guards of a proposed move (v1's guards on shapes).

    Returns
    -------
    valid : bool
        whether the move is allowed
    '''
    node_pos = np.array(proposed.node_pos)
    binder = proposed.binder
    dist_proposed = alpha_polygon.distance(proposed.pad)
    if dist_proposed < cfg.min_binder_length and dist_proposed < alpha_polygon.distance(current.pad):
        return False # < minimal binder length
    if np.dot(node_pos - drawing_centroid, np.array(proposed.position) - node_pos) < -0.5:
        return False # keep labels outside
    if proposed.pad.intersects(placed_union):
        return False # overlaps the drawing or a label at its node
    if any(binder.line.intersects(ink) for ink in ink_polys):
        return False # binder through a label
    end = Point(binder.end)
    if end.intersects(placed_ink_union):
        return False
    for o_lid in movable:
        if o_lid == lid:
            continue
        other = pending.get(o_lid, overflow[o_lid])
        if proposed.pad.intersects(other.pad):
            return False # overlaps another overflow label
        if proposed.pad.intersects(other.binder.line):
            return False # overlaps a binder
        if Point(other.binder.end).distance(binder.line) < 0.15:
            return False # binder too close to another attachment
        if end.distance(other.binder.line) < 0.15:
            return False # attachment too close to another binder
        if end.distance(other.pad) < 0.15:
            return False # attachment too close to another overflow label
    view = {**overflow, **pending}
    return binder_valid(layout, lid, proposed, binder, view, soft=True)

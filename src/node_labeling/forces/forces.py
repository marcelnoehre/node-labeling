import copy
import numpy as np
import networkx as nx

from typing import Dict, List
from shapely import unary_union
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import nearest_points

from node_labeling.models.label_type import LabelType
from node_labeling.models.label_candidate import LabelCandidate
from node_labeling.utils.geometry import *
from node_labeling.utils.config import Config

def cross2d(a: np.ndarray, b: np.ndarray) -> float:
    '''
    Z-component of the 3D cross product of two 2D vectors.
    '''
    return a[0] * b[1] - a[1] * b[0]

def optimize_overflow_labels(
    G: nx.Graph, 
    label_candidates: Dict[int, List[LabelCandidate]], 
    overflow_candidates: Dict[int, LabelCandidate], 
    unbounded_overflow_labels: List[int],
    alpha_shape: List[int],
    cfg: Config
) -> Dict[int, LabelCandidate]:
    '''
    Refines overflow label positions using force-directed optimization 
    calibrated for small-scale coordinate systems.
    
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
    alpha_shape: List[int]
        boundary walk of the drawing
    cfg : Config
        configuration

    Returns
    -------
    overflow_candidates: Dict[int, LabelCandidate]
        refined overflow candidates
    '''
    alpha_polygon = Polygon([G.nodes[n]['pos'] for n in alpha_shape])
    drawing_centroid = np.array([alpha_polygon.centroid.x, alpha_polygon.centroid.y])
    placed_polys = [
        Polygon(candidate.pad_bbox_corners)
        for candidates in label_candidates.values()
        for candidate in candidates
    ]
    placed_union = unary_union(placed_polys + [alpha_polygon])
    ink_polys = [
        Polygon(candidate.ink_bbox_corners)
        for candidates in label_candidates.values()
        for candidate in candidates
    ]
    placed_ink_union = unary_union(ink_polys + [alpha_polygon])
    alpha_nodes = [
        Point(G.nodes[nid]['pos']).buffer(0.15)
        for nid in alpha_shape    
    ]

    # iterative force refinement
    for i in range(cfg.force_iterations):
        pending_updates = {}
        unbounded_overflow_labels.sort(key=lambda lid: Point(overflow_candidates[lid].center).distance(placed_union))

        for lid in unbounded_overflow_labels:
            ol = overflow_candidates[lid]
            node_pos = np.array(G.nodes[ol.node_id]['pos'])
            current_center = np.array(ol.center)
            pad_bbox_poly = Polygon(ol.pad_bbox_corners)
            ew, eh = label_wh(ol.exp_bbox_corners)
            force_vector = np.array([0.0, 0.0])

            # repulsion from label candidates
            threshold = max(ew, eh)
            for ink in ink_polys + alpha_nodes:
                p1, p2 = nearest_points(ink, pad_bbox_poly)
                dist_to_ink = np.linalg.norm(np.array([p2.x - p1.x, p2.y - p1.y]))

                if dist_to_ink < threshold:
                    radial_vec = current_center - drawing_centroid
                    radial_unit = radial_vec / (np.linalg.norm(radial_vec) + 1e-6)
                    tangent_vec = np.array([-radial_unit[1], radial_unit[0]])

                    ink_center = np.array([ink.centroid.x, ink.centroid.y])
                    vec_to_ink = ink_center - drawing_centroid
                    side = cross2d(radial_unit, vec_to_ink / (np.linalg.norm(vec_to_ink) + 1e-6))
                    slide_dir = tangent_vec if side < 0 else -tangent_vec

                    mag = (threshold - dist_to_ink) * cfg.w_inner_proximity
                    force_vector += slide_dir * mag

            # tangential shift
            threshold = max(ew, eh) * 0.5
            for o_lid in unbounded_overflow_labels:
                if lid == o_lid:
                    continue
                o_ol = overflow_candidates[o_lid]
                other_poly = Polygon(o_ol.pad_bbox_corners)
                p1, p2 = nearest_points(pad_bbox_poly, other_poly)
                dist = np.linalg.norm(np.array([p2.x - p1.x, p2.y - p1.y]))
                if dist < threshold:
                    radial_vec = current_center - drawing_centroid
                    radial_unit = radial_vec / (np.linalg.norm(radial_vec) + 1e-6)
                    tangent_vec = np.array([-radial_unit[1], radial_unit[0]])

                    vec_to_other = np.array(o_ol.center) - drawing_centroid
                    side = cross2d(radial_unit, vec_to_other / (np.linalg.norm(vec_to_other) + 1e-6))
                    slide_dir = tangent_vec if side < 0 else -tangent_vec
                    
                    mag = (threshold - dist) * cfg.w_global_proximity
                    force_vector += slide_dir * mag

            # spring force pulling binder
            node_pos = np.array(G.nodes[ol.node_id]['pos'])
            anchor_pos = np.array(ol.anchor.pos)
            binder = LineString([node_pos, anchor_pos])

            intersect_result = alpha_polygon.boundary.intersection(binder)
            if intersect_result.is_empty:
                closest_pt = None
            elif isinstance(intersect_result, Point):
                closest_pt = intersect_result
            else:
                anchor_pt = Point(anchor_pos)
                closest_pt = min(intersect_result.geoms, key=lambda p: p.distance(anchor_pt))

            intersection_pos = np.array(closest_pt.coords[0])

            gap_vec = anchor_pos - intersection_pos
            dist = np.linalg.norm(gap_vec) 

            displacement = dist - cfg.min_binder_length + 0.1
            mag = (displacement**2) * cfg.w_spring
               
            unit_dir = gap_vec / (dist + 1e-6)
            force_vector -= unit_dir * mag

            # binder repulsion
            current_anchor_pt = ol.anchor.pos
            binder_line = LineString([current_anchor_pt, node_pos])

            radial_vec = current_center - drawing_centroid
            radial_unit = radial_vec / (np.linalg.norm(radial_vec) + 1e-6)
            tangent_vec = np.array([-radial_unit[1], radial_unit[0]])
            
            binder_threshold = 0.2

            def _apply_binder_repulsion(obstacle_poly: Polygon, obstacle_centroid_pt: Point):
                '''
                Push binder away from obstacles.

                Parameters
                ----------
                obstacle_poly: Polygon
                    polygon of obstacle
                
                obstacle_centroid_pt: Point
                    center of the obstacle
                '''
                nonlocal force_vector
                dist_binder = binder_line.distance(obstacle_poly)
                
                if dist_binder < binder_threshold:
                    obs_center = np.array([obstacle_centroid_pt.x, obstacle_centroid_pt.y])
                    vec_to_obs = obs_center - drawing_centroid
                    side = cross2d(radial_unit, vec_to_obs / (np.linalg.norm(vec_to_obs) + 1e-6))
                    slide_dir = tangent_vec if side < 0 else -tangent_vec
                    mag = (binder_threshold - dist_binder) * cfg.w_binder_dodge
                    force_vector += slide_dir * mag

            for ink_poly in ink_polys + alpha_nodes:
                _apply_binder_repulsion(ink_poly, ink_poly.centroid)

            for o_lid, other_ol in overflow_candidates.items():
                if o_lid == lid:
                    continue
                other_poly = Polygon(other_ol.pad_bbox_corners)
                _apply_binder_repulsion(other_poly, other_poly.centroid)

            if ol.label_type in [LabelType.INTENT, LabelType.EXTENT]:
                pw, ph = label_wh(ol.pad_bbox_corners)
                min_y = current_center[1] - ph/2
                max_y = current_center[1] + ph/2
                is_violating = (ol.label_type == LabelType.INTENT and min_y < node_pos[1]) or (ol.label_type == LabelType.EXTENT and max_y > node_pos[1])
                if is_violating:
                    up_vec = np.array([0.0, 1.0])
                    side = cross2d(radial_unit, up_vec)
                    direction_multiplier = 1.0 if ol.label_type == LabelType.INTENT else -1.0
                    slide_dir = tangent_vec if (side * direction_multiplier) > 0 else -tangent_vec
                    infringement = (node_pos[1] - min_y) if ol.label_type == LabelType.INTENT else (max_y - node_pos[1])
                    force_vector += slide_dir * (infringement + 0.1) * cfg.w_half_plane

            cooling = 1.0 - (i / cfg.force_iterations)
            step_size = cfg.force_step_size * cooling
            force_mag = np.linalg.norm(force_vector)
            delta = force_vector * step_size
            if force_mag > 1.0:
                delta = (force_vector / force_mag) * step_size

            proposed_center = current_center + delta

            # guards
            valid = True

            pw, ph = label_wh(ol.pad_bbox_corners)
            proposed_corners = bbox_corners(*proposed_center, pw/2, ph/2)
            proposed_poly = Polygon(proposed_corners)
            proposed_anchor = Anchor(ol.anchor.anchor_type, *proposed_center, *proposed_corners)
            proposed_binder = LineString([proposed_anchor.pos, node_pos])

            dist_current = alpha_polygon.distance(Polygon(ol.pad_bbox_corners))
            dist_proposed = alpha_polygon.distance(proposed_poly)
            if dist_proposed < cfg.min_binder_length:
                if dist_proposed < dist_current:
                    valid = False # < minimal binder length

            if valid:
                radial_dir = node_pos - drawing_centroid
                label_dir = proposed_center - node_pos
                if np.dot(radial_dir, label_dir) < -0.5:
                    valid = False # keep labels outside

            if valid:
                if proposed_poly.intersects(placed_ink_union):
                    valid = False # poly overlaps drawing

            if valid:
                for ink in ink_polys:
                    if proposed_binder.intersects(ink):
                        valid = False # binder overlaps label candidates
                        
            if valid:
                if Point(proposed_anchor.pos).intersects(placed_ink_union):
                    valid = False
        
            if valid:
                for o_lid in unbounded_overflow_labels:
                    if o_lid == lid:
                        continue
                
                    other = overflow_candidates[o_lid]
                    other_poly_pad = Polygon(other.pad_bbox_corners)
                    other_binder = LineString([other.anchor.pos, G.nodes[overflow_candidates[o_lid].node_id]['pos']])

                    if proposed_poly.intersects(other_poly_pad):
                        valid = False
                        break # poly overlaps overflow label

                    if proposed_poly.intersects(other_binder):
                        valid = False
                        break # poly overlaps binder

                    if Point(other.anchor.pos).distance(proposed_binder) < 0.15:
                        valid = False
                        break # binder too close to node

                    if Point(proposed_anchor.pos).distance(other_binder) < 0.15:
                        valid = False
                        break # anchor too close to binder

                    if Point(proposed_anchor.pos).distance(other_poly_pad) < 0.15:
                        valid = False
                        break # anchor too close to overflow label

            if valid:
                tmp_overflow_candidates = copy.deepcopy(overflow_candidates)
                for o_lid, other_ol in overflow_candidates.items():
                    if o_lid == lid:
                        continue
                
                    if o_lid in pending_updates and pending_updates[o_lid] is not None:
                        new_center, new_anchor = pending_updates[o_lid]
                        tmp_overflow_candidates[o_lid].update_position(*new_center, new_anchor)

                if not binding_line_valid(G, lid, proposed_anchor, label_candidates, tmp_overflow_candidates, overflow_candidates.keys(), [], True):
                    valid = False # invalid binding line

            if valid:
                pending_updates[lid] = (proposed_center, proposed_anchor)

        if not pending_updates:
            return overflow_candidates

        for lid, (new_center, new_anchor) in pending_updates.items():
            overflow_candidates[lid].update_position(*new_center, new_anchor)

    return overflow_candidates

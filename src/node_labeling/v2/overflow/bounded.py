import numpy as np
import shapely

from typing import Dict, List, Optional, Tuple

from shapely import unary_union
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid

from node_labeling.models.label_type import LabelType
from node_labeling.topology.faces import node_faces
from node_labeling.utils.constants import MM_PER_UNIT
from node_labeling.v2.label.slider import center_line
from node_labeling.v2.models.candidate import Candidate
from node_labeling.v2.overflow.binders import Layout, attach_many, binder_valid
from node_labeling.v2.utils.shapes import translate_many

FACE_STEP_MM: float = 0.5
REFINE_STEP_MM: float = 0.1

def _positions(space: BaseGeometry, candidate: Candidate, step: float, around: Optional[np.ndarray] = None) -> np.ndarray:
    '''
    Drawing points at which the exp shape of an overflow label lies inside the
    space and the label keeps its half plane (intent: center line at or above
    the node, extent: at or below).

    Parameters
    ----------
    space : BaseGeometry
        free part of a face (prepared)
    candidate : Candidate
        overflow label
    step : float
        grid step
    around : Optional[np.ndarray]
        if given, a grid of ±1 coarse step around this point instead of the whole space

    Returns
    -------
    positions : np.ndarray
        (n, 2) valid drawing points
    '''
    shape = candidate.shape
    exp = shape.overflow_exp
    ex0, ey0, ex1, ey1 = exp.bounds
    if around is None:
        sx0, sy0, sx1, sy1 = space.bounds
        xs = np.arange(sx0 - ex0, sx1 - ex1 + 1e-12, step)
        ys = np.arange(sy0 - ey0, sy1 - ey1 + 1e-12, step)
    else:
        r = FACE_STEP_MM / MM_PER_UNIT
        xs = around[0] + np.arange(-r, r + 1e-12, step)
        ys = around[1] + np.arange(-r, r + 1e-12, step)
    if not len(xs) or not len(ys):
        return np.empty((0, 2))
    P = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    node_y = candidate.node_pos[1]
    center = P[:, 1] + center_line(shape)
    if candidate.label_type == LabelType.INTENT:
        P = P[center >= node_y]
    elif candidate.label_type == LabelType.EXTENT:
        P = P[center <= node_y]
    if not len(P):
        return P
    return P[shapely.contains(space, translate_many(exp, P))]

def _fits(space: BaseGeometry, candidate: Candidate) -> bool:
    '''
    Parameters
    ----------
    space : BaseGeometry
        free part of a face
    candidate : Candidate
        overflow label

    Returns
    -------
    fits : bool
        whether the label fits into the space somewhere
    '''
    if space.is_empty:
        return False
    ex0, ey0, ex1, ey1 = candidate.shape.overflow_exp.bounds
    sx0, sy0, sx1, sy1 = space.bounds
    if ex1 - ex0 > sx1 - sx0 or ey1 - ey0 > sy1 - sy0 or candidate.shape.overflow_exp.area > space.area:
        return False
    return len(_positions(space, candidate, FACE_STEP_MM / MM_PER_UNIT)) > 0

def _free(layout: Layout, lid: int, candidate: Candidate, positions: np.ndarray, overflow: Dict[int, Candidate]) -> np.ndarray:
    '''
    Parameters
    ----------
    layout : Layout
        fixed context
    lid : int
        label id
    candidate : Candidate
        overflow label
    positions : np.ndarray
        (n, 2) drawing points
    overflow : Dict[int, Candidate]
        overflow labels (those with a binder are placed)

    Returns
    -------
    free : np.ndarray
        whether the exp shape at each position avoids placed overflow labels and other nodes
    '''
    exps = translate_many(candidate.shape.overflow_exp, positions)
    ok = np.ones(len(positions), dtype=bool)
    for o_lid, other in overflow.items():
        if o_lid != lid and other.binder is not None:
            ok &= ~shapely.intersects(exps, other.exp)
    obst = layout.obstacles
    others = obst.node_xy[np.array([n != candidate.node_id for n in obst.node_ids], dtype=bool)]
    if len(others):
        hit = shapely.intersects(exps[:, None], shapely.points(others)[None, :]).any(axis=1)
        ok &= ~hit
    return ok

def _find_valid_position(
        layout: Layout,
        overflow: Dict[int, Candidate],
        space: BaseGeometry,
        bounded_faces: List[List[int]],
        lid: int
) -> Optional[Candidate]:
    '''
    Position inside the space with the shortest valid binder (strict binder
    rules, at most one edge crossing, attachment away from the edges around the
    node), refined on a finer grid around the best coarse position.

    Parameters
    ----------
    layout : Layout
        fixed context
    overflow : Dict[int, Candidate]
        overflow labels
    space : BaseGeometry
        free part of the face
    bounded_faces : List[List[int]]
        bounded faces of the planar graph
    lid : int
        label to place

    Returns
    -------
    placed : Optional[Candidate]
        the label at its position with its binder, None if nothing is valid
    '''
    candidate = overflow[lid]
    edges = node_faces(layout.G, candidate.node_id, bounded_faces)

    def best(P: np.ndarray) -> Optional[Candidate]:
        if not len(P):
            return None
        P = P[_free(layout, lid, candidate, P, overflow)]
        cands = [candidate.moved(p) for p in P]
        binders = attach_many(cands)
        for i in np.argsort([b.length for b in binders], kind='stable'):
            if binder_valid(layout, lid, cands[i], binders[i], overflow, soft=False, edges=edges, max_edge_crossings=1):
                return cands[i].moved(cands[i].position, binders[i])
        return None

    coarse = best(_positions(space, candidate, FACE_STEP_MM / MM_PER_UNIT))
    if coarse is None:
        return None
    fine = best(_positions(space, candidate, REFINE_STEP_MM / MM_PER_UNIT, np.asarray(coarse.position)))
    if fine is not None and fine.binder.length < coarse.binder.length:
        return fine
    return coarse

def bounded_overflow_labels(
        layout: Layout,
        overflow: Dict[int, Candidate],
        bounded_faces: List[List[int]],
        centroids: List[Tuple[float, float]],
        alpha_shape: List[int]
) -> Dict[int, Candidate]:
    '''
    Place overflow labels inside the faces of the diagram (v1's order of faces
    and labels, on shapes): the largest free face first; per face the label of an
    incident node closest to the face centroid, else of any node; the position
    with the shortest valid binder. A label is not placed in a face if its node
    is closer to the outer boundary than to the face, or if the binder would be
    longer than that distance (then the exterior is better).

    Parameters
    ----------
    layout : Layout
        fixed context
    overflow : Dict[int, Candidate]
        overflow labels, unplaced ones without binder
    bounded_faces : List[List[int]]
        bounded faces of the planar graph
    centroids : List[Tuple[float, float]]
        centroid of each bounded face
    alpha_shape : List[int]
        boundary walk of the alpha shape

    Returns
    -------
    overflow : Dict[int, Candidate]
        overflow labels, those placed in a face with position and binder
    '''
    G = layout.G
    placed_union = unary_union([c.pad for c in layout.direct])
    space_map: Dict[int, BaseGeometry] = {}
    for fid, face in enumerate(bounded_faces):
        poly = Polygon([G.nodes[n]['pos'] for n in face])
        if not poly.is_valid:
            poly = make_valid(poly)
        space_map[fid] = poly.difference(placed_union)
        shapely.prepare(space_map[fid])

    fit_cache: Dict[Tuple[int, int], bool] = {}

    def fits(lid: int, fid: int) -> bool:
        if (lid, fid) not in fit_cache:
            fit_cache[(lid, fid)] = _fits(space_map[fid], overflow[lid])
        return fit_cache[(lid, fid)]

    pending = [lid for lid, c in overflow.items() if c.binder is None]
    face_to_labels: Dict[int, List[int]] = {}
    for lid in pending:
        for fid in space_map:
            if fits(lid, fid):
                face_to_labels.setdefault(fid, []).append(lid)

    alpha_boundary = Polygon([G.nodes[n]['pos'] for n in alpha_shape]).boundary
    placed: List[int] = []

    def block(lid: int, fid: int):
        face_to_labels[fid] = [l for l in face_to_labels.get(fid, []) if l != lid]

    while True:
        chosen_face, chosen_label = None, None
        for fid in sorted(space_map, key=lambda f: space_map[f].area, reverse=True):
            fitting = [lid for lid in face_to_labels.get(fid, []) if lid not in placed and fits(lid, fid)]
            if not fitting:
                continue
            centroid = np.array(centroids[fid])
            dist = lambda lid: np.linalg.norm(np.array(overflow[lid].node_pos) - centroid)
            incident = [lid for lid in fitting if overflow[lid].node_id in bounded_faces[fid]]
            chosen_label = min(incident or fitting, key=dist)
            chosen_face = fid
            break
        if chosen_face is None:
            break

        lid = chosen_label
        node = overflow[lid].node_pos
        dist_to_boundary = alpha_boundary.distance(Point(node))
        if dist_to_boundary < np.linalg.norm(np.array(node) - np.array(centroids[chosen_face])):
            block(lid, chosen_face)
            continue

        found = _find_valid_position(layout, overflow, space_map[chosen_face], bounded_faces, lid)
        if found is None or found.binder.length > dist_to_boundary:
            block(lid, chosen_face)
            continue

        placed.append(lid)
        overflow[lid] = found
        space_map[chosen_face] = space_map[chosen_face].difference(found.exp)
        shapely.prepare(space_map[chosen_face])
        for key in [k for k in fit_cache if k[1] == chosen_face]:
            del fit_cache[key]

    return overflow

import numpy as np
import networkx as nx
import shapely

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Hashable, List, Optional, Tuple

from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from node_labeling.models.label_type import LabelType
from node_labeling.utils.constants import OUTER_NODE_DIST, SHAPE_TOLERANCE_MM, MM_PER_UNIT
from node_labeling.utils.graph_lattice import GraphLattice
from node_labeling.v2.label.slider import Slider
from node_labeling.v2.models.label_shape import LabelShape
from node_labeling.v2.utils.shapes import translate_many

NEIGHBOR_EPS: float = 1e-6

@dataclass
class Obstacles:
    '''
    Fixed obstacles of the diagram (normalized units).

    Attributes
    ----------
    edges : np.ndarray
        edges of the planarized graph (LineStrings)
    edge_ends : List[Tuple[Hashable, Hashable]]
        end nodes of every edge, in the order of `edges`
    edge_tree : STRtree
        spatial index over `edges`
    node_ids : List[Hashable]
        lattice nodes
    node_xy : np.ndarray
        (n, 2) lattice node positions, in the order of `node_ids`
    hull : Optional[Polygon]
        polygon of the alpha shape (prepared), None if degenerate
    neighbors : Dict[Hashable, np.ndarray]
        positions of the lattice neighbors (parents and children) of every node
    '''
    edges: np.ndarray
    edge_ends: List[Tuple[Hashable, Hashable]]
    edge_tree: STRtree
    node_ids: List[Hashable]
    node_xy: np.ndarray
    hull: Optional[Polygon]
    neighbors: Dict[Hashable, np.ndarray]

def build_obstacles(G: nx.Graph, lattice: GraphLattice, alpha_shape: List[int]) -> Obstacles:
    '''
    Parameters
    ----------
    G : nx.Graph
        planarized graph containing normalized positions
    lattice : GraphLattice
        lattice structure
    alpha_shape : List[int]
        boundary walk of the alpha shape

    Returns
    -------
    obstacles : Obstacles
        edges, nodes, hull and neighbors
    '''
    edge_ends = list(G.edges())
    edges = np.array([LineString([G.nodes[u]['pos'], G.nodes[v]['pos']]) for u, v in edge_ends], dtype=object)
    node_ids = list(lattice.nodes)
    node_xy = np.array([G.nodes[n]['pos'] for n in node_ids], dtype=float).reshape(-1, 2)

    hull = None
    if alpha_shape:
        coords = np.array([G.nodes[n]['pos'] for n in alpha_shape])
        if np.ptp(coords[:, 0]) > 0:
            hull = Polygon(coords)
            if not hull.is_valid:
                hull = hull.buffer(0)
            shapely.prepare(hull)

    neighbors = {
        n: np.array([G.nodes[m]['pos'] for m in [*lattice.children(n), *lattice.parents(n)]], dtype=float).reshape(-1, 2)
        for n in node_ids
    }
    return Obstacles(edges, edge_ends, STRtree(edges), node_ids, node_xy, hull, neighbors)

class LabelRules:
    '''
    Hard constraints of one label, evaluated at many slider positions at once.
    The v1 filters on shapes instead of boxes:

    - half plane: intent labels keep their center line at or above the node,
      extent labels at or below (general: no restriction)
    - outer node (node within `OUTER_NODE_DIST` of the alpha shape boundary):
      the ink stays outside the alpha shape
    - nodes: no other node inside the exp shape
    - edges: no edge closer to the ink than the edge clearance
    - neighbors: the edges to the lattice neighbors stay out of the pad (soft:
      dropped by the caller if it would remove every position)

    Parameters
    ----------
    obstacles : Obstacles
        fixed obstacles
    node_id : Hashable
        node of the label
    shape : LabelShape
        label shape
    label_type : LabelType
        label type
    slider : Slider
        slider of the label
    edge_clearance : float
        minimal distance between ink and edges (normalized units)
    '''
    def __init__(self, obstacles: Obstacles, node_id: Hashable, shape: LabelShape, label_type: LabelType,
                 slider: Slider, edge_clearance: float):
        self.obstacles = obstacles
        self.node_id = node_id
        self.node = np.asarray(slider.node)
        self.shape = shape
        self.label_type = label_type
        self.slider = slider
        self.ink_clear = clearance_shape(shape, edge_clearance)
        hull = obstacles.hull
        self.outer = hull is not None and Point(slider.node).distance(hull.boundary) <= OUTER_NODE_DIST
        others = np.array([n != node_id for n in obstacles.node_ids], dtype=bool)
        self.other_xy = obstacles.node_xy[others]
        nb = obstacles.neighbors.get(node_id, np.empty((0, 2)))
        nb = nb[np.hypot(*(nb - self.node).T) > NEIGHBOR_EPS] if len(nb) else nb
        if len(nb):
            u = (nb - self.node) / np.hypot(*(nb - self.node).T)[:, None]
            starts = self.node + NEIGHBOR_EPS * u
            self.neighbor_edges = shapely.linestrings(np.stack([starts, nb], axis=1))
        else:
            self.neighbor_edges = np.empty(0, dtype=object)

    def base_ok(self, t: np.ndarray, positions: np.ndarray) -> np.ndarray:
        '''
        Parameters
        ----------
        t : np.ndarray
            slider arc lengths
        positions : np.ndarray
            (n, 2) drawing points at t

        Returns
        -------
        ok : np.ndarray
            whether each position passes the half plane, outer node, node and edge rules
        '''
        ok = self._half_plane(positions)
        idx = np.flatnonzero(ok)
        if len(idx):
            ok[idx] = ~self._edge_hits(positions[idx])
        idx = np.flatnonzero(ok)
        if len(idx):
            ok[idx] = ~self._node_hits(t[idx], positions[idx])
        if self.outer:
            idx = np.flatnonzero(ok)
            if len(idx):
                ok[idx] = ~shapely.intersects(translate_many(self.shape.ink, positions[idx]), self.obstacles.hull)
        return ok

    def neighbor_ok(self, t: np.ndarray, positions: np.ndarray) -> np.ndarray:
        '''
        Parameters
        ----------
        t : np.ndarray
            slider arc lengths
        positions : np.ndarray
            (n, 2) drawing points at t

        Returns
        -------
        ok : np.ndarray
            whether no edge to a lattice neighbor enters the pad
        '''
        if not len(self.neighbor_edges):
            return np.ones(len(positions), dtype=bool)
        pads = translate_many(self.shape.pad, positions)
        return ~shapely.intersects(pads[:, None], self.neighbor_edges[None, :]).any(axis=1)

    def exp_reach(self, t: np.ndarray) -> np.ndarray:
        '''
        Sweep vector of the exp shape at slider positions (see `LabelShape.exp`).

        Parameters
        ----------
        t : np.ndarray
            slider arc lengths

        Returns
        -------
        reach : np.ndarray
            (n, 2) reach vectors
        '''
        q = self.slider.local(t)
        lines = shapely.shortest_line(self.shape.ink, shapely.points(q))
        ink_pt = shapely.get_coordinates(lines).reshape(-1, 2, 2)[:, 0]
        away = ink_pt - q
        scale = np.abs(away).max(axis=1)
        scale[scale < 1e-12] = np.inf
        return self.shape.padding * away / scale[:, None]

    def _half_plane(self, positions: np.ndarray) -> np.ndarray:
        center = positions[:, 1] + self.slider.center_y
        if self.label_type == LabelType.INTENT:
            return center >= self.node[1] - 1e-9
        if self.label_type == LabelType.EXTENT:
            return center <= self.node[1] + 1e-9
        return np.ones(len(positions), dtype=bool)

    def _edge_hits(self, positions: np.ndarray) -> np.ndarray:
        hits = np.zeros(len(positions), dtype=bool)
        found = self.obstacles.edge_tree.query(translate_many(self.ink_clear, positions), predicate='intersects')
        hits[found[0]] = True
        return hits

    def _node_hits(self, t: np.ndarray, positions: np.ndarray) -> np.ndarray:
        '''
        A node X lies in exp = pad ⊕ [0, reach] iff the box spanned by X and
        X - reach meets the pad, so the exp shape is never built.
        '''
        hits = np.zeros(len(positions), dtype=bool)
        if not len(self.other_xy):
            return hits
        x0, y0, x1, y1 = self.shape.pad.bounds
        p = self.shape.padding
        lo = positions[:, None, :] + np.array([x0 - p, y0 - p])
        hi = positions[:, None, :] + np.array([x1 + p, y1 + p])
        near = ((self.other_xy[None] >= lo) & (self.other_xy[None] <= hi)).all(axis=2)
        i, j = np.nonzero(near)
        if not len(i):
            return hits
        reach = self.exp_reach(t[np.unique(i)])
        reach = reach[np.searchsorted(np.unique(i), i)]
        X = self.other_xy[j]
        Y = X - reach
        boxes = shapely.box(np.minimum(X[:, 0], Y[:, 0]), np.minimum(X[:, 1], Y[:, 1]),
                            np.maximum(X[:, 0], Y[:, 0]), np.maximum(X[:, 1], Y[:, 1]))
        pads = translate_many(self.shape.pad, positions[i])
        hit = shapely.intersects(pads, boxes)
        hits[i[hit]] = True
        return hits

@lru_cache(maxsize=None)
def clearance_shape(shape: LabelShape, clearance: float) -> BaseGeometry:
    '''
    Ink grown by the edge clearance (cached per shape).

    Parameters
    ----------
    shape : LabelShape
        label shape
    clearance : float
        clearance (normalized units)

    Returns
    -------
    grown : BaseGeometry
        simplified grown ink
    '''
    grown = shape.ink.buffer(clearance)
    return grown.simplify(SHAPE_TOLERANCE_MM / MM_PER_UNIT)

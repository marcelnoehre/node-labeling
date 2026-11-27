import numpy as np
import networkx as nx
import shapely

from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Tuple

from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from node_labeling.utils.constants import NODE_RADIUS
from node_labeling.v2.filter.shape_rules import Obstacles
from node_labeling.v2.models.binder import Binder
from node_labeling.v2.models.candidate import Candidate

ATTACH_SAMPLES: int = 32
OWN_EPS: float = 1e-6

@dataclass
class Layout:
    '''
    Fixed context of the overflow stages: the diagram and the labels placed
    directly at their nodes.

    Attributes
    ----------
    G : nx.Graph
        planarized graph containing normalized positions
    obstacles : Obstacles
        edges, nodes, hull
    direct : List[Candidate]
        labels placed at their nodes
    ink_tree : STRtree
        index over the inks of `direct`
    pad_tree : STRtree
        index over the pads of `direct`
    '''
    G: nx.Graph
    obstacles: Obstacles
    direct: List[Candidate]
    ink_tree: STRtree
    pad_tree: STRtree

    @classmethod
    def build(cls, G: nx.Graph, obstacles: Obstacles, direct: List[Candidate]) -> 'Layout':
        '''
        Parameters
        ----------
        G : nx.Graph
            planarized graph
        obstacles : Obstacles
            fixed obstacles
        direct : List[Candidate]
            labels placed at their nodes

        Returns
        -------
        layout : Layout
            context with spatial indexes
        '''
        inks = [c.ink for c in direct]
        pads = [c.pad for c in direct]
        shapely.prepare(inks + pads)
        return cls(G, obstacles, direct, STRtree(inks), STRtree(pads))

    def node_pos(self, node_id: Hashable) -> Tuple[float, float]:
        '''
        Parameters
        ----------
        node_id : Hashable
            node

        Returns
        -------
        pos : Tuple[float, float]
            node position
        '''
        x, y = self.G.nodes[node_id]['pos']
        return float(x), float(y)

def attach(candidate: Candidate) -> Binder:
    '''
    Shortest binder of an overflow label: from its node to the nearest point of
    its dot ring (ink grown by dot radius + gap). It never crosses its own label.

    Parameters
    ----------
    candidate : Candidate
        overflow label at its position

    Returns
    -------
    binder : Binder
        straight binder
    '''
    node = Point(candidate.node_pos)
    ring = candidate.dot_ring.exterior
    a = ring.interpolate(ring.project(node))
    return Binder(start=candidate.node_pos, end=(a.x, a.y))

def attach_many(candidates: List[Candidate]) -> List[Binder]:
    '''
    `attach` for many positions of one label at once.

    Parameters
    ----------
    candidates : List[Candidate]
        the same label at different positions

    Returns
    -------
    binders : List[Binder]
        shortest binder per position
    '''
    if not candidates:
        return []
    shape = candidates[0].shape
    pos = np.array([c.position for c in candidates])
    q = np.asarray(candidates[0].node_pos) - pos
    lines = shapely.shortest_line(shape.dot_ring.exterior, shapely.points(q))
    ends = shapely.get_coordinates(lines).reshape(-1, 2, 2)[:, 0] + pos
    return [Binder(start=c.node_pos, end=(float(x), float(y))) for c, (x, y) in zip(candidates, ends)]

def alternatives(candidate: Candidate, samples: int = ATTACH_SAMPLES) -> List[Binder]:
    '''
    Binders to other points of the dot ring that do not pass through the own
    label, shortest first (the first one is `attach`).

    Parameters
    ----------
    candidate : Candidate
        overflow label at its position
    samples : int
        number of ring points tried

    Returns
    -------
    binders : List[Binder]
        straight binders sorted by length
    '''
    ext = candidate.shape.dot_ring.exterior
    local = shapely.get_coordinates(shapely.line_interpolate_point(ext, np.linspace(0, 1, samples, endpoint=False), normalized=True))
    pts = local + np.asarray(candidate.position)
    node = np.asarray(candidate.node_pos)
    d = pts - node
    n = np.hypot(*d.T)
    keep = n > OWN_EPS
    pts, d, n = pts[keep], d[keep], n[keep]
    stubs = shapely.linestrings(np.stack([np.repeat(node[None], len(pts), 0), pts - d / n[:, None] * OWN_EPS], axis=1))
    ok = ~shapely.intersects(stubs, candidate.dot_ring)
    order = np.argsort(n, kind='stable')
    return [attach(candidate)] + [Binder(start=candidate.node_pos, end=(float(x), float(y))) for x, y in pts[order][ok[order]]]

def binder_valid(
        layout: Layout,
        lid: int,
        candidate: Candidate,
        binder: Binder,
        overflow: Dict[int, Candidate],
        soft: bool = False,
        edges: Optional[List[Tuple[Hashable, Hashable]]] = None,
        max_edge_crossings: int = -1
) -> bool:
    '''
    Check a binder (v1's `binding_line_valid` on shapes).

    - it does not touch the exp shape of another placed overflow label
    - strict: it does not cross another binder
    - soft: it does not touch the ink of a label placed at its node; strict: its pad
    - it keeps `NODE_RADIUS` to every other node
    - the attachment keeps `NODE_RADIUS` to the given edges
    - it crosses at most `max_edge_crossings` edges (if >= 0)

    Parameters
    ----------
    layout : Layout
        fixed context
    lid : int
        label id of the binder's label
    candidate : Candidate
        the label at its position
    binder : Binder
        binder to check
    overflow : Dict[int, Candidate]
        overflow labels; those with a binder count as placed
    soft : bool
        less restrictive mode
    edges : Optional[List[Tuple[Hashable, Hashable]]]
        edges the attachment keeps away from
    max_edge_crossings : int
        maximal number of edge crossings, -1 for no limit

    Returns
    -------
    valid : bool
        whether the binder is valid
    '''
    line = binder.line
    for o_lid, other in overflow.items():
        if o_lid == lid or other.binder is None:
            continue
        if line.intersects(other.exp):
            return False
        if not soft and line.crosses(other.binder.line):
            return False

    tree = layout.ink_tree if soft else layout.pad_tree
    if len(tree.query(line, predicate='intersects')):
        return False

    obst = layout.obstacles
    others = np.array([nid != candidate.node_id for nid in obst.node_ids], dtype=bool)
    if others.any() and (shapely.distance(line, shapely.points(obst.node_xy[others])) < NODE_RADIUS).any():
        return False

    end = Point(binder.end)
    for a, b in edges or []:
        if end.distance(LineString([layout.node_pos(a), layout.node_pos(b)])) < NODE_RADIUS:
            return False

    if max_edge_crossings >= 0:
        hits = obst.edge_tree.query(line, predicate='crosses')
        if len(hits) > max_edge_crossings:
            return False
    return True

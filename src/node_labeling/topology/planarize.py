import networkx as nx

from collections import defaultdict
from typing import Dict, List, Tuple
from shapely.strtree import STRtree
from shapely.geometry import LineString, Point

from node_labeling.utils.shapely_utils import as_points

def find_intersections(
        relations: List[Tuple[int, int]],
        positions: Dict[int, Tuple[float, float]],
) -> List[Tuple[int, int, Point]]:
    '''
    Compute all edge-edge intersections (non-adjacent edges only).

    Parameters
    ----------
    relations : List[Tuple[int, int]]
        list of (u, v) edge tuples
    positions : Dict[int, Tuple[float, float]] 
        dictionary mapping node id to (x, y) positions

    Returns
    -------
    intersections : List[Tuple[int, int, Point]]
        list of (edge_i, edge_j, shapely_point)
    '''
    edges = [LineString([positions[e[0]], positions[e[1]]]) for e in relations]
    tree = STRtree(edges)

    intersections = []
    for i, edge in enumerate(edges):
        for j in tree.query(edge, predicate='intersects'):
            # symmetric invariant
            if j <= i:
                continue
            # common endpoint
            if set(relations[i]) & set(relations[j]):
                continue
            # geometric intersection
            geom = edge.intersection(edges[j])
            for pt in as_points(geom):
                intersections.append((i, j, pt))

    return intersections


def build_planar_graph(
        relations: List[Tuple[int, int]],
        positions: Dict[int, Tuple[float, float]],
        intersections: List[Tuple[int, int, Point]],
) -> nx.Graph:
    '''
    Build a planar NetworkX graph by splitting each edge at its intersections.

    Original concept nodes are keyed by their integer id; synthetic
    intersection nodes are keyed by their (x, y) float tuple.

    Parameters
    ----------
    relations : List[Tuple[int, int]]
        list of (u, v) edge tuples (original graph)
    positions : Dict[int, Tuple[float, float]]
        dictionary mapping node id to (x, y) positions
    intersections : List[Tuple[int, int, Point]]
        list of intersections

    Returns
    -------
    planar_graph : nx.Graph 
        graph with a 'pos' attribute on every node
    '''
    G = nx.Graph()

    # original nodes
    for nid, pos in positions.items():
        G.add_node(nid, pos=pos)

    # lectical mapping
    pos_to_node = {pos: nid for nid, pos in positions.items()}

    # dummy vertices from intersections
    edge_crossings: Dict[int, list] = defaultdict(list)
    for i, j, pt in intersections:
        node = pos_to_node.get((pt.x, pt.y), (pt.x, pt.y))
        if node not in G:
            G.add_node(node, pos=(pt.x, pt.y))
        for eid in (i, j):
            e = relations[eid]
            x0, y0 = positions[e[0]]
            x1, y1 = positions[e[1]]
            dx, dy = x1 - x0, y1 - y0
            denom = dx * dx + dy * dy
            t = ((pt.x - x0) * dx + (pt.y - y0) * dy) / denom if denom else 0
            if t <= 0 or t >= 1:
                continue
            edge_crossings[eid].append((t, node))

    # subdivide edges
    for eid, e in enumerate(relations):
        splits = sorted(edge_crossings[eid])
        chain = [e[0]] + [node for _, node in splits] + [e[1]]
        dedup_chain = [chain[0]]
        for node in chain[1:]:
            if node != dedup_chain[-1]:
                dedup_chain.append(node)
        for a, b in zip(dedup_chain, dedup_chain[1:]):
            G.add_edge(a, b)

    return G

import networkx as nx

from typing import List, Tuple
from shapely.geometry import Point

from node_labeling.utils.constants import TARGET_HEIGHT

def normalize_positions(G: nx.Graph) -> float:
    '''
    Normalize all node 'pos' attributes in-place so that the graph's
    bounding box has a height of *cfg.target_height*, with the aspect ratio
    preserved and the origin placed at the bounding-box minimum.

    Parameters
    ----------
    G : nx.Graph
        graph whose nodes have a 'pos' attribute

    Returns
    -------
    mm_per_unit : float
        scale factor
    '''
    xs = [G.nodes[n]['pos'][0] for n in G.nodes]
    ys = [G.nodes[n]['pos'][1] for n in G.nodes]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    height = max_y - min_y
    if height == 0:
        raise ValueError('All nodes have the same y-coordinate!')

    scale = TARGET_HEIGHT / height

    for n in G.nodes:
        x, y = G.nodes[n]['pos']
        G.nodes[n]['pos'] = ((x - min_x) * scale, (y - min_y) * scale)

    G.graph['normalized_height'] = TARGET_HEIGHT
    G.graph['normalized_width']  = (max_x - min_x) * scale
    G.graph['norm_min_x']        = min_x
    G.graph['norm_min_y']        = min_y
    G.graph['norm_scale']        = scale

    return scale

def normalize_intersections(G: nx.Graph, intersections: List[Tuple[int, int, Point]]) -> List[Tuple[float, float]]:
    '''
    Normalize intersection coordinates to match the normalized graph.

    Parameters
    ----------
    G : nx.Graph
        graph containing normalization metadata
    intersections : List[Tuple[int, int, Point]]
        list of intersections

    Returns
    -------
    normalized_intersections : List[Tuple[float, float]]
        scaled intersection points
    '''
    _s   = G.graph['norm_scale']
    _mx  = G.graph['norm_min_x']
    _my  = G.graph['norm_min_y']
    return [
        ((pt.coords[0][0] - _mx) * _s, (pt.coords[0][1] - _my) * _s)
        for _, _, pt in intersections
    ]
import alphashape
import numpy as np
import networkx as nx

from typing import List, Tuple, Set
from shapely.geometry import Polygon
from scipy.spatial import KDTree

def _shoelace(G: nx.Graph, nodes: List[int], scale: float) -> float:
    '''
    Area of a polygon (shoelace)

    Parameters
    ----------
    G: nx.Graph
        graph containing the positions
    nodes: List[int]
        nodes spanning the polygon
    scale: float
        scale factor (mm)

    Returns
    -------
    area : float
        area of the polygon in mm^2
    '''
    pts = [G.nodes[nid]['pos'] for nid in nodes]
    n = len(pts)
    norm_area = abs(sum(
        pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
        for i in range(n)
    )) / 2
    return norm_area / (scale**2)

def _centroid(G: nx.Graph, nodes: List[int]) -> Tuple[float, float]:
    '''
    Centroid of a polygon

    Parameters
    ----------
    G: nx.Graph
        graph containing the positions
    nodes: List[int]
        nodes spanning the polygon

    Returns
    -------
    centroid: Tuple[float, float]
        centroid of the polygon
    '''
    pts = [G.nodes[nid]['pos'] for nid in nodes]
    c = Polygon(pts).centroid
    return c.x, c.y

def extract_faces(G: nx.Graph, scale: float) -> Tuple[List[int], List[List[int]], List[float], List[Tuple[float, float]]]:
    '''
    Extract all faces of a planar graph.

    Parameters
    ----------
    G : nx.Graph
        planar graph
    scale : float
        mm per unit

    Returns
    -------
    alpha_shape : List[int]
        alpha shape containing the graph
    bounded_faces : Tuple[List[List[int]]
        list of bounded faces
    areas : List[float]
        area of each bounded face
    centroids : List[Tuple[float, float]]
        centroid of each bounded face

    Raises
    ------
    ValueError : if G is not planar
    '''
    is_planar, embedding = nx.check_planarity(G)
    if not is_planar:
        raise ValueError('Graph is not planar — cannot extract faces!')
    
    planar_nodes = list(G.nodes())
    points = np.array([G.nodes[nid]['pos'] for nid in planar_nodes])
    x_range = np.ptp(points[:, 0])

    if x_range == 0:
        return np.argsort(points[:, 1]).tolist(), [], [], []

    seen  = set()
    faces = []
    for u, v in G.edges():
        for a, b in [(u, v), (v, u)]:
            face = embedding.traverse_face(a, b)
            key  = frozenset(face)
            if key not in seen:
                seen.add(key)
                faces.append(face)
    
    hull_poly = alphashape.alphashape(points, 0.1)
    boundary_coords = np.array(hull_poly.exterior.coords)[:-1]
    tree = KDTree(points)
    _, indices = tree.query(boundary_coords)
    alpha_shape = [planar_nodes[i] for i in indices]

    face_areas = [(f, _shoelace(G, f, scale)) for f in faces]
    outer_face = max(face_areas, key=lambda x: x[1])[0]
    bounded = [(f, a) for f, a in face_areas if f is not outer_face]
    bounded_faces = [f for f, _ in bounded]
    areas = [a for _, a in bounded]
    centroids = [_centroid(G, f) for f in bounded_faces]

    return alpha_shape, bounded_faces, areas, centroids

def node_faces(
        G: nx.Graph,
        node: int,
        bounded_faces: List[List[int]]
) -> List[Tuple[int, int]]:
    '''
    Collect all unique edges from faces that contain the node

    Parameters
    ----------
    G : nx.Graph
        graph containing the positions
    node : int
        node for which to find assigned faces
    bounded_faces: List[List[int]]
        boundary walk of each bounded face
    
    Returns
    -------
    face_edges : List
        a list of (u, v) tuples.
    '''
    edges = list(G.edges(node))
    seen: Set[frozenset] = set(frozenset(e) for e in edges)
    for face in bounded_faces:
        if node not in face:
            continue

        face_edges = {frozenset((face[i], face[(i + 1) % len(face)])) for i in range(len(face))}
        for e in face_edges:
            if e not in seen:
                seen.add(e)
                u, v = tuple(e)
                edges.append((u, v))

    return edges

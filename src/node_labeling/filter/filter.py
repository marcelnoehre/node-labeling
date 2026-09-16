import copy
import numpy as np
import networkx as nx

from shapely import Polygon, Point, LineString, box
from typing import Dict, List

from node_labeling.utils.graph_lattice import GraphLattice
from node_labeling.models.anchor import AnchorType
from node_labeling.models.label_candidate import LabelCandidate
from node_labeling.topology.faces import node_faces
from node_labeling.filter.rules import *
from node_labeling.filter.matching import *

def restrict_outer_node_candidates(
    G: nx.Graph,
    label_candidates: Dict[int, List[LabelCandidate]],
    alpha_shape: List[int],
) -> Dict[int, List[LabelCandidate]]:
    '''
    Restrict the label candidates for nodes on the boundary of the alpha shape
    to candidates whose ink_bbox does not intersect with the alpha shape.

    Parameters
    ----------
    G : nx.Graph
        graph containing the positions
    label_candidates : Dict[int, List[LabelCandidate]]
        active label candidates
    alpha_shape : List[int],
        boundary walk of the alpha shape

    Returns
    -------
    filtered_candidates : Dict[int, List[LabelCandidate]]
        remaining label candidates
    '''
    if not alpha_shape:
        return label_candidates
    
    hull_coords = np.array([G.nodes[node]['pos'] for node in alpha_shape])
    if np.ptp(hull_coords[:, 0]) == 0:
        return label_candidates
    
    hull_poly = Polygon(hull_coords)
    if not hull_poly.is_valid:
        hull_poly = hull_poly.buffer(0)

    filtered_candidates = copy.deepcopy(label_candidates)

    for lid, candidate_list in filtered_candidates.items():
        if not candidate_list:
            continue
        node_pt = Point(G.nodes[label_candidates[lid][0].node_id]['pos'])
        if node_pt.distance(hull_poly.boundary) > 0.25:
            continue

        valid = []
        for candidate in candidate_list:
            candidate_poly = Polygon(candidate.ink_bbox_corners)

            if not hull_poly.contains(candidate_poly) and not hull_poly.intersects(candidate_poly):
                valid.append(candidate)

        filtered_candidates[lid] = valid

    return filtered_candidates

def filter_candidates_by_nodes(
        G: nx.Graph,
        label_candidates: Dict[int, List[LabelCandidate]],
        nodes: List[int],
) -> Dict[int, List[LabelCandidate]]:
    '''
    Remove label candidates whose expanded bounding box intersects with a non-incident node.

    Parameters
    ----------
    G : nx.Graph
        graph containing the positions
    label_candidates : Dict[int, List[LabelCandidate]]
        active label candidates
    nodes : List[int]
        list of nodes

    Returns
    -------
    filtered_candidates : Dict[int, List[LabelCandidate]]
        remaining label candidates
    '''
    filtered_candidates: Dict[int, List[LabelCandidate]] = {}

    for lid, candidates in label_candidates.items():
        surviving = []
        for candidate in candidates:
            (bl_x, bl_y), _, (tr_x, tr_y), _ = candidate.exp_bbox_corners
            occupied = False
            for nid in nodes:
                if nid == candidate.node_id:
                    continue

                ox, oy = G.nodes[nid]['pos']
                if (bl_x <= ox <= tr_x) and (bl_y <= oy <= tr_y):
                    occupied = True
                    break
            
            if not occupied:
                surviving.append(candidate)
                
        filtered_candidates[lid] = surviving

    return filtered_candidates

def filter_candidates_by_edges(
        G: nx.Graph,
        label_candidates: Dict[int, List[LabelCandidate]],
        bounded_faces: List[List[int]]
) -> Dict[int, List[LabelCandidate]]:
    '''
    Remove label candidates whose padding bounding box intersects with graph edges.

    Parameters
    ----------
    G : nx.Graph
        graph containing the positions
    label_candidates : Dict[int, List[LabelCandidate]]
        active label candidates
    bounded_faces : List[List[int]]
        boundary walk of each bounded faces 

    Returns
    -------
    filtered_candidates : Dict[int, List[LabelCandidate]]
        remaining label candidates
    '''
    filtered_candidates: Dict[int, List[LabelCandidate]] = {}

    for lid, candidates in label_candidates.items():
        if not candidates:
            continue
        nid = candidates[0].node_id
        edges = [
            LineString([G.nodes[u]['pos'], G.nodes[v]['pos']])
            for u, v in node_faces(G, nid, bounded_faces)
        ]

        surviving = []
        for candidate in candidates:
            ibl, _, itr, _ = candidate.ink_bbox_corners
            ink_shape = box(ibl[0], ibl[1], itr[0], itr[1])
            pad_shape = ink_shape.buffer(0.01)
            if not any(pad_shape.intersects(e) for e in edges):
                surviving.append(candidate)

        filtered_candidates[lid] = surviving

    return filtered_candidates

def filter_candidates_by_neighbor_direction(
        G: nx.Graph,
        label_candidates: Dict[int, List[LabelCandidate]],
        lattice: GraphLattice
) -> Dict[int, List[LabelCandidate]]:
    '''
    Filter candidates based on their direct neighbors.

    Parameters
    ----------
    G : nx.Graph
        graph containing the positions
    label_candidates : Dict[int, List[LabelCandidate]]
        active label candidates
    lattice : GraphLattice
        lattice structure containing neighbors

    Returns
    -------
    filtered_candidates : Dict[int, List[LabelCandidate]]
        remaining label candidates
    '''
    filtered_candidates = copy.deepcopy(label_candidates)

    for lid, candidates in label_candidates.items():
        if not candidates or len(candidates) == 1:
            filtered_candidates[lid] = candidates
            continue

        nid = candidates[0].node_id
        node_x = G.nodes[nid]['pos'][0]
        node_y = G.nodes[nid]['pos'][1]
        neighbors = list(lattice.children(nid)) + list(lattice.parents(nid))

        has_top_left = any((G.nodes[nb]['pos'][0] < node_x and G.nodes[nb]['pos'][1] > node_y) for nb in neighbors)
        has_bottom_left = any((G.nodes[nb]['pos'][0] < node_x and G.nodes[nb]['pos'][1] < node_y) for nb in neighbors)
        has_top_right = any((G.nodes[nb]['pos'][0] > node_x and G.nodes[nb]['pos'][1] > node_y) for nb in neighbors)
        has_bottom_right = any((G.nodes[nb]['pos'][0] > node_x and G.nodes[nb]['pos'][1] < node_y) for nb in neighbors)

        filter = []
        if has_top_left:
            filter.extend([AnchorType.B, AnchorType.R, AnchorType.BR])
        if has_top_right:
            filter.extend([AnchorType.B, AnchorType.L, AnchorType.BL])
        if has_bottom_left:
            filter.extend([AnchorType.T, AnchorType.R, AnchorType.TR])
        if has_bottom_right:
            filter.extend([AnchorType.T, AnchorType.L, AnchorType.TL])

        by_type: Dict[str, List[LabelCandidate]] = {}
        for c in candidates:
            by_type.setdefault(c.label_type, []).append(c)

        for group in by_type.values():
            filtered = [c.anchor.anchor_type for c in group if c.anchor.anchor_type in filter]

            # do not filter all
            if len(filtered) == len(group):
                continue

            filtered_candidates[lid] = [
                c for c in filtered_candidates[lid] 
                if c.anchor.anchor_type not in filtered
            ]
    
    return filtered_candidates

def filter_hyrid(label_candidates: Dict[int, List[LabelCandidate]]) -> Dict[int, List[LabelCandidate]]:
    '''
    Modified Hybrid algorithm with a readability tie breaker Preserves alternatives.
    Candidates of the same feature are allowed to overlap each other, 
    but will still block candidates from DIFFERENT features.

    Parameters
    ----------
    label_candidates : Dict[int, List[LabelCandidate]]
        active label candidates

    Returns
    -------
    filtered_candidates : Dict[int, List[LabelCandidate]]
        remaining label candidates
    '''
    conflicts = build_conflict_map(label_candidates)
    while True:
        label_candidates, changed_L1 = apply_L1(label_candidates, conflicts)
        label_candidates, changed_L2 = apply_L2(label_candidates, conflicts)
        label_candidates, changed_L3 = apply_L3(label_candidates, conflicts)
        if not changed_L1 and not changed_L2 and not changed_L3:
            break

    unresolved = {
        lid: cands
        for lid, cands in label_candidates.items()
        if len(cands) > 1
    }
    chosen: Dict[int, LabelCandidate] = {
        lid: cands[0]
        for lid, cands in label_candidates.items()
        if len(cands) == 1
    }

    if unresolved:
        components = connected_components(unresolved, conflicts)
        for comp in components:
            cliques = kt_reduce(comp, conflicts)
            if not cliques:
                continue
            match = maximum_bipartite_matching(list(comp.keys()), cliques)
            chosen.update(match)

    placed_entries: List[Tuple[int, LabelCandidate]] = []
    filtered_candidates: Dict[int, List[LabelCandidate]] = {lid: [] for lid in label_candidates}
    for lid in sorted(filtered_candidates.keys(), key=lambda lid: chosen[lid].anchor.anchor_type.rank if lid in chosen else float('inf')):
        for cand in sorted(label_candidates[lid], key=lambda c: c.anchor.anchor_type.rank):
            if not any(p_lid != lid and pad_overlap(cand, p_cand) for p_cand, p_lid in placed_entries):
                filtered_candidates[lid].append(cand)
                placed_entries.append((cand, lid))
                break

    return filtered_candidates
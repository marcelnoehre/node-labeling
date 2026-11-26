import networkx as nx

from typing import Dict, List

from node_labeling.utils.constants import *
from node_labeling.models.anchor import AnchorType, Anchor
from node_labeling.v1.models.label_candidate import LabelCandidate
from node_labeling.v1.utils.geometry import label_wh, label_rows, label_padding, overflow_boxes

def _compute_overflow_candidate(G: nx.Graph, initial_candidate: LabelCandidate) -> LabelCandidate:
    '''
    Compute overflow candidate based on initial data.

    Parameters
    ----------
    G : nx.Graph
        graph containing positions
    initial_candidate : LabelCandidate
        initial label data

    Returns
    -------
    overflow_candidate : LabelCandidate
        derived overflow candidate
    '''

    if 'normalized_height' not in G.graph:
        raise ValueError("Call normalize_positions(G) before compute.")

    ink_w, ink_h = label_wh(initial_candidate.ink_bbox_corners)
    padding = label_padding(ink_h, label_rows(initial_candidate.text))

    nid = initial_candidate.node_id
    cx, cy = G.nodes[nid]['pos']

    (ibl, ibr, itr, itl), (pbl, pbr, ptr, ptl), (ebl, ebr, etr, etl) = overflow_boxes(
        cx, cy, ink_w / 2.0, ink_h / 2.0, padding
    )

    return LabelCandidate(
        node_id=nid,
        anchor=Anchor(AnchorType.O, cx, cy, pbl, pbr, ptr, ptl),
        label_type=initial_candidate.label_type,
        ink_bbox_corners=(ibl, ibr, itr, itl),
        pad_bbox_corners=(pbl, pbr, ptr, ptl),
        exp_bbox_corners=(ebl, ebr, etr, etl),
        center=(cx, cy),
        text=initial_candidate.text
    )

def generate_overflow_candidates(
        G: nx.Graph, 
        label_candidates: Dict[int, List[LabelCandidate]], 
        initial_candidates: Dict[int, List[LabelCandidate]]
) -> Dict[int, LabelCandidate]:
    '''
    Generate overflow candidates for each label without remaining a placed candidate.

    Parameters
    ----------
    G : nx.Graph
        graph containing positions
    label_candidates: Dict[int, List[LabelCandidate]], 
        remaining label candidates
    initial_candidates: Dict[int, List[LabelCandidate]], 
        initial label data

    Returns
    -------
    overflow_candidates : Dict[int, LabelCandidate]
        derived overflow candidates
    '''
    return {
        lid: _compute_overflow_candidate(G, initial_candidates[lid][0])
        for lid, candidates in label_candidates.items()
        if not candidates
    }
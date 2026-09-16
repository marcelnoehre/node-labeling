import numpy as np
import networkx as nx

from typing import Dict, List

from node_labeling.utils.constants import *
from node_labeling.models.anchor import AnchorType, Anchor
from node_labeling.models.label_candidate import LabelCandidate
from node_labeling.utils.geometry import bbox_corners

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

    init_bl, init_br, _, init_tl = initial_candidate.ink_bbox_corners
    ink_w, ink_h = abs(init_br[0] - init_bl[0]), abs(init_tl[1] - init_bl[1])

    corner_multiplier = 1 / np.sqrt(2)
    s = (1.0 / corner_multiplier)
    rows = len(initial_candidate.text.split(r'\\[-1pt]'))
    padding = (ink_h / rows - 0.1 * rows) * 0.5

    half_iw = ink_w / 2.0
    half_ih = ink_h / 2.0
    half_pw  = (ink_w + 2 * padding / 3) / 2.0
    half_ph  = (ink_h + 2 * padding / 3) / 2.0
    half_ew = (ink_w + 2 * padding * s) / 2.0
    half_eh = (ink_h + 2 * padding * s) / 2.0
    
    nid = initial_candidate.node_id
    cx, cy = G.nodes[nid]['pos']

    ibl, ibr, itr, itl = bbox_corners(cx, cy, half_iw, half_ih)
    pbl, pbr, ptr, ptl = bbox_corners(cx, cy, half_pw, half_ph)
    ebl, ebr, etr, etl = bbox_corners(cx, cy, half_ew, half_eh)

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
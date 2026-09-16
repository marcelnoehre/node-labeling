import numpy as np
import networkx as nx

from typing import List, Dict

from node_labeling.utils.config import Config
from node_labeling.utils.constants import *
from node_labeling.utils.graph_lattice import GraphLattice
from node_labeling.models.anchor import AnchorType, Anchor
from node_labeling.models.label_type import LabelType
from node_labeling.models.label_candidate import LabelCandidate
from node_labeling.label.formatter import *

def _compute_label_candidates(
        G: nx.Graph,
        node_id: int,
        label_text: str,
        label_type: LabelType
) -> List[LabelCandidate]:
    '''
    Compute all candidates for a label regarding the 8-position model.

    Parameters
    ----------
    G : nx.Graph
        graph containing normalized positions
    node_id : int
        node_id the label is assigned to
    label_text : str
        formatted label string
    label_type : LabelType
        general, extent, or intent

    Returns
    -------
    label_candidates : List[LabelCandidate]
        all label candidates for a given label
    '''
    if 'normalized_height' not in G.graph:
        raise ValueError('Normalize positions before computing label candidates')
    
    mm_per_unit = PHYSICAL_HEIGHT_MM / G.graph['normalized_height']
    units_per_mm = 1.0 / mm_per_unit

    ink_w_mm, ink_h_mm = measure_ink_mm(label_text)

    rows = len(label_text.split(r'\\[-1pt]'))
    padding = (ink_h_mm / rows - 0.1 * rows) * 0.5

    # Inner half-extents (ink only, no padding)
    half_iw = (ink_w_mm  * units_per_mm) / 2.0
    half_ih = (ink_h_mm  * units_per_mm) / 2.0

    corner_multiplier = 1 / np.sqrt(2)

    node_x, node_y = G.nodes[node_id]['pos']
    lid_candidates: List[LabelCandidate] = []
    
    for anchor in label_type.anchors:
        is_top    = anchor in [AnchorType.T, AnchorType.TL, AnchorType.TR]
        is_bottom = anchor in [AnchorType.B, AnchorType.BL, AnchorType.BR]
        is_left   = anchor in [AnchorType.L, AnchorType.TL, AnchorType.BL]
        is_right  = anchor in [AnchorType.R, AnchorType.TR, AnchorType.BR]
        is_corner = anchor not in [AnchorType.T, AnchorType.B, AnchorType.L, AnchorType.R]
        
        s = corner_multiplier if is_corner else (1.0 / corner_multiplier)

        p_top    = (padding * s) if is_top else padding
        p_bottom = (padding * s) if is_bottom else padding
        p_left   = (padding * s) if is_left else padding
        p_right  = (padding * s) if is_right else padding

        total_w_units = (ink_w_mm + p_left + p_right) * units_per_mm
        total_h_units = (ink_h_mm + p_top + p_bottom) * units_per_mm

        half_w = total_w_units / 2.0
        half_h = total_h_units / 2.0

        offset_x = 0
        if is_left:
            offset_x = half_w
        if is_right: 
            offset_x = -half_w

        offset_y = 0
        if is_top:
            offset_y = -half_h
        if is_bottom:
            offset_y = half_h

        cx = node_x + offset_x
        cy = node_y + offset_y

        pbl = (cx - half_w, cy - half_h)
        ptr = (cx + half_w, cy + half_h)
        pbr, ptl = (ptr[0], pbl[1]), (pbl[0], ptr[1])

        ink_cx = cx + ((p_left - p_right) * units_per_mm / 2.0)
        ink_cy = cy + ((p_bottom - p_top) * units_per_mm / 2.0)
        
        ibl = (ink_cx - half_iw, ink_cy - half_ih)
        itr = (ink_cx + half_iw, ink_cy + half_ih)
        ibr, itl = (itr[0], ibl[1]), (ibl[0], itr[1])

        exp_l = (padding * units_per_mm) if is_right else 0
        exp_r = (padding * units_per_mm) if is_left else 0
        exp_t = (padding * units_per_mm) if is_bottom else 0
        exp_b = (padding * units_per_mm) if is_top else 0

        ebl = (pbl[0] - exp_l, pbl[1] - exp_b)
        etr = (ptr[0] + exp_r, ptr[1] + exp_t)
        ebr, etl = (etr[0], ebl[1]), ((ebl[0], etr[1]))

        lid_candidates.append(LabelCandidate(
            node_id=node_id,
            anchor=Anchor(anchor, cx, cy, pbl, pbr, ptr, ptl),
            label_type=label_type,
            ink_bbox_corners=(ibl, ibr, itr, itl),
            pad_bbox_corners=(pbl, pbr, ptr, ptl),
            exp_bbox_corners=(ebl, ebr, etr, etl),
            center=(cx, cy),
            text=label_text
        ))

    return lid_candidates

def generate_label_candidates(G: nx.Graph, lattice: GraphLattice, cfg: Config) -> Dict[int, List[LabelCandidate]]:
    '''
    Generate all label candidates.

    Parameters
    ----------
    G : nx.Graph
        graph containing the positions
    lattice : GraphLattice
        lattice structure and labels loaded from graphml
    cfg : Config
        configuration

    Returns
    -------
    label_candidates : Dict[int, List[LabelCandidate]]
        initial set of label candidates
    '''
    label_candidates = {}
    lid = 0

    if cfg.label_config[LabelType.GENERAL]:
        for nid in lattice.nodes:
            general_txt = format_label_text(cfg, lattice.general(nid), LabelType.GENERAL)
            if general_txt:
                label_candidates[lid] = _compute_label_candidates(G, nid, general_txt, LabelType.GENERAL)
                lid += 1

    if cfg.label_config[LabelType.EXTENT]:
        for nid in lattice.nodes:
            objects = sorted(str(g) for g in lattice.extent(nid))
            extent_txt = format_label_text(cfg, ', '.join(objects), LabelType.EXTENT)
            if extent_txt:
                label_candidates[lid] = _compute_label_candidates(G, nid, extent_txt, LabelType.EXTENT)
                lid += 1

    if cfg.label_config[LabelType.INTENT]:
        for nid in lattice.nodes:
            attributes = sorted(str(m) for m in lattice.intent(nid))
            intent_txt = format_label_text(cfg, ', '.join(attributes), LabelType.INTENT)
            if intent_txt:
                label_candidates[lid] = _compute_label_candidates(G, nid, intent_txt, LabelType.INTENT)
                lid += 1
    
    return label_candidates
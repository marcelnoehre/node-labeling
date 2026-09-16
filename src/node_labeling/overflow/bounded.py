import copy
import numpy as np
import networkx as nx

from shapely import box, unary_union
from shapely.geometry import Point, Polygon
from shapely.validation import make_valid
from typing import Dict, List, Optional, Tuple

from node_labeling.models.label_type import LabelType
from node_labeling.models.label_candidate import LabelCandidate
from node_labeling.models.anchor import Anchor, AnchorType
from node_labeling.utils.geometry import *
from node_labeling.topology.faces import *

def _eroded_space(space: Polygon, w: float, h: float, below: float, above: float) -> Polygon:
    '''
    Return the set of valid center positions for a label inside space.

    Parameters
    ----------
    space : Polygon
        search space
    w : float
        width of the label
    h : float
        height of the label
    below : float
        max y-ccordinate for extent labels 
    above : float
        min y-coordinate for intent labels
    
    Returns
    -------
    eroded : Polygon
        eroded space
    '''
    smaller, larger = min(w, h), max(w, h)
    eroded = space.buffer(-smaller / 2, join_style=2)
    if eroded.is_empty:
        return eroded
    
    eroded = eroded.buffer(-(larger - smaller) / 2, join_style=2)

    # clip search space for fca version
    if (below is not None) or (above is not None):
        min_x, min_y, max_x, max_y = space.bounds
        limit_top = min(max_y, below) if below is not None else max_y
        limit_bottom = max(min_y, above) if above is not None else min_y
        clip_box = box(min_x, limit_bottom, max_x, limit_top)
        eroded = eroded.intersection(clip_box)
    
    return eroded

def _fits(space: Polygon, w: float, h: float, type: LabelType, node_y: float) -> bool:
    '''
    Check wether label fits into polygon.

    Parameters
    ----------
    space : Polygon
        search space
    w : float
        width of the label
    h : float
        height of the label
    type : LabelType
        general, extent, or intent 
    node_y : float
        y-coordinate of the node
    
    Returns
    -------
    fits : bool
        wether a label fits into the eroded search space 
    '''
    if space.is_empty:
        return False
    
    s_minx, s_miny, s_maxx, s_maxy = space.bounds
    if w > (s_maxx - s_minx) or h > (s_maxy - s_miny) or w * h > space.area:
        return False
    
    below = None
    above = None
    if type == LabelType.EXTENT:
        below = node_y
    if type == LabelType.INTENT:
        above = node_y

    return not _eroded_space(space, w, h, below, above).is_empty

def _fitting_faces(
    overflow_candidates: Dict[int, LabelCandidate],
    processed_faces: List[Tuple[int, Polygon]],
) -> Dict[int, List[int]]:
    '''
    For each overflow label, return the face_ids where it fits.

    Parameters
    ----------
    overflow_candidates : Dict[int, LabelCandidate]
        overflow candidates to be placed
    processed_faces : List[Tuple[int, Polygon]]
        pruned faces

    Returns
    -------
    fitting_faces : Dict[int, List[int]]
        fitting faces for each overflow label
    '''
    fitting_faces: Dict[int, List[int]] = {}
    for lid, ol in overflow_candidates.items():
        ew, eh = label_wh(ol.exp_bbox_corners)
        fitting_faces[lid] = [
            fid for fid, space in processed_faces
            if _fits(space, ew, eh, ol.label_type, ol.center[1])
        ]

    return fitting_faces

def _adjust_anchors(
        G: nx.Graph, 
        label_candidates: Dict[int, List[LabelCandidate]], 
        overflow_candidates: Dict[int, LabelCandidate]
    ) -> Dict[int, LabelCandidate]:
    '''
    Translate label to improve anchoring.

    Parameters
    ----------
    G : nx.Graph
        graph containing positions
    label_candidates: Dict[int, List[LabelCandidate]], 
        already placed candidates
    overflow_candidates: Dict[int, LabelCandidate], 
        placed overflow candidates

    Returns
    -------
    overflow_candidates: Dict[int, LabelCandidate]
        refined overflow candidates
    '''
    bounded_overflow_labels = sorted(
        [lid for lid, ol in overflow_candidates.items() if ol.anchor.anchor_type != AnchorType.O], 
        key=lambda lid: Point(overflow_candidates[lid].anchor.pos).distance(Point(G.nodes[overflow_candidates[lid].node_id]['pos']))
    )

    for lid in bounded_overflow_labels:
        ol = overflow_candidates[lid]
        cx, cy = ol.center
        anchor_candidates = [
            Anchor(at, *ol.center, *ol.pad_bbox_corners)
            for at in AnchorType if at != AnchorType.O
        ]

        label_center = np.array([cx, cy])
        node_pos = G.nodes[ol.node_id]['pos']
        node_pt = np.array(node_pos)
        vec = node_pt - label_center
        dist_to_node = np.linalg.norm(vec)
        unit = vec / dist_to_node if dist_to_node > 0 else vec
        scored_anchors: List[Tuple[float, Anchor]] = []
        for anchor in anchor_candidates:
            va = np.array(anchor.pos) - label_center
            na = np.linalg.norm(va)
            align = float(np.dot(unit, va / na)) if na > 0 else -1.0
            scored_anchors.append((align, anchor))

        scored_anchors.sort(key=lambda x: x[0], reverse=True)

        for _, anchor in scored_anchors:
            if ol.anchor.anchor_type == anchor.anchor_type:
                break

            anchor_offset_from_center = np.array(anchor.pos) - label_center
            fixed_anchor_pos = np.array(ol.anchor.pos)
            new_center = fixed_anchor_pos - anchor_offset_from_center
            pw, ph = label_wh(ol.pad_bbox_corners)
            new_corners = bbox_corners(*new_center, pw/2, ph/2)
            new_anchor = Anchor(anchor.anchor_type, *ol.anchor.pos, *new_corners)

            tmp_ol = copy.deepcopy(ol)
            tmp_ol.update_position(*new_center, new_anchor)

            tmp_pad_poly = Polygon(tmp_ol.pad_bbox_corners)
            tmp_exp_poly = Polygon(tmp_ol.exp_bbox_corners)

            if any(tmp_exp_poly.intersects(Polygon(l[0].pad_bbox_corners)) for l in label_candidates.values() if l):
                continue # overlaps with existing label
    
            if any(tmp_exp_poly.intersects(Polygon(oc.exp_bbox_corners)) for oc in overflow_candidates.values() if (oc.node_id != tmp_ol.node_id) and (oc.anchor.anchor_type != AnchorType.O)):
                continue # overlaps another overflow label
 
            if any(
                tmp_pad_poly.intersects(LineString([G.nodes[oc.node_id]['pos'], oc.anchor.pos])) 
                for oc in overflow_candidates.values() if (oc.node_id != tmp_ol.node_id) and (oc.anchor.anchor_type != AnchorType.O)
            ):
                continue # overlaps with a binder of another overflow label

            overflow_candidates[lid] = tmp_ol
            break

    return overflow_candidates

def _find_valid_position(
    G: nx.Graph,
    label_candidates: Dict[int, List[LabelCandidate]],
    overflow_candidates: Dict[int, LabelCandidate],
    placed_overflow: List[int],
    space: Polygon,
    bounded_faces: List[List[int]],
    lid: int
) -> Optional[Tuple[float, float, Anchor]]:
    '''
    Valid center position inside the eroded space that is closest to the respective node and has a valid binding line.

    Parameters
    ----------
    G : nx.Graph
        graph containing positions
    label_candidates : Dict[int, List[LabelCandidate]]
        already placed label candidates
    overflow_candidates : Dict[int, LabelCandidate]
        overflow candidates to be placed
    placed_overflow : List[int]
        already placed overflow candidates
    space : Polygon
        eroded space
    bounded_faces: List[List[int]]
        bounded faces of the planar graph
    lid : int
        label to be placed

    Returns
    -------
        cx : float
            x-coordinate of new center
        cy : float
            y-coordinate of new center
        anchor : Anchor
            new anchor position
    '''
    ol = overflow_candidates[lid]
    ew, eh = label_wh(ol.exp_bbox_corners)
    node_pos = G.nodes[ol.node_id]['pos']
    node_pt = np.array(node_pos)
    node_point = Point(node_pos)

    # eroded space
    below = None
    above = None
    if ol.label_type == LabelType.EXTENT:
        below = ol.center[1]
    if ol.label_type == LabelType.INTENT:
        above = ol.center[1]
    eroded = _eroded_space(space, ew, eh, below, above)
    if eroded.is_empty:
        return None

    # one candidate per anchor
    anchor_candidates = [
        Anchor(at, node_pt[0], node_pt[1], *ol.pad_bbox_corners).pos
        for at in AnchorType if at != AnchorType.O
    ]

    # place as close as possible to original node
    if eroded.contains(node_point):
        anchor_candidates.append((float(node_pt[0]), float(node_pt[1])))
    else:
        nearest_pt = eroded.boundary.interpolate(eroded.boundary.project(node_point))
        anchor_candidates.append((nearest_pt.x, nearest_pt.y))

    # fine grid fallback
    minx, miny, maxx, maxy = eroded.bounds
    step = min(ew, eh) / 16
    x = minx
    while x <= maxx:
        y = miny
        while y <= maxy:
            if eroded.contains(Point(x, y)):
                anchor_candidates.append((x, y))
            y += step
        x += step

    def min_anchor_dist(center: Tuple[float, float]) -> float:
        '''
        Get the anchor with minimum distance to the node based on the label center.

        Parameters
        ----------
        center : Tuple[float, float]
            given label center
        
        Returns
        -------
        min_dist : float
            minimal distance to any anchor point
        '''
        _w, _h = label_wh(ol.pad_bbox_corners)
        anchors = [
            Anchor(at, *center, *bbox_corners(*center, _w/2, _h/2)).pos
            for at in AnchorType if at != AnchorType.O
        ]
        return min(
            np.hypot(pos[0] - node_pt[0], pos[1] - node_pt[1])
            for pos in anchors
        )

    anchor_candidates.sort(key=min_anchor_dist)
    placed_bboxes = [
        bbox_polygon(*overflow_candidates[o_lid].center, *label_wh(overflow_candidates[o_lid].exp_bbox_corners))
        for o_lid in placed_overflow if o_lid != lid
    ]

    for cx, cy in anchor_candidates:
        if not eroded.contains(Point(cx, cy)):
            continue

        exp_bbox = bbox_polygon(cx, cy, ew, eh)

        # expanded bbox does not fit into the eroded space
        if not space.contains(exp_bbox):
            continue
        
        # expanded bbox overlaps with another overflow label
        if any(exp_bbox.intersects(pb) for pb in placed_bboxes):
            continue
        
        # expanded bbox overlaps another node
        if any(
            exp_bbox.contains(Point(data['pos']))
            for node_id, data in G.nodes(data=True)
            if isinstance(node_id, int) and node_id != ol.node_id
        ):
            continue

        # closest valid anchor
        _w, _h = label_wh(ol.pad_bbox_corners)
        anchors = [
            Anchor(at, *node_pos, *bbox_corners(cx, cy, _w/2, _h/2))
            for at in AnchorType if at != AnchorType.O
        ]
        sorted_anchors = sorted(
            anchors,
            key=lambda anchor: np.hypot(
                anchor.pos[0] - node_pt[0],
                anchor.pos[1] - node_pt[1]
            )
        )

        for anchor in sorted_anchors:
            if binding_line_valid(G, lid, anchor, label_candidates, overflow_candidates, placed_overflow, node_faces(G, overflow_candidates[lid].node_id, bounded_faces), False, 1):
                return cx, cy, anchor
                
    # no valid position found
    return None

def bounded_overflow_labels(
    G: nx.Graph,
    label_candidates: Dict[int, List[LabelCandidate]],
    overflow_candidates: Dict[int, LabelCandidate],
    bounded_faces: List[List[int]],
    centroids: List[Tuple],
    alpha_shape: List[int]
) -> Dict[int, LabelCandidate]:
    '''
    Place overflow labels in the graph interior.

    Parameters
    ----------
    G : nx.Graph
        graph containing positions
    label_candidates : Dict[int, List[LabelCandidate]]
        already placed label candidates
    overflow_candidates : Dict[int, LabelCandidate]
        overflow candidates to be placed
    bounded_faces : List[List]
        bounded faces of the planar graph 
    centroids : List[Tuple]
        centroid of each bounded face
    alpha_shape : List[int]
        boundary walk of the alpha shape

    Returns
    -------
    overflow_candidates : Dict[int, LabelCandidate]
        overflow candidates placed if possible
    '''
    face_data = sorted(
        [
            {
                'fid': id,
                'polygon': Polygon([G.nodes[n]['pos'] for n in face]),
            } 
        for id, face in enumerate(bounded_faces)],
        key=lambda x: x['polygon'].area,
        reverse=True
    )

    # subtract expanded bboxes of placed labels from each face
    placed_union = unary_union([
        Polygon(candidate.pad_bbox_corners)
        for candidates in label_candidates.values()
        for candidate in candidates
    ])

    processed_faces = []
    for fd in face_data:
        poly: Polygon = fd['polygon']
        if not poly.is_valid:
            poly = make_valid(poly)
        processed_faces.append((fd['fid'], poly.difference(placed_union)))

    fitting_faces = _fitting_faces(overflow_candidates, processed_faces)

    space_map: Dict[int, Polygon] = dict(processed_faces)
    placed_overflow: List[int] = []

    face_to_labels: Dict[int, List[int]] = {}
    for lid, face_ids in fitting_faces.items():
        for fid in face_ids:
            face_to_labels.setdefault(fid, []).append(lid)

    alpha_boundary = Polygon([G.nodes[n]['pos'] for n in alpha_shape]).boundary

    while True:
        face_order = sorted(
            space_map.keys(),
            key=lambda fid: space_map[fid].area,
            reverse=True,
        )
        chosen_face = None
        chosen_label = None

        for fid in face_order:
            space = space_map[fid]
            face_centroid = np.array(centroids[fid])

            # labels that possibly fit into the face
            candidates = [
                lid for lid in face_to_labels.get(fid, [])
                if lid not in placed_overflow
                and _fits(
                    space, *label_wh(overflow_candidates[lid].exp_bbox_corners), 
                    type=overflow_candidates[lid].label_type, 
                    node_y=overflow_candidates[lid].center[1]
                )
            ]

            if not candidates:
                continue

            # node closest to the center of remaining space in the face
            incident_candidates = [lid for lid in candidates if lid in bounded_faces[fid]]
            chosen_label = min(
                incident_candidates,
                key=lambda lid: np.linalg.norm(np.array(G.nodes[overflow_candidates[lid].node_id]['pos']) - face_centroid),
                default=None
            )

            if not chosen_label:
                non_incident_candidates = [lid for lid in candidates if lid not in bounded_faces[fid]]
                chosen_label = min(
                    non_incident_candidates,
                    key=lambda lid: np.linalg.norm(np.array(G.nodes[overflow_candidates[lid].node_id]['pos']) - face_centroid),
                    default=None
                )

            chosen_face = fid
            break

        if chosen_face is None or chosen_label is None:
            break

        ol = overflow_candidates[chosen_label]
        space = space_map[chosen_face]

        dist_to_boundary = alpha_boundary.distance(Point(G.nodes[ol.node_id]['pos']))
        dist_to_face = np.linalg.norm(
            np.array(G.nodes[ol.node_id]['pos']) - np.array(centroids[chosen_face])
        )
        if dist_to_boundary < dist_to_face:
            fitting_faces[chosen_label] = [f for f in fitting_faces[chosen_label] if f != chosen_face]
            face_to_labels[chosen_face] = [
                lid for lid in face_to_labels[chosen_face] if lid != chosen_label
            ]
            continue

        fitting_face = _find_valid_position(G, label_candidates, overflow_candidates, placed_overflow, space, bounded_faces, chosen_label)

        if fitting_face is None or np.hypot(fitting_face[2].pos[0] - G.nodes[ol.node_id]['pos'][0],
                fitting_face[2].pos[1] - G.nodes[ol.node_id]['pos'][1]) > dist_to_boundary:
            # if no valid position found block the combination of face and label
            fitting_faces[chosen_label] = [f for f in fitting_faces[chosen_label] if f != chosen_face]
            face_to_labels[chosen_face] = [
                lid for lid in face_to_labels[chosen_face] if lid != chosen_label
            ]
            continue

        cx, cy, anchor = fitting_face
        placed_overflow.append(chosen_label)
        ew, eh = label_wh(overflow_candidates[chosen_label].exp_bbox_corners)
        space_map[chosen_face] = space_map[chosen_face].difference(bbox_polygon(cx, cy, ew, eh))

        ol.update_position(cx, cy, anchor)

        overflow_candidates[chosen_label] = ol

    return _adjust_anchors(G, label_candidates, overflow_candidates)

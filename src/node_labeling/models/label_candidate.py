import numpy as np

from typing import Tuple
from dataclasses import dataclass

from node_labeling.utils.geometry import bbox_corners
from node_labeling.models.anchor import Anchor, AnchorType
from node_labeling.models.label_type import LabelType

@dataclass
class LabelCandidate:
    '''
    One candidate placement for a node label.

    Parameters
    ----------
    node_id : int
        the node assigned to this label
    anchor : Anchor
        which corner or side of the outer bbox is placed at the node position
    label_type : LabelType
        general, extent (objects, below node) or intent (attributes, above node)
    ink_bbox_corners : Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]
        ink bbox (BL, BR, TR, TL)
    pad_bbox_corners : Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]
        padding bbox (BL, BR, TR, TL)
    exp_bbox_corners : Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]
        outer bbox expanded on the free sides (BL, BR, TR, TL)
    center : Tuple[float, float]
        center of the bbox
    text : str
        rendered label content
    '''
    node_id: int
    anchor: Anchor
    label_type: LabelType
    ink_bbox_corners: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    pad_bbox_corners: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    exp_bbox_corners: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    center: Tuple[float, float]
    text: str

    def update_position(self, cx: float, cy: float, anchor: Anchor):
        '''
        Update label center and anchor point.

        Parameters
        ----------
        cx : float
            x-coordinate of new center
        cy : float
            y-coordinate of new center
        anchor : Anchor
            new anchor point
        '''
        ibl, ibr, _, itl = self.ink_bbox_corners
        half_iw = (ibr[0] - ibl[0]) / 2.0
        half_ih = (itl[1] - ibl[1]) / 2.0
        ebl, ebr, _, etl = self.exp_bbox_corners
        orig_px = ((ebr[0] - ebl[0]) / 2.0) - half_iw
        orig_py = ((etl[1] - ebl[1]) / 2.0) - half_ih
        half_ew = half_iw + orig_px
        half_eh = half_ih + orig_py

        is_top    = anchor.anchor_type in [AnchorType.T, AnchorType.TL, AnchorType.TR]
        is_bottom = anchor.anchor_type in [AnchorType.B, AnchorType.BL, AnchorType.BR]
        is_left   = anchor.anchor_type in [AnchorType.L, AnchorType.TL, AnchorType.BL]
        is_right  = anchor.anchor_type in [AnchorType.R, AnchorType.TR, AnchorType.BR]
        corner_multiplier = 1 / np.sqrt(2)
        rows = len(self.text.split(r'\\[-1pt]'))
        padding = ((itl[1] - ibl[1]) / rows - 0.1 * rows) * 0.5
        p_top    = (padding * corner_multiplier) if is_top else padding
        p_bottom = (padding * corner_multiplier) if is_bottom else padding
        p_left   = (padding * corner_multiplier) if is_left else padding
        p_right  = (padding * corner_multiplier) if is_right else padding

        self.center = (cx, cy)
        self.anchor = anchor
        self.ink_bbox_corners = bbox_corners(cx, cy, half_iw, half_ih)
        self.pad_bbox_corners = (
            (cx - half_iw - p_left,  cy - half_ih - p_bottom),
            (cx + half_iw + p_right,  cy - half_ih - p_bottom),
            (cx + half_iw + p_right,  cy + half_ih + p_top),
            (cx - half_iw - p_left,  cy + half_ih + p_top),
        )
        self.exp_bbox_corners = bbox_corners(cx, cy, half_ew, half_eh)

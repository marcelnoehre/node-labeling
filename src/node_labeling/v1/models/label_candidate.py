from typing import Tuple
from dataclasses import dataclass

from node_labeling.v1.utils.geometry import label_wh, label_rows, label_padding, overflow_boxes, binder_anchor
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
        center of the ink bbox (where the text is drawn)
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
        Move an overflow label. The ink box is centered at the new center, the pad and exp boxes are
        rebuilt around it with the same padding on every side (see `overflow_boxes`), so the size of
        the boxes does not depend on the stage that moved the label.

        Parameters
        ----------
        cx : float
            x-coordinate of new center (center of the ink box)
        cy : float
            y-coordinate of new center (center of the ink box)
        anchor : Anchor
            new anchor point (on the padded box)
        '''
        half_iw, half_ih = (v / 2.0 for v in label_wh(self.ink_bbox_corners))
        padding = label_padding(2.0 * half_ih, label_rows(self.text))

        self.center = (cx, cy)
        self.anchor = anchor
        self.ink_bbox_corners, self.pad_bbox_corners, self.exp_bbox_corners = overflow_boxes(
            cx, cy, half_iw, half_ih, padding
        )

    def tighten_anchor(self):
        '''
        Move the binder anchor of a placed overflow label from the pad box to just outside the text
        (see `binder_anchor`). The layout stages keep the anchor on the pad box, so every binder they
        check ends there. The extra piece runs inside the label's own pad box, which no other label,
        binder or edge enters. Call it once, after the last layout stage. Unplaced labels (anchor O) are left as they are.
        '''
        if self.anchor.anchor_type == AnchorType.O:
            return
        half_iw, half_ih = (v / 2.0 for v in label_wh(self.ink_bbox_corners))
        padding = label_padding(2.0 * half_ih, label_rows(self.text))
        self.anchor = binder_anchor(self.anchor.anchor_type, *self.center, half_iw, half_ih, padding)

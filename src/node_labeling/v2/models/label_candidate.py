import math
import shapely

from typing import TYPE_CHECKING, Optional, Tuple
from dataclasses import dataclass

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from node_labeling.v2.utils.geometry import label_wh, label_rows, label_padding, overflow_boxes, binder_anchor
from node_labeling.models.anchor import Anchor, AnchorType
from node_labeling.models.label_type import LabelType
if TYPE_CHECKING:
    from node_labeling.v2.models.candidate import Candidate

DIRECTION_ANCHORS = [AnchorType.L, AnchorType.BL, AnchorType.B, AnchorType.BR,
                     AnchorType.R, AnchorType.TR, AnchorType.T, AnchorType.TL]

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
    ink_shape, pad_shape, exp_shape : Optional[BaseGeometry]
        label shapes of a label placed on its slider (v2). Until the overflow and
        force stages work on shapes, such a label also carries the bounding boxes
        of its shapes, and the stages read its geometry through `ink_poly`,
        `pad_poly` and `exp_poly`.
    '''
    node_id: int
    anchor: Anchor
    label_type: LabelType
    ink_bbox_corners: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    pad_bbox_corners: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    exp_bbox_corners: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    center: Tuple[float, float]
    text: str
    ink_shape: Optional[BaseGeometry] = None
    pad_shape: Optional[BaseGeometry] = None
    exp_shape: Optional[BaseGeometry] = None

    @classmethod
    def from_candidate(cls, candidate: 'Candidate') -> 'LabelCandidate':
        '''
        Wrap a label placed on its slider for the stages that still expect a
        `LabelCandidate`. The anchor sits at the node; its type only names the
        direction from the node to the label (dev plot colors).

        Parameters
        ----------
        candidate : Candidate
            placed label

        Returns
        -------
        label_candidate : LabelCandidate
            the same label with its shapes and their bounding boxes
        '''
        def corners(g: BaseGeometry):
            x0, y0, x1, y1 = g.bounds
            return (x0, y0), (x1, y0), (x1, y1), (x0, y1)

        ink, pad, exp = candidate.ink, candidate.pad, candidate.exp
        shapely.prepare([ink, pad, exp])
        x0, y0, x1, y1 = ink.bounds
        nx_, ny = candidate.node_pos
        angle = math.atan2((y0 + y1) / 2.0 - ny, (x0 + x1) / 2.0 - nx_)
        anchor_type = DIRECTION_ANCHORS[round(angle / (math.pi / 4)) % 8]
        anchor = Anchor(anchor_type, *candidate.position, *corners(pad))
        anchor.pos = (nx_, ny)
        return cls(
            node_id=candidate.node_id, anchor=anchor, label_type=candidate.label_type,
            ink_bbox_corners=corners(ink), pad_bbox_corners=corners(pad), exp_bbox_corners=corners(exp),
            center=candidate.position, text=candidate.text,
            ink_shape=ink, pad_shape=pad, exp_shape=exp
        )

    @property
    def ink_poly(self) -> BaseGeometry:
        '''
        Returns
        -------
        ink : BaseGeometry
            ink shape, or the ink box
        '''
        return self.ink_shape if self.ink_shape is not None else Polygon(self.ink_bbox_corners)

    @property
    def pad_poly(self) -> BaseGeometry:
        '''
        Returns
        -------
        pad : BaseGeometry
            pad shape, or the pad box
        '''
        return self.pad_shape if self.pad_shape is not None else Polygon(self.pad_bbox_corners)

    @property
    def exp_poly(self) -> BaseGeometry:
        '''
        Returns
        -------
        exp : BaseGeometry
            exp shape, or the exp box
        '''
        return self.exp_shape if self.exp_shape is not None else Polygon(self.exp_bbox_corners)

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

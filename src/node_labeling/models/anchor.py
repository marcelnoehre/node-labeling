from dataclasses import dataclass
from enum import Enum
from typing import Tuple

class AnchorType(Enum):
    '''
    (rank, plain, color)
    '''
    L = (1, 'L', 'tab:pink')
    T = (2, 'T', 'tab:purple')
    B = (3, 'B', 'tab:brown')
    R = (4, 'R', 'tab:grey')
    TL = (5, 'TL', 'tab:red')
    BL = (6, 'BL', 'tab:green')
    TR = (7, 'TR', 'tab:blue')
    BR = (8, 'BR', 'tab:orange')
    O = (9, 'Overflow', 'tab:cyan')

    @property
    def rank(self) -> int:
        '''
        rank : int
            rank of the anchor in the readability tie breaker
        '''
        return self.value[0]
    
    @property
    def plain(self) -> str:
        '''
        plain : str
            plain text of anchor name
        '''
        return self.value[1]

    @property
    def color(self) -> str:
        '''
        color : str
            color of anchor for dev output
        '''
        return self.value[2]

@dataclass
class Anchor:
    anchor_type: AnchorType
    pos: Tuple[float, float]

    def __init__(self, anchor_type: AnchorType, cx: float, cy: float, pbl: float, pbr: float, ptr: float, ptl: float):
        self.anchor_type = anchor_type
        self.pos = {
            AnchorType.T: ((ptl[0] + ptr[0]) / 2, ptl[1]),
            AnchorType.B: ((pbl[0] + pbr[0]) / 2, pbl[1]),
            AnchorType.L: (ptl[0], (ptl[1] + pbl[1]) / 2),
            AnchorType.R: (ptr[0], (ptr[1] + pbr[1]) / 2),
            AnchorType.TL: ptl,
            AnchorType.TR: ptr,
            AnchorType.BL: pbl,
            AnchorType.BR: pbr,
            AnchorType.O: (cx, cy)
        }[anchor_type]
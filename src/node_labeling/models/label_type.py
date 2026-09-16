from enum import Enum
from node_labeling.models.anchor import AnchorType

from typing import List

class LabelType(Enum):
    '''
    (LaTeX, anchors)
    '''
    GENERAL = (r'\textrm', [at for at in AnchorType if at != AnchorType.O])
    EXTENT = (r'\textrm', [AnchorType.T, AnchorType.TL, AnchorType.TR])
    INTENT = (r'\textit', [AnchorType.BL, AnchorType.B, AnchorType.BR])

    @property
    def latex(self) -> str:
        '''
        latex : str
            latex mode for label type
        '''
        return self.value[0]

    @property
    def anchors(self) -> List[AnchorType]:
        '''
        -------
        anchors : List[AnchorType]
            valid anchors
        '''
        return self.value[1]
    
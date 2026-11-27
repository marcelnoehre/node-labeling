from dataclasses import dataclass, replace
from functools import cached_property
from typing import Hashable, Optional, Tuple

from shapely import affinity
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from node_labeling.models.label_type import LabelType
from node_labeling.v2.models.binder import Binder
from node_labeling.v2.models.label_shape import LabelShape

@dataclass(frozen=True, eq=False)
class Candidate:
    '''
    A label shape placed at a position (normalized units). Immutable: moving a
    candidate creates a new one (`moved`), so the cached shapes stay valid.

    Attributes
    ----------
    label_id : int
        label id
    node_id : Hashable
        node the label belongs to
    node_pos : Tuple[float, float]
        position of that node
    label_type : LabelType
        general, extent, or intent
    text : str
        formatted LaTeX label
    shape : LabelShape
        ink and pad shape relative to the drawing point
    position : Tuple[float, float]
        drawing point of the text (`ha='center', va='center'`)
    overflow : bool
        whether the label is attached by a binder instead of touching its node
    binder : Optional[Binder]
        binder from the node to the attachment point (placed overflow labels only)
    '''
    label_id: int
    node_id: Hashable
    node_pos: Tuple[float, float]
    label_type: LabelType
    text: str
    shape: LabelShape
    position: Tuple[float, float]
    overflow: bool = False
    binder: Optional[Binder] = None

    @cached_property
    def ink(self) -> BaseGeometry:
        '''
        Returns
        -------
        ink : BaseGeometry
            glyph outlines at the position
        '''
        return affinity.translate(self.shape.ink, *self.position)

    @cached_property
    def pad(self) -> Polygon:
        '''
        Returns
        -------
        pad : Polygon
            padded outline at the position
        '''
        return affinity.translate(self.shape.pad, *self.position)

    @cached_property
    def exp(self) -> Polygon:
        '''
        Returns
        -------
        exp : Polygon
            expanded outline at the position: grown away from the node, or
            symmetric for overflow labels (see `LabelShape.exp`)
        '''
        if self.overflow:
            local = self.shape.overflow_exp
        else:
            local = self.shape.exp((self.node_pos[0] - self.position[0], self.node_pos[1] - self.position[1]))
        return affinity.translate(local, *self.position)

    @cached_property
    def body(self) -> BaseGeometry:
        '''
        Returns
        -------
        body : BaseGeometry
            solid label block at the position (see `LabelShape.body`)
        '''
        return affinity.translate(self.shape.body, *self.position)

    @cached_property
    def pad_hull(self) -> Polygon:
        '''
        Returns
        -------
        hull : Polygon
            convex hull of the pad at the position
        '''
        return affinity.translate(self.shape.pad_hull, *self.position)

    @cached_property
    def dot_ring(self) -> Polygon:
        '''
        Returns
        -------
        ring : Polygon
            outline for the binder dot at the position (see `LabelShape.dot_ring`)
        '''
        return affinity.translate(self.shape.dot_ring, *self.position)

    def moved(self, position: Tuple[float, float], binder: Optional[Binder] = None) -> 'Candidate':
        '''
        Parameters
        ----------
        position : Tuple[float, float]
            new drawing point
        binder : Optional[Binder]
            new binder (overflow labels)

        Returns
        -------
        candidate : Candidate
            the same label at the new position
        '''
        return replace(self, position=(float(position[0]), float(position[1])), binder=binder)

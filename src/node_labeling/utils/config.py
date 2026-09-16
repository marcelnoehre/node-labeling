from dataclasses import dataclass
from node_labeling.models.label_type import LabelType

@dataclass
class Config:
    # dev mode
    plot: bool = False
    runtime: bool = True
    # data
    label_config = {
        LabelType.GENERAL: True,
        LabelType.EXTENT:  True,
        LabelType.INTENT:  True
    }
    # visualization
    font_size: str = r'\footnotesize'
    k_rows: int = 2
    max_row_chars: int = 10
    # grid
    grid_step: float = 0.5
    min_label_dist: float = 1.0
    max_label_dist: float = 2.0
    top_k: int = 100
    # weights (cost)
    w_align: float = 1.0                # anchor alignment
    w_angle: float = 0.5                # natural angle
    w_boundary: float = 3.0             # close to polygon boundary
    w_binder: float = 10.0              # length of binding line
    w_binder_intersect: float = 100.0   # penalty for binder intersecting another label
    w_binder_cross: float = 8.0         # penalty for crossing binding lines
    w_tight_overlap: float = 80.0       # penalty for ink overlaps 
    w_overlap: float = 10.0             # penalty for padding overlaps
    w_padding: float = 5.0              # penalty for padding
    w_miss: float = 1e6                 # penalty for unplaced label
    w_type: float = 100.0               # penalty for labels in the wrong half space
    # hungarian solver
    max_hungarian_iterations: int = 50
    max_hungarian_fixes: int = 3
    iter_penalty_multiplier: float = 2.0
    # weights (forces)
    w_inner_proximity: float = 5.0      # push away from internal drawing
    w_global_proximity: float = 5.0     # push away from other overflow labels
    w_spring: float = 1.0               # pull toward node
    w_binder_dodge: float = 5.0         # push binder away from fixed labels
    w_half_plane: float = 5.0           # keep labels in the correct half plane
    # force refinement
    force_iterations: int = 10000
    force_step_size: float = 0.1
    min_binder_length: float = 0.25

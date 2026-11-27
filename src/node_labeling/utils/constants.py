import re
import matplotlib.cm as cm

# visualization
TARGET_HEIGHT: float = 10.0
PHYSICAL_HEIGHT_MM: float = 100.0
MM_PER_UNIT: float = PHYSICAL_HEIGHT_MM / TARGET_HEIGHT
NODE_SIZE: float = 50.0
NODE_RADIUS: float = 0.2
LINE_WIDTH: float = 1.0
CMAP = cm.YlOrRd
DPI: float = 150.0
FIGURE_PADDING_MM: float = 1.27
COLORBAR_WIDTH_IN: float = 1.3
OUTER_MARGIN = 0.5
BINDER_DOT_SIZE: float = 14.0
BINDER_DOT_GAP_MM: float = 0.2

# rendering
ROW_SKIP_PT: float = -2.0
ROW_BREAK: str = rf'\\[{ROW_SKIP_PT:g}pt]'
SHAPE_TOLERANCE_MM: float = 0.02
PADDING_MM: float = 1.45
PADDING_FONT_PT: float = 8.0
EDGE_CLEARANCE_MM: float = 0.5
NODE_GAP_MM: float = 0.5
LABEL_DISTANCE: float = 2 ** 0.5
OUTER_NODE_DIST: float = 0.25
SLIDER_STEP_MM: float = 0.1
SLIDER_TOL_MM: float = 1e-3
PLACEMENT_STEP_MM: float = 0.25
LATEX_CHARS = frozenset('\\{_^')
MATH_RE = re.compile(r'(\$[^$]+\$)')
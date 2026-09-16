import re
import matplotlib.cm as cm

# visualization
TARGET_HEIGHT: float = 10.0
PHYSICAL_HEIGHT_MM: float = 100.0
NODE_SIZE: float = 50.0
NODE_RADIUS: float = 0.2
LINE_WIDTH: float = 1.0
CMAP = cm.YlOrRd
DPI: float = 150.0
MARGIN: float = 1.0
OUTER_MARGIN = 0.5

# rendering
LATEX_CHARS = frozenset('\\{_^')
MATH_RE = re.compile(r'(\$[^$]+\$)')
# Node Labeling
This repository implements a two-phase algorithm for labeling line diagrams of ordered sets.

## Usage

### Install via PyPI
Install the package from [PyPI](https://pypi.org/project/node-labeling/) using pip.

```bash
pip install node-labeling
```

Execute the layout generation using the `node-labeling` command.
```bash
node-labeling
```

### Install from source
Clone the repository and initialize the environment. uv will automatically create a virtual
environment and sync dependencies based on the pyproject.toml.

```bash
uv sync
```

Execute the layout generation using uv run to ensure the correct environment context.
```bash
uv run node-labeling
```

The script will prompt you for a path to a `.graphml` file. Each node needs `x`/`y` position data,
and optionally `general` data as a plain string and `extent`/`intent` label data (`;`-separated
names).
To export the final layout to a PDF add the `--export` flag.
```bash
uv run node-labeling --export
```
The `--export` flag also works with the PyPI install:
```bash
node-labeling --export
```

### Use as a library
Node Labeling can also be called directly from Python via `node_labeling.label`.

```python
from node_labeling import label

# graphml is a path to a .graphml file
labels = label('path/to/graph.graphml')
```

By default `label` returns the final label placement for every node as a dict mapping label id to
`LabelCandidate`. Pass `export=True` to instead write the final layout to a PDF in `figs/`.
```python
label('path/to/graph.graphml', export=True)
```

Pass a `Config` instance to override any parameter listed below.
```python
from node_labeling import label
from node_labeling.utils.config import Config

labels = label('path/to/graph.graphml', cfg=Config(top_k=200))
```

## Configuration
| Category | Parameter | Description |
| :--- | :--- | :--- |
| **Dev Mode** | `plot` | Whether to plot intermediate steps |
| | `runtime` | Whether to display runtime information |
| **Data** | `label_config` | Label configuration |
| **Visualize** | `font_size` | Font size for labels |
| | `k_rows` | Number of rows for layout |
| | `max_row_chars` | Maximum characters per row |
| **Grid**| `grid_step` | Grid step size |
| | `min_label_dist` | Minimum label distance from drawing |
| | `max_label_dist` | Maximum label distance from drawing |
| | `top_k` | Top k labels to consider |
| **Weights (cost)** | `w_align` | Anchor alignment |
| | `w_angle` | Natural angle |
| | `w_boundary` | Close to polygon boundary |
| | `w_binder` | Length of binding line |
| | `w_binder_intersect` | Penalty for binder intersecting another label |
| | `w_binder_cross` | Penalty for crossing binding lines |
| | `w_tight_overlap` | Penalty for ink overlaps |
| | `w_overlap` | Penalty for padding overlaps |
| | `w_padding` | Penalty for padding |
| | `w_miss` | Penalty for unplaced label |
| | `w_type` | Penalty for label type mismatch |
| **Hungarian Solver** | `max_hungarian_iterations` | Maximum iterations for the Hungarian solver |
| | `max_hungarian_fixes` | Maximum fixes for the Hungarian solver |
| | `iter_penalty_multiplier` | Multiplier for penalties in each iteration of the Hungarian solver |
| **Forces** | `w_inner_proximity` | Weight for pushing away from internal drawing |
| | `w_global_proximity` | Weight for pushing away from other overflow labels |
| | `w_spring` | Weight for pulling toward node |
| | `w_binder_dodge` | Weight for pushing binder away from fixed labels |
| | `w_half_plane` | Weight for keeping labels in the correct half plane |
| | `force_iterations` | Number of iterations for force refinement |
| | `force_step_size` | Step size for force refinement |
| | `min_binder_length` | Minimum length for binding lines |

## License
Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
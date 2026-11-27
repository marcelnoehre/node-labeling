import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from shapely.geometry.base import BaseGeometry
from typing import Dict, List, Tuple

from node_labeling.utils.constants import (
    BINDER_DOT_SIZE, CMAP, COLORBAR_WIDTH_IN, DPI, FIGURE_PADDING_MM, LINE_WIDTH, MM_PER_UNIT, NODE_SIZE
)
from node_labeling.models.label_type import LabelType
from node_labeling.v2.label.sliding import SlidingLabel
from node_labeling.v2.models.candidate import Candidate

TYPE_COLORS = {LabelType.GENERAL: 'tab:green', LabelType.EXTENT: 'tab:blue', LabelType.INTENT: 'tab:purple'}
OVERFLOW_COLOR = 'tab:orange'
UNPLACED_COLOR = 'tab:red'

def _color(candidate: Candidate) -> str:
    '''
    Dev plot color: by label type, overflow labels orange, unplaced ones red.

    Parameters
    ----------
    candidate : Candidate
        label

    Returns
    -------
    color : str
        matplotlib color
    '''
    if candidate.overflow:
        return OVERFLOW_COLOR if candidate.binder is not None else UNPLACED_COLOR
    return TYPE_COLORS[candidate.label_type]

def _rings(geom: BaseGeometry) -> List[np.ndarray]:
    '''
    Parameters
    ----------
    geom : BaseGeometry
        Polygon or MultiPolygon

    Returns
    -------
    rings : List[np.ndarray]
        exterior coordinates of every polygon
    '''
    return [np.asarray(p.exterior.coords) for p in getattr(geom, 'geoms', [geom]) if not p.is_empty]

def _content_bounds(
        G: nx.Graph,
        candidates: List[Candidate],
        dev: bool
) -> Tuple[float, float, float, float]:
    '''
    Bounds of everything that is drawn: nodes (with their marker), letters (dev:
    exp shapes) and binder attachments.

    Parameters
    ----------
    G : nx.Graph
        graph containing positions
    candidates : List[Candidate]
        all drawn labels and grid positions
    dev : bool
        wether the pad and exp shapes are drawn as well

    Returns
    -------
    bounds : Tuple[float, float, float, float]
        (min_x, min_y, max_x, max_y) including the figure padding
    '''
    marker_radius = (np.sqrt(NODE_SIZE) / 2.0 + LINE_WIDTH / 2.0) * 25.4 / 72.0 / MM_PER_UNIT
    xs, ys = [], []
    for _, data in G.nodes(data=True):
        x, y = data['pos']
        xs += [x - marker_radius, x + marker_radius]
        ys += [y - marker_radius, y + marker_radius]

    for candidate in candidates:
        x0, y0, x1, y1 = (candidate.exp if dev else candidate.ink).bounds
        xs += [x0, x1]
        ys += [y0, y1]
        attach = candidate.binder.end if candidate.binder is not None else candidate.node_pos
        xs.append(attach[0])
        ys.append(attach[1])

    pad = FIGURE_PADDING_MM / MM_PER_UNIT
    return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad

def _draw_shapes(ax: Axes, candidate: Candidate, color: str):
    '''
    Dev view of a label: pad filled, letters outlined, exp dotted.

    Parameters
    ----------
    ax : Axes
        the axes of the figure
    candidate : Candidate
        label
    color : str
        color
    '''
    for ring in _rings(candidate.pad):
        ax.add_patch(mpatches.Polygon(ring, closed=True, facecolor=color, edgecolor=color, alpha=0.25,
                                      linewidth=0.8, zorder=3, clip_on=False))
    for ring in _rings(candidate.exp):
        ax.add_patch(mpatches.Polygon(ring, closed=True, facecolor='none', edgecolor=color, alpha=0.55,
                                      linestyle=':', linewidth=1.0, zorder=3, clip_on=False))
    for ring in _rings(candidate.ink):
        ax.add_patch(mpatches.Polygon(ring, closed=True, facecolor='none', edgecolor=color, alpha=0.9,
                                      linewidth=0.4, zorder=4, clip_on=False))

def _draw_label(G: nx.Graph, ax: Axes, candidate: Candidate, dev: bool = False):
    '''
    Draw a label: its text at the drawing point and, for overflow labels, the
    binder with its dot. Dev mode adds the shapes.

    Parameters
    ----------
    G : nx.Graph
        graph containing positions
    ax : Axes
        the axes of the figure
    candidate : Candidate
        label
    dev : bool
        wether to draw the shapes
    '''
    color = _color(candidate)
    if dev:
        _draw_shapes(ax, candidate, color)

    binder = candidate.binder
    if binder is not None:
        ax.scatter(*binder.end, color='grey', s=BINDER_DOT_SIZE, linewidths=0, zorder=6, alpha=0.8, clip_on=False)
        style = dict(color='grey', linestyle='--', linewidth=0.5, alpha=0.4, zorder=4)
        if dev and not binder.straight:
            style.update(color='tab:red', alpha=0.9)
        if binder.straight:
            nx_, ny = G.nodes[candidate.node_id]['pos']
            ax.plot([binder.end[0], nx_], [binder.end[1], ny], clip_on=False, **style)
        else:
            ax.add_patch(mpatches.PathPatch(binder.path(), facecolor='none', edgecolor=style['color'],
                                            linestyle='--', linewidth=0.5, alpha=0.4, zorder=4, clip_on=False))

    ax.text(
        *candidate.position, candidate.text,
        ha='center', va='center',
        color=color if dev else 'black',
        clip_on=False,
        alpha=1.0,
        zorder=7,
    )

def _draw_slider(ax: Axes, label: SlidingLabel):
    '''
    Dev view of a slider: the path of the label's center line point (dashed),
    feasible positions (light, thick) and preferred ones (dark).

    Parameters
    ----------
    ax : Axes
        the axes of the figure
    label : SlidingLabel
        label with slider and feasible set
    '''
    color = TYPE_COLORS[label.label_type] if not label.feasible.intervals.is_empty else '0.6'

    def arc(a: float, b: float, **kw):
        t = np.linspace(a, b, max(2, int((b - a) / 0.005)))
        P = label.slider.positions(t) + [0.0, label.slider.center_y]
        ax.plot(*P.T, solid_capstyle='butt', clip_on=False, **kw)

    arc(0.0, label.slider.length, color=color, linewidth=0.5, linestyle='--', zorder=3)
    for a, b in label.feasible.intervals:
        arc(a, b, color=color, linewidth=3.0, alpha=0.3, zorder=3)
    for a, b in label.feasible.preferred:
        arc(a, b, color=color, linewidth=1.4, zorder=3)

def plot_graph(
        G: nx.Graph,
        nodes: List[int],
        edges: List[Tuple[int, int]],
        output_path: str,
        title: str = '',
        # data
        intersections: List[Tuple] = [],
        show_intersections: bool = False,
        # alpha shape
        alpha_shape: List[int] = [],
        show_alpha_shape: bool = False,
        # faces
        bounded_faces: List[List[int]] = [],
        areas: List[float] = [],
        centroids: List[Tuple[float, float]] = [],
        show_face_areas: bool = False,
        show_face_sizes: bool = False,
        # labels
        label_candidates: Dict[int, Candidate] = {},
        colored_label_candidates: bool = False,
        show_legend: bool = False,
        # overflow labels
        overflow_candidates: Dict[int, Candidate] = {},
        # grid positions of the exterior labels
        grid_candidates: Dict[int, List[Tuple[Candidate, float]]] = {},
        # sliders
        sliders: List[SlidingLabel] = []
) -> None:
    '''
    Draw the graph and save to a PDF.

    The figure is sized so that one normalized unit is exactly MM_PER_UNIT mm, which is the scale
    the label shapes were measured for (text is rendered at a fixed point size).

    Parameters
    ----------
    G : nx.Graph
        planar graph containing normalized positions
    nodes : List[int]
        original (non-dummy) nodes
    edges : List[Tuple[int, int]]
        cover relations
    output_path : str
        path of the PDF
    title : str
        window title
    intersections, show_intersections
        edge crossings and wether to draw them
    alpha_shape, show_alpha_shape
        boundary walk of the alpha shape and wether to draw it
    bounded_faces, areas, centroids, show_face_areas, show_face_sizes
        bounded faces with their area (mm^2) and centroid, wether to draw them and their size
    label_candidates, colored_label_candidates, show_legend
        labels placed at their nodes, wether to draw their shapes (dev mode, colored by label
        type), wether to draw the color legend
    overflow_candidates : Dict[int, Candidate]
        overflow labels
    grid_candidates : Dict[int, List[Tuple[Candidate, float]]]
        grid positions of the exterior labels (drawn as points)
    sliders : List[SlidingLabel]
        sliders with feasible sets to draw
    '''
    dev = colored_label_candidates
    drawn = list(label_candidates.values()) + list(overflow_candidates.values())
    min_x, min_y, max_x, max_y = _content_bounds(G, drawn, dev)
    grid_pts = np.array([c.position for cands in grid_candidates.values() for c, _ in cands]).reshape(-1, 2)
    slider_pts = np.vstack([L.slider.positions(np.linspace(0, L.slider.length, 50)) + [0, L.slider.center_y]
                            for L in sliders]) if sliders else np.empty((0, 2))
    extra = np.vstack([grid_pts, slider_pts])
    if len(extra):
        pad = FIGURE_PADDING_MM / MM_PER_UNIT
        min_x, min_y = min(min_x, extra[:, 0].min() - pad), min(min_y, extra[:, 1].min() - pad)
        max_x, max_y = max(max_x, extra[:, 0].max() + pad), max(max_y, extra[:, 1].max() + pad)
    width_in = (max_x - min_x) * MM_PER_UNIT / 25.4
    height_in = (max_y - min_y) * MM_PER_UNIT / 25.4
    show_colorbar = bool(show_face_areas and bounded_faces and areas)
    total_width_in = width_in + (COLORBAR_WIDTH_IN if show_colorbar else 0.0)

    fig = plt.figure(figsize=(total_width_in, height_in), dpi=DPI)
    ax = fig.add_axes((0.0, 0.0, width_in / total_width_in, 1.0))
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(title)

    ##### vertices #####
    for nid in nodes:
        x, y = G.nodes[nid]['pos']
        ax.scatter(x, y, facecolor='white', edgecolor='black', linewidth=LINE_WIDTH, s=NODE_SIZE, zorder=100)

    ##### edges #####
    for i, j in edges:
        x0, y0 = G.nodes[i]['pos']
        x1, y1 = G.nodes[j]['pos']
        ax.plot([x0, x1], [y0, y1], color='black', linewidth=LINE_WIDTH, zorder=2)

    ##### intersections #####
    if show_intersections:
        for pt in intersections:
            ax.scatter(
                pt[0], pt[1],
                facecolor='white',
                edgecolor='tab:red',
                linewidth=LINE_WIDTH,
                s=NODE_SIZE,
                zorder=10
            )

    ##### alpha_shape #####
    if show_alpha_shape and alpha_shape:
        pts = [G.nodes[nid]['pos'] for nid in alpha_shape]
        for i in range(len(pts)):
            p1 = pts[i]
            p2 = pts[(i + 1) % len(pts)]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='tab:red', linestyle='-', linewidth=LINE_WIDTH, zorder=3)

    ##### faces #####
    if show_face_areas and bounded_faces and areas:
        norm = plt.Normalize(min(areas), max(areas))
        for face, area, centroid in zip(bounded_faces, areas, centroids):
            pts = [G.nodes[nid]['pos'] for nid in face]
            patch = mpatches.Polygon(
                pts, closed=True,
                facecolor=CMAP(norm(area)), edgecolor='none',
                alpha=0.4, zorder=1,
            )
            ax.add_patch(patch)
            if show_face_sizes:
                ax.annotate(
                    f'{area:.2f}', xy=centroid,
                    ha='center', va='center',
                    fontsize=9, color='black', zorder=5,
                )
        sm = cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(min(areas), max(areas)))
        sm.set_array([])
        cax = fig.add_axes((
            (width_in + 0.1) / total_width_in, 0.1,
            0.15 / total_width_in, 0.8
        ))
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label(r'Face area ($\mathrm{mm}^2$)', fontsize=16)
        cb.ax.tick_params(labelsize=14)

    ##### sliders #####
    for label in sliders:
        _draw_slider(ax, label)

    ##### labels #####
    for candidate in label_candidates.values():
        _draw_label(G, ax, candidate, dev)
    for candidate in overflow_candidates.values():
        _draw_label(G, ax, candidate, dev)

    ##### grid positions #####
    for lid, cands in grid_candidates.items():
        if cands:
            P = np.array([c.position for c, _ in cands])
            ax.scatter(*P.T, s=4, color=OVERFLOW_COLOR, alpha=0.5, linewidths=0, zorder=5, clip_on=False)

    if dev and show_legend:
        handles = [mpatches.Patch(facecolor=c, edgecolor=c, alpha=0.6, label=t.name.lower()) for t, c in TYPE_COLORS.items()]
        handles += [mpatches.Patch(facecolor=OVERFLOW_COLOR, alpha=0.6, label='overflow'),
                    mpatches.Patch(facecolor=UNPLACED_COLOR, alpha=0.6, label='not placed')]
        ax.legend(handles=handles, loc='best', fontsize=8, title='Label', framealpha=0.8)

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    fig.savefig(output_path, format='pdf')
    plt.close(fig)

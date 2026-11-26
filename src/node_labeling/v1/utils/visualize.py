import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Dict, List, Tuple

from node_labeling.utils.constants import *
from node_labeling.models.anchor import AnchorType
from node_labeling.models.label_type import LabelType
from node_labeling.v1.models.label_candidate import LabelCandidate

def _ink_center(candidate: LabelCandidate) -> Tuple[float, float]:
    '''
    Center of the ink box, i.e. where the text has to be drawn.

    Parameters
    ----------
    candidate : LabelCandidate
        label candidate

    Returns
    -------
    center : Tuple[float, float]
        center of the ink box
    '''
    (x0, y0), _, (x1, y1), _ = candidate.ink_bbox_corners
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0

def _content_bounds(
        G: nx.Graph,
        candidates: List[LabelCandidate],
        colored_label_candidates: bool
) -> Tuple[float, float, float, float]:
    '''
    Bounds of everything that is drawn: nodes (with their marker) and label boxes.

    Parameters
    ----------
    G : nx.Graph
        graph containing positions
    candidates : List[LabelCandidate]
        all drawn label, overflow and grid candidates
    colored_label_candidates : bool
        wether the pad and exp boxes are drawn as well

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
        boxes = [candidate.ink_bbox_corners]
        if colored_label_candidates:
            boxes += [candidate.pad_bbox_corners, candidate.exp_bbox_corners]
        for corners in boxes:
            xs += [corner[0] for corner in corners]
            ys += [corner[1] for corner in corners]
        xs.append(candidate.anchor.pos[0])
        ys.append(candidate.anchor.pos[1])

    pad = FIGURE_PADDING_MM / MM_PER_UNIT
    return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad

def _draw_candidate(
        ax: Axes,
        candidate: LabelCandidate,
        colored_label_candidates: bool = False
):
    '''
    Place the label candidate and visualize its bounding boxes.

    ax : Axes
        the axes of the figure
    candidate : LabelCandidate
        label candidate to be visualized
    colored_label_candidates : bool
        wether to draw the bounding boxes
    '''
    if colored_label_candidates:
        color = candidate.anchor.anchor_type.color
        ax.add_patch(mpatches.Polygon(
            candidate.ink_bbox_corners, closed=True, facecolor='none', edgecolor=color,
            alpha=0.9, linestyle='-', linewidth=0.8, zorder=4, clip_on=False
        ))
        ax.add_patch(mpatches.Polygon(
            candidate.pad_bbox_corners, closed=True, facecolor=color, edgecolor=color,
            alpha=0.30, linestyle='-', linewidth=1.6, zorder=3, clip_on=False
        ))
        ax.add_patch(mpatches.Polygon(
            candidate.exp_bbox_corners, closed=True,
            facecolor='none', edgecolor=color,
            alpha=0.55, linestyle=':', linewidth=1.0,
            zorder=3, clip_on=False
        ))

        anchor_pos = candidate.anchor.pos
        ax.scatter(*anchor_pos, color=color, s=30, zorder=10, alpha=0.9, clip_on=False)

    text_color = color if colored_label_candidates else 'black'
    cx, cy = _ink_center(candidate)

    ax.text(
        cx, cy, candidate.text,
        ha='center', va='center',
        color=text_color,
        clip_on=False,
        alpha=1.0,
        zorder=7,
    )

def _draw_overflow_candidate(
        G: nx.Graph,
        ax: Axes,
        candidate: LabelCandidate,
        colored_label_candidates: bool = False
):
    '''
    Place the overflow candidate and visualize its bounding boxes.

    G : nx.Graph
        graph containing positions
    ax : Axes
        the axes of the figure
    candidate : LabelCandidate
        overflow candidate to be visualized
    colored_label_candidates : bool
        wether to draw the bounding boxes
    '''
    if colored_label_candidates:
        color = candidate.anchor.anchor_type.color
        ax.add_patch(mpatches.Polygon(
            candidate.ink_bbox_corners, closed=True,
            facecolor='none', edgecolor=color,
            alpha=0.9, linestyle='-', linewidth=0.8,
            zorder=4, clip_on=False
        ))
        ax.add_patch(mpatches.Polygon(
            candidate.pad_bbox_corners, closed=True,
            facecolor=color, edgecolor=color,
            alpha=0.30, linestyle='-', linewidth=1.6,
            zorder=3, clip_on=False
        ))

        ax.scatter(*candidate.anchor.pos, color=color, s=30, zorder=6, alpha=0.9, clip_on=False)

        ax.add_patch(mpatches.Polygon(
            candidate.exp_bbox_corners, closed=True,
            facecolor='none', edgecolor=color,
            alpha=0.55, linestyle=':', linewidth=1.0,
            zorder=3, clip_on=False
        ))

    if candidate.anchor.anchor_type != AnchorType.O:
        ax.scatter(*candidate.anchor.pos, color='grey', s=BINDER_DOT_SIZE, linewidths=0, zorder=6, alpha=0.8, clip_on=False)
        nx_, ny = G.nodes[candidate.node_id]['pos']
        ax.plot(
            [candidate.anchor.pos[0], nx_], 
            [candidate.anchor.pos[1], ny], 
            color='grey', linestyle='--', 
            linewidth=0.5, alpha=0.4, 
            zorder=4, clip_on=False
        )

    text_color = color if colored_label_candidates else 'black'
    cx, cy = _ink_center(candidate)

    ax.text(
        cx, cy, candidate.text,
        ha='center', va='center',
        color=text_color,
        clip_on=False,
        alpha=1.0,
        zorder=7,
    )

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
        # label candidates
        label_candidates: Dict[int, List[LabelCandidate]] = {},
        colored_label_candidates: bool = False,
        show_legend: bool = False,
        # overflow candidates
        overflow_candidates: Dict[int, LabelCandidate] = {},
        # grid candidates
        grid_candidates: Dict[int, List[Tuple[LabelCandidate, float]]] = {}
) -> None:
    '''
    Draw the graph and save to a PDF.

    The figure is sized so that one normalized unit is exactly MM_PER_UNIT mm, which is the scale
    the label boxes were measured for (text is rendered at a fixed point size).

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
        label candidates, wether to draw their boxes and anchors, wether to draw the anchor legend
    overflow_candidates : Dict[int, LabelCandidate]
        overflow labels
    grid_candidates : Dict[int, List[Tuple[LabelCandidate, float]]]
        grid candidates of the exterior labels
    '''
    drawn = (
        [candidate for candidates in label_candidates.values() for candidate in candidates]
        + list(overflow_candidates.values())
        + [candidate for candidates in grid_candidates.values() for candidate, _ in candidates]
    )
    min_x, min_y, max_x, max_y = _content_bounds(G, drawn, colored_label_candidates)
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

    ##### label candidates #####
    if label_candidates:
        for candidates in label_candidates.values():
            for candidate in candidates:
                _draw_candidate(ax, candidate, colored_label_candidates)

        if colored_label_candidates:
            legend_handles = [
                mpatches.Patch(facecolor=at.color, edgecolor=at.color, alpha=0.6, label=at.plain)
                for at in AnchorType if at != AnchorType.O
            ]
            if show_legend:
                ax.legend(handles=legend_handles, loc='upper left', fontsize=8, title='Label anchor', framealpha=0.8)
    
    if overflow_candidates:
        for overflow_candidate in overflow_candidates.values():
            _draw_overflow_candidate(G, ax, overflow_candidate, colored_label_candidates)

    if grid_candidates:
        for candidates in grid_candidates.values():
            for grid_candidate, _ in candidates:
                _draw_overflow_candidate(G, ax, grid_candidate, colored_label_candidates)

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    fig.savefig(output_path, format='pdf')
    plt.close(fig)

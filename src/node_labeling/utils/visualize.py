import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Dict, List, Tuple

from node_labeling.utils.constants import *
from node_labeling.models.anchor import AnchorType
from node_labeling.models.label_type import LabelType
from node_labeling.models.label_candidate import LabelCandidate

def _trim_figure(fig: Figure, ax: Axes) -> None:
    '''
    Trim whitespace around the drawing by:
        1. rasterizing the figure to an RGBA array
        2. finding the bounding box of all non-white pixels
        3. resizing the figure and shifting the axes

    Parameters
    ----------
    fig : Figure
        the figure to trim
    ax : Axes
        the axes of the figure
    '''
    # 1. rasterize
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    rgba = np.frombuffer(buf, dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
    rgb = rgba[:, :, :3]

    # 2. bbox if white px
    is_content = np.any(rgb < 250, axis=2)
    rows = np.any(is_content, axis=1)
    cols = np.any(is_content, axis=0)

    if not rows.any():
        return

    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    # 3. resize
    dpi = fig.get_dpi()
    fig_w_px, fig_h_px = fig.canvas.get_width_height()
    pad_px = int(round(0.05 * dpi))
    col_min = max(0,        col_min - pad_px)
    col_max = min(fig_w_px, col_max + pad_px)
    row_min = max(0,        row_min - pad_px)
    row_max = min(fig_h_px, row_max + pad_px)
    new_w_in = (col_max - col_min) / dpi
    new_h_in = (row_max - row_min) / dpi
    ax_pos = ax.get_position()
    ax_l_px = ax_pos.x0 * fig_w_px
    ax_b_px = ax_pos.y0 * fig_h_px
    ax_w_px = ax_pos.width  * fig_w_px
    ax_h_px = ax_pos.height * fig_h_px
    crop_b_px = fig_h_px - row_max
    crop_l_px = col_min
    new_ax_l = (ax_l_px - crop_l_px) / (col_max - col_min)
    new_ax_b = (ax_b_px - crop_b_px) / (row_max - row_min)
    new_ax_w = ax_w_px / (col_max - col_min)
    new_ax_h = ax_h_px / (row_max - row_min)

    fig.set_size_inches(new_w_in, new_h_in)
    ax.set_position([new_ax_l, new_ax_b, new_ax_w, new_ax_h])

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
    cx, cy = candidate.center

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
        ax.scatter(*candidate.anchor.pos, color='grey', s=5, zorder=6, alpha=0.8, clip_on=False)
        nx_, ny = G.nodes[candidate.node_id]['pos']
        ax.plot(
            [candidate.anchor.pos[0], nx_], 
            [candidate.anchor.pos[1], ny], 
            color='grey', linestyle='--', 
            linewidth=0.5, alpha=0.4, 
            zorder=4, clip_on=False
        )

    text_color = color if colored_label_candidates else 'black'
    cx, cy = candidate.center

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

    Parameters
    ----------
    TODO
    '''
    fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI)
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
        cb = plt.colorbar(sm, ax=ax, shrink=0.8)
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

    xs = [G.nodes[nid]['pos'][0] for nid in G.nodes]
    ys = [G.nodes[nid]['pos'][1] for nid in G.nodes]
    ax.set_xlim(min(xs) - MARGIN, max(xs) + MARGIN)
    ax.set_ylim(min(ys) - MARGIN, max(ys) + MARGIN)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    _trim_figure(fig, ax)

    plt.savefig(output_path, format='pdf')
    plt.close('all')
    plt.close(fig)

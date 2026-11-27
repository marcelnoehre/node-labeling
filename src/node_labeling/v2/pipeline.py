import time

from pathlib import Path
from typing import Dict, Optional

from node_labeling.utils.config import Config
from node_labeling.v2.utils.visualize import plot_graph
from node_labeling.utils.normalize import *
from node_labeling.utils.graph_lattice import GraphLattice
from node_labeling.topology.planarize import *
from node_labeling.topology.faces import *
from node_labeling.v2.filter.shape_rules import build_obstacles
from node_labeling.v2.label.sliding import collect_labels, prepare_labels, place_labels
from node_labeling.v2.models.candidate import Candidate
from node_labeling.v2.overflow.binders import Layout
from node_labeling.v2.overflow.freespace import place_overflow

def label(graphml: str, cfg: Optional[Config] = None, export: bool = False) -> Optional[Dict[int, Candidate]]:
    '''
    Run the v2 node labeling pipeline end to end.

    Parameters
    ----------
    graphml : str
        Path to a .graphml file. Each node needs `x`/`y` position data, and
        optionally `general` data as a plain string and `extent`/`intent`
        label data (`;`-separated names).
    cfg : Config, optional
        Configuration overrides. Defaults to `Config()`.
    export : bool
        If True, write the final layout to a PDF in `figs/` and return None.
        If False, skip the export and return the computed label placements instead.

    Returns
    -------
    labels : Optional[Dict[int, Candidate]]
        The final label for each label id (position, shapes, binder for overflow labels),
        or None if export is True.
    '''
    init_time = time.perf_counter()
    cfg = cfg or Config()
    if cfg.plot or export:
        Path('figs').mkdir(exist_ok=True)

    lattice = GraphLattice(graphml)
    nodes = lattice.nodes
    relations = list(lattice.cover_relations())
    positions = lattice.positions

    # planarize graph
    start_time = time.perf_counter()
    intersections = find_intersections(relations, positions)
    G = build_planar_graph(relations, positions, intersections)
    planarize_duration = (time.perf_counter() - start_time) * 1000
    if cfg.runtime:
        print(f"\033[1;36m[RUNTIME]\033[0m Planarize runtime: \033[1;32m{planarize_duration:.2f} ms\033[0m")

    # normalize positions
    normalize_positions(G)
    intersection_points = normalize_intersections(G, intersections)
    if cfg.plot:
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/input.pdf'
        )
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/intersections.pdf',
            intersections=intersection_points,
            show_intersections=True
        )

    # faces
    start_time = time.perf_counter()
    alpha_shape, bounded_faces, areas, centroids = extract_faces(G)
    topology_duration = (time.perf_counter() - start_time) * 1000
    if cfg.runtime:
        print(f"\033[1;36m[RUNTIME]\033[0m Compute topology: \033[1;32m{topology_duration:.2f} ms\033[0m")
    if cfg.plot:
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/faces.pdf',
            intersections=intersection_points,
            show_intersections=True,
            alpha_shape=alpha_shape,
            show_alpha_shape=True,
            bounded_faces=bounded_faces,
            areas=areas,
            centroids=centroids,
            show_face_areas=True
        )

    # labels on sliders: shapes, sliders and feasible sets (T5/T6)
    start_time = time.perf_counter()
    obstacles = build_obstacles(G, lattice, alpha_shape)
    sliding_labels = prepare_labels(G, collect_labels(lattice, cfg), obstacles, cfg)
    feasible_duration = (time.perf_counter() - start_time) * 1000
    if cfg.runtime:
        print(f"\033[1;36m[RUNTIME]\033[0m Sliders and feasible sets: \033[1;32m{feasible_duration:.2f} ms\033[0m")

    # continuous placement on the sliders
    start_time = time.perf_counter()
    placed = place_labels(sliding_labels, obstacles)
    direct = {lid: candidate for lid, candidate in placed.items() if candidate is not None}
    overflow = {
        L.label_id: Candidate(
            label_id=L.label_id, node_id=L.node_id, node_pos=L.slider.node, label_type=L.label_type,
            text=L.text, shape=L.shape, position=L.slider.node, overflow=True
        )
        for L in sliding_labels if placed[L.label_id] is None
    }
    layout = Layout.build(G, obstacles, list(direct.values()), bounded_faces, alpha_shape)
    placement_duration = (time.perf_counter() - start_time) * 1000
    if cfg.runtime:
        print(f"\033[1;36m[RUNTIME]\033[0m Sliding placement: \033[1;32m{placement_duration:.2f} ms\033[0m")
    if cfg.plot:
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/placed.pdf',
            label_candidates=direct,
            colored_label_candidates=True,
            show_legend=True
        )
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/sliders.pdf',
            label_candidates=direct,
            colored_label_candidates=True,
            sliders=sliding_labels
        )
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/all_overflow_candidates.pdf',
            label_candidates=direct,
            colored_label_candidates=True,
            overflow_candidates=overflow
        )

    start_time = time.perf_counter()
    overflow, spots = place_overflow(layout, overflow, cfg)
    overflow_duration = (time.perf_counter() - start_time) * 1000
    if cfg.runtime:
        print(f"\033[1;36m[RUNTIME]\033[0m Overflow placement: \033[1;32m{overflow_duration:.2f} ms\033[0m")
    if cfg.plot:
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/overflow_positions.pdf',
            label_candidates=direct,
            colored_label_candidates=True,
            overflow_candidates=overflow,
            grid_candidates=spots
        )

    if cfg.plot:
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/final.pdf',
            label_candidates=direct,
            colored_label_candidates=True,
            overflow_candidates=overflow
        )

    total_duration = (time.perf_counter() - init_time) * 1000
    if cfg.runtime:
        print(f"\033[1;36m[RUNTIME]\033[0m Total runtime: \033[1;32m{total_duration:.2f} ms\033[0m")

    if export:
        plot_graph(
            G,
            nodes,
            relations,
            output_path=f'figs/{Path(graphml).stem}.pdf',
            label_candidates=direct,
            overflow_candidates=overflow
        )
        return None

    return dict(sorted({**direct, **overflow}.items()))

import time

from pathlib import Path
from typing import Dict, Optional

from node_labeling.utils.config import Config
from node_labeling.v2.utils.visualize import plot_graph
from node_labeling.utils.normalize import *
from node_labeling.utils.graph_lattice import GraphLattice
from node_labeling.v2.models.label_candidate import LabelCandidate
from node_labeling.topology.planarize import *
from node_labeling.topology.faces import *
from node_labeling.v2.label.generate_candidates import *
from node_labeling.v2.filter.shape_rules import build_obstacles
from node_labeling.v2.label.sliding import prepare_labels, place_labels
from node_labeling.v2.label.generate_overflow import *
from node_labeling.v2.overflow.bounded import *
from node_labeling.v2.overflow.unbounded import *
from node_labeling.v2.forces.forces import *

def label(graphml: str, cfg: Optional[Config] = None, export: bool = False) -> Optional[Dict[int, LabelCandidate]]:
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
    labels : Optional[Dict[int, LabelCandidate]]
        The final placed label candidate for each label id, or None if export is True.
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

    # labels on sliders: shapes, sliders and feasible sets (T5/T6). The box candidates are only kept
    # for the overflow stages, which still work on boxes.
    start_time = time.perf_counter()
    initial_candidates = generate_label_candidates(G, lattice, cfg)
    obstacles = build_obstacles(G, lattice, alpha_shape)
    sliding_labels = prepare_labels(
        G,
        [(lid, cands[0].node_id, cands[0].label_type, cands[0].text) for lid, cands in initial_candidates.items()],
        obstacles,
        cfg
    )
    feasible_duration = (time.perf_counter() - start_time) * 1000
    if cfg.runtime:
        print(f"\033[1;36m[RUNTIME]\033[0m Sliders and feasible sets: \033[1;32m{feasible_duration:.2f} ms\033[0m")

    # continuous placement on the sliders
    start_time = time.perf_counter()
    placed = place_labels(sliding_labels, obstacles)
    label_candidates = {
        lid: [LabelCandidate.from_candidate(candidate)] if candidate is not None else []
        for lid, candidate in placed.items()
    }
    placement_duration = (time.perf_counter() - start_time) * 1000
    if cfg.runtime:
        print(f"\033[1;36m[RUNTIME]\033[0m Sliding placement: \033[1;32m{placement_duration:.2f} ms\033[0m")
    if cfg.plot:
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/placed.pdf',
            label_candidates=label_candidates,
            colored_label_candidates=True
        )

    overflow_candidates = generate_overflow_candidates(G, label_candidates, initial_candidates)
    if cfg.plot:
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/all_overflow_candidates.pdf',
            label_candidates=label_candidates,
            colored_label_candidates=True,
            overflow_candidates=overflow_candidates
        )

    start_time = time.perf_counter()
    overflow_candidates = bounded_overflow_labels(G, label_candidates, overflow_candidates, bounded_faces, centroids, alpha_shape)
    bounded_overflow_duration = (time.perf_counter() - start_time) * 1000
    if cfg.runtime:
        print(f"\033[1;36m[RUNTIME]\033[0m Bounded overflow runtime: \033[1;32m{bounded_overflow_duration:.2f} ms\033[0m")
    if cfg.plot:
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/bounded_overflow_candidates.pdf',
            label_candidates=label_candidates,
            colored_label_candidates=True,
            overflow_candidates=overflow_candidates
        )

    unbounded: List[int] = [lid for lid, ol in overflow_candidates.items() if ol.anchor.anchor_type == AnchorType.O]
    grid_candidates, overflow_candidates, grid_duration, hungarian_duration = unbounded_overflow_labels(G, label_candidates, overflow_candidates, alpha_shape, cfg)
    if cfg.runtime:
        print(f"\033[1;36m[RUNTIME]\033[0m Grid creation runtime: \033[1;32m{grid_duration:.2f} ms\033[0m")
        print(f"\033[1;36m[RUNTIME]\033[0m Hungarian solver runtime: \033[1;32m{hungarian_duration:.2f} ms\033[0m")
    if cfg.plot:
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/unbounded_grid_candidates.pdf',
            label_candidates=label_candidates,
            colored_label_candidates=True,
            overflow_candidates=overflow_candidates,
            grid_candidates=grid_candidates
        )
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/unbounded_overflow_candidates.pdf',
            label_candidates=label_candidates,
            colored_label_candidates=True,
            overflow_candidates=overflow_candidates
        )

    start_time = time.perf_counter()
    overflow_candidates = optimize_overflow_labels(G, label_candidates, overflow_candidates, unbounded, alpha_shape, cfg)
    force_duration = (time.perf_counter() - start_time) * 1000

    for ol in overflow_candidates.values():
        ol.tighten_anchor()
    if cfg.runtime:
        print(f"\033[1;36m[RUNTIME]\033[0m Force refinement runtime: \033[1;32m{force_duration:.2f} ms\033[0m")
    if cfg.plot:
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/force_refined.pdf',
            label_candidates=label_candidates,
            colored_label_candidates=True,
            overflow_candidates=overflow_candidates
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
            label_candidates=label_candidates,
            overflow_candidates=overflow_candidates
        )
        return None

    labels: Dict[int, LabelCandidate] = {
        **{lid: candidates[0] for lid, candidates in label_candidates.items() if candidates},
        **overflow_candidates
    }
    return labels

import time

from pathlib import Path
from typing import Dict, List, Optional

from node_labeling.utils.config import Config
from node_labeling.v2.utils.visualize import plot_graph
from node_labeling.utils.normalize import *
from node_labeling.utils.graph_lattice import GraphLattice
from node_labeling.v2.models.label_candidate import LabelCandidate
from node_labeling.topology.planarize import *
from node_labeling.topology.faces import *
from node_labeling.v2.filter.shape_rules import build_obstacles
from node_labeling.v2.label.sliding import collect_labels, prepare_labels, place_labels
from node_labeling.v2.models.candidate import Candidate
from node_labeling.v2.overflow.binders import Layout
from node_labeling.v2.overflow.bounded import bounded_overflow_labels
from node_labeling.v2.overflow.unbounded import unbounded_overflow_labels
from node_labeling.v2.forces.forces import optimize_overflow_labels

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
    layout = Layout.build(G, obstacles, list(direct.values()))
    placement_duration = (time.perf_counter() - start_time) * 1000
    if cfg.runtime:
        print(f"\033[1;36m[RUNTIME]\033[0m Sliding placement: \033[1;32m{placement_duration:.2f} ms\033[0m")
    if cfg.plot:
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/placed.pdf',
            label_candidates=_label_candidates(direct, overflow),
            colored_label_candidates=True
        )
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/all_overflow_candidates.pdf',
            label_candidates=_label_candidates(direct, overflow),
            colored_label_candidates=True,
            overflow_candidates=_overflow_candidates(overflow)
        )

    start_time = time.perf_counter()
    overflow = bounded_overflow_labels(layout, overflow, bounded_faces, centroids, alpha_shape)
    bounded_overflow_duration = (time.perf_counter() - start_time) * 1000
    if cfg.runtime:
        print(f"\033[1;36m[RUNTIME]\033[0m Bounded overflow runtime: \033[1;32m{bounded_overflow_duration:.2f} ms\033[0m")
    if cfg.plot:
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/bounded_overflow_candidates.pdf',
            label_candidates=_label_candidates(direct, overflow),
            colored_label_candidates=True,
            overflow_candidates=_overflow_candidates(overflow)
        )

    unbounded = [lid for lid, candidate in overflow.items() if candidate.binder is None]
    grid_candidates, overflow, grid_duration, hungarian_duration = unbounded_overflow_labels(layout, overflow, alpha_shape, cfg)
    if cfg.runtime:
        print(f"\033[1;36m[RUNTIME]\033[0m Grid creation runtime: \033[1;32m{grid_duration:.2f} ms\033[0m")
        print(f"\033[1;36m[RUNTIME]\033[0m Hungarian solver runtime: \033[1;32m{hungarian_duration:.2f} ms\033[0m")
    if cfg.plot:
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/unbounded_grid_candidates.pdf',
            label_candidates=_label_candidates(direct, overflow),
            colored_label_candidates=True,
            overflow_candidates=_overflow_candidates(overflow),
            grid_candidates={
                lid: [(LabelCandidate.from_candidate(c), cost) for c, cost in cands]
                for lid, cands in grid_candidates.items()
            }
        )
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/unbounded_overflow_candidates.pdf',
            label_candidates=_label_candidates(direct, overflow),
            colored_label_candidates=True,
            overflow_candidates=_overflow_candidates(overflow)
        )

    start_time = time.perf_counter()
    overflow = optimize_overflow_labels(layout, overflow, unbounded, alpha_shape, cfg)
    force_duration = (time.perf_counter() - start_time) * 1000
    if cfg.runtime:
        print(f"\033[1;36m[RUNTIME]\033[0m Force refinement runtime: \033[1;32m{force_duration:.2f} ms\033[0m")

    label_candidates = _label_candidates(direct, overflow)
    overflow_candidates = _overflow_candidates(overflow)
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

def _label_candidates(direct: Dict[int, Candidate], overflow: Dict[int, Candidate]) -> Dict[int, List[LabelCandidate]]:
    '''
    Parameters
    ----------
    direct : Dict[int, Candidate]
        labels placed at their nodes
    overflow : Dict[int, Candidate]
        overflow labels

    Returns
    -------
    label_candidates : Dict[int, List[LabelCandidate]]
        placed labels for plotting and the return value (empty list for overflow labels)
    '''
    out: Dict[int, List[LabelCandidate]] = {lid: [LabelCandidate.from_candidate(c)] for lid, c in direct.items()}
    out.update({lid: [] for lid in overflow})
    return dict(sorted(out.items()))

def _overflow_candidates(overflow: Dict[int, Candidate]) -> Dict[int, LabelCandidate]:
    '''
    Parameters
    ----------
    overflow : Dict[int, Candidate]
        overflow labels

    Returns
    -------
    overflow_candidates : Dict[int, LabelCandidate]
        overflow labels for plotting and the return value
    '''
    return {lid: LabelCandidate.from_candidate(c) for lid, c in sorted(overflow.items())}
